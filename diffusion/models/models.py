# Copyright 2022 MosaicML Diffusion authors
# SPDX-License-Identifier: Apache-2.0

"""Constructors for diffusion models."""

from typing import List, Optional

import torch
from composer.devices import DeviceGPU
from diffusers import AutoencoderKL, DDIMScheduler, DDPMScheduler, UNet2DConditionModel
from torchmetrics import MeanSquaredError
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.multimodal.clip_score import CLIPScore
from transformers import CLIPTextModel, CLIPTokenizer, PretrainedConfig

from diffusion.models.pixel_diffusion import PixelSpaceDiffusion
from diffusion.models.stable_diffusion import StableDiffusion
from diffusion.schedulers.schedulers import ContinuousTimeScheduler

try:
    import xformers  # type: ignore
    del xformers
    is_xformers_installed = True
except:
    is_xformers_installed = False


def stable_diffusion_2(
    model_name: str = 'stabilityai/stable-diffusion-2-base',
    pretrained: bool = True,
    train_metrics: Optional[List] = None,
    val_metrics: Optional[List] = None,
    val_guidance_scales: Optional[List] = None,
    val_seed: int = 1138,
    loss_bins: Optional[List] = None,
    precomputed_latents: bool = False,
    encode_latents_in_fp16: bool = True,
    fsdp: bool = True,
):
    """Stable diffusion v2 training setup.

    Requires batches of matched images and text prompts to train. Generates images from text
    prompts.

    Args:
        model_name (str, optional): Name of the model to load. Defaults to 'stabilityai/stable-diffusion-2-base'.
        pretrained (bool, optional): Whether to load pretrained weights. Defaults to True.
        train_metrics (list, optional): List of metrics to compute during training. If None, defaults to
            [MeanSquaredError()].
        val_metrics (list, optional): List of metrics to compute during validation. If None, defaults to
            [MeanSquaredError(), FrechetInceptionDistance(normalize=True)].
        val_guidance_scales (list, optional): List of scales to use for validation guidance. If None, defaults to
            [1.0, 3.0, 7.0].
        val_seed (int, optional): Seed to use for generating evaluation images. Defaults to 1138.
        loss_bins (list, optional): List of tuples of (min, max) values to use for loss binning. If None, defaults to
            [(0, 1)].
        precomputed_latents (bool, optional): Whether to use precomputed latents. Defaults to False.
        encode_latents_in_fp16 (bool, optional): Whether to encode latents in fp16. Defaults to True.
        fsdp (bool, optional): Whether to use FSDP. Defaults to True.
    """
    if train_metrics is None:
        train_metrics = [MeanSquaredError()]
    if val_metrics is None:
        val_metrics = [MeanSquaredError(), FrechetInceptionDistance(normalize=True)]
    if val_guidance_scales is None:
        val_guidance_scales = [1.0, 3.0, 7.0]
    if loss_bins is None:
        loss_bins = [(0, 1)]
    # Fix a bug where CLIPScore requires grad
    for metric in val_metrics:
        if isinstance(metric, CLIPScore):
            metric.requires_grad_(False)

    if pretrained:
        unet = UNet2DConditionModel.from_pretrained(model_name, subfolder='unet')
    else:
        config = PretrainedConfig.get_config_dict(model_name, subfolder='unet')
        unet = UNet2DConditionModel(**config[0])

    if encode_latents_in_fp16:
        vae = AutoencoderKL.from_pretrained(model_name, subfolder='vae', torch_dtype=torch.float16)
        text_encoder = CLIPTextModel.from_pretrained(model_name, subfolder='text_encoder', torch_dtype=torch.float16)
    else:
        vae = AutoencoderKL.from_pretrained(model_name, subfolder='vae')
        text_encoder = CLIPTextModel.from_pretrained(model_name, subfolder='text_encoder')

    tokenizer = CLIPTokenizer.from_pretrained(model_name, subfolder='tokenizer')
    noise_scheduler = DDPMScheduler.from_pretrained(model_name, subfolder='scheduler')
    inference_noise_scheduler = DDIMScheduler.from_pretrained(model_name, subfolder='scheduler')

    model = StableDiffusion(
        unet=unet,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        noise_scheduler=noise_scheduler,
        inference_noise_scheduler=inference_noise_scheduler,
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        val_guidance_scales=val_guidance_scales,
        val_seed=val_seed,
        loss_bins=loss_bins,
        precomputed_latents=precomputed_latents,
        encode_latents_in_fp16=encode_latents_in_fp16,
        fsdp=fsdp,
    )
    if torch.cuda.is_available():
        model = DeviceGPU().module_to_device(model)
        if is_xformers_installed:
            model.unet.enable_xformers_memory_efficient_attention()
            model.vae.enable_xformers_memory_efficient_attention()
    return model


def discrete_pixel_diffusion(model_name: str = 'stabilityai/stable-diffusion-2-base', prediction_type='epsilon'):
    """Discrete pixel diffusion training setup.

    Uses the same clip and unet config as stable diffusion, but operates in pixel space rather than latent space.

    Args:
        model_name (str, optional): Name of the model config to load. Defaults to 'stabilityai/stable-diffusion-2-base'.
        prediction_type (str, optional): Type of prediction to use. One of 'sample', 'epsilon', 'v_prediction'. Defaults to 'epsilon'.
    """
    # Get the stable diffusion 2 unet config
    config = PretrainedConfig.get_config_dict(model_name, subfolder='unet')
    # Set the number of channels to 3
    config[0]['in_channels'] = 3
    # Set the number of out channels to 3
    config[0]['out_channels'] = 3
    # Create the pixel space unet based on the SD2 unet.
    unet = UNet2DConditionModel(**config[0])
    # Get the SD2 text encoder and tokenizer:
    text_encoder = CLIPTextModel.from_pretrained(model_name, subfolder='text_encoder')
    tokenizer = CLIPTokenizer.from_pretrained(model_name, subfolder='tokenizer')
    # Hard code the sheduler config
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000,
                                    beta_start=0.00085,
                                    beta_end=0.012,
                                    beta_schedule='scaled_linear',
                                    trained_betas=None,
                                    variance_type='fixed_small',
                                    clip_sample=False,
                                    prediction_type=prediction_type,
                                    thresholding=False,
                                    dynamic_thresholding_ratio=0.995,
                                    clip_sample_range=1.0,
                                    sample_max_value=1.0)
    inference_scheduler = DDIMScheduler(num_train_timesteps=1000,
                                        beta_start=0.00085,
                                        beta_end=0.012,
                                        beta_schedule='scaled_linear',
                                        trained_betas=None,
                                        clip_sample=False,
                                        set_alpha_to_one=False,
                                        steps_offset=1,
                                        prediction_type=prediction_type,
                                        thresholding=False,
                                        dynamic_thresholding_ratio=0.995,
                                        clip_sample_range=1.0,
                                        sample_max_value=1.0)

    # Create the pixel space diffusion model
    model = PixelSpaceDiffusion(unet,
                                text_encoder,
                                tokenizer,
                                noise_scheduler,
                                inference_scheduler=inference_scheduler,
                                prediction_type=prediction_type,
                                train_metrics=[MeanSquaredError()],
                                val_metrics=[MeanSquaredError()])

    if torch.cuda.is_available():
        model = DeviceGPU().module_to_device(model)
        if is_xformers_installed:
            model.model.enable_xformers_memory_efficient_attention()
    return model


def continuous_pixel_diffusion(model_name: str = 'stabilityai/stable-diffusion-2-base',
                               prediction_type='epsilon',
                               use_ode=False,
                               train_t_max=1.570795,
                               inference_t_max=1.56):
    """Continuous pixel diffusion training setup.

    Uses the same clip and unet config as stable diffusion, but operates in pixel space rather than latent space. Uses the continuous time parameterization as in the VP process in https://arxiv.org/abs/2011.13456.

    Args:
        model_name (str, optional): Name of the model config to load. Defaults to 'stabilityai/stable-diffusion-2-base'.
        prediction_type (str, optional): Type of prediction to use. One of 'sample', 'epsilon', 'v_prediction'. Defaults to 'epsilon'.
        use_ode (bool, optional): Whether to do generation using the probability flow ODE. If not used, uses the reverse diffusion process. Defaults to False.
        train_t_max (float, optional): Maximum timestep during training. Defaults to 1.570795 (pi/2).
        inference_t_max (float, optional): Maximum timestep during inference. Defaults to 1.56 (pi/2 - 0.01 for stability).
    """
    # Get the stable diffusion 2 unet config
    config = PretrainedConfig.get_config_dict(model_name, subfolder='unet')
    # Set the number of channels to 3
    config[0]['in_channels'] = 3
    # Set the number of out channels to 3
    config[0]['out_channels'] = 3
    # Create the pixel space unet based on the SD2 unet.
    unet = UNet2DConditionModel(**config[0])
    # Get the SD2 text encoder and tokenizer:
    text_encoder = CLIPTextModel.from_pretrained(model_name, subfolder='text_encoder')
    tokenizer = CLIPTokenizer.from_pretrained(model_name, subfolder='tokenizer')
    # Hard code the sheduler config
    noise_scheduler = ContinuousTimeScheduler(t_max=train_t_max, prediction_type=prediction_type)
    inference_scheduler = ContinuousTimeScheduler(t_max=inference_t_max,
                                                  prediction_type=prediction_type,
                                                  use_ode=use_ode)

    # Create the pixel space diffusion model
    model = PixelSpaceDiffusion(unet,
                                text_encoder,
                                tokenizer,
                                noise_scheduler,
                                inference_scheduler=inference_scheduler,
                                prediction_type=prediction_type,
                                continuous_time=True,
                                train_metrics=[MeanSquaredError()],
                                val_metrics=[MeanSquaredError()])

    if torch.cuda.is_available():
        model = DeviceGPU().module_to_device(model)
        if is_xformers_installed:
            model.model.enable_xformers_memory_efficient_attention()
    return model
