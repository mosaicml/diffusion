# Copyright 2022 MosaicML Diffusion authors
# SPDX-License-Identifier: Apache-2.0

"""Constructors for diffusion models."""

from typing import List, Optional

import torch
from composer.devices import DeviceGPU
from diffusers import AutoencoderKL, DDIMScheduler, DDPMScheduler, EulerDiscreteScheduler, UNet2DConditionModel
from torchmetrics import MeanSquaredError
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.multimodal.clip_score import CLIPScore
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer, PretrainedConfig

from diffusion.models.layers import ClippedAttnProcessor2_0, ClippedXFormersAttnProcessor, zero_module
from diffusion.models.pixel_diffusion import PixelDiffusion
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
    prediction_type: str = 'epsilon',
    train_metrics: Optional[List] = None,
    val_metrics: Optional[List] = None,
    val_guidance_scales: Optional[List] = None,
    val_seed: int = 1138,
    loss_bins: Optional[List] = None,
    precomputed_latents: bool = False,
    encode_latents_in_fp16: bool = True,
    fsdp: bool = True,
    clip_qkv: Optional[float] = None,
):
    """Stable diffusion v2 training setup.

    Requires batches of matched images and text prompts to train. Generates images from text
    prompts.

    Args:
        model_name (str): Name of the model to load. Defaults to 'stabilityai/stable-diffusion-2-base'.
        pretrained (bool): Whether to load pretrained weights. Defaults to True.
        prediction_type (str): The type of prediction to use. Must be one of 'sample',
            'epsilon', or 'v_prediction'. Default: `epsilon`.
        train_metrics (list, optional): List of metrics to compute during training. If None, defaults to
            [MeanSquaredError()].
        val_metrics (list, optional): List of metrics to compute during validation. If None, defaults to
            [MeanSquaredError(), FrechetInceptionDistance(normalize=True)].
        val_guidance_scales (list, optional): List of scales to use for validation guidance. If None, defaults to
            [1.0, 3.0, 7.0].
        val_seed (int): Seed to use for generating evaluation images. Defaults to 1138.
        loss_bins (list, optional): List of tuples of (min, max) values to use for loss binning. If None, defaults to
            [(0, 1)].
        precomputed_latents (bool): Whether to use precomputed latents. Defaults to False.
        encode_latents_in_fp16 (bool): Whether to encode latents in fp16. Defaults to True.
        fsdp (bool): Whether to use FSDP. Defaults to True.
        clip_qkv (float, optional): If not None, clip the qkv values to this value. Defaults to None.
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
    inference_noise_scheduler = DDIMScheduler(num_train_timesteps=noise_scheduler.config.num_train_timesteps,
                                              beta_start=noise_scheduler.config.beta_start,
                                              beta_end=noise_scheduler.config.beta_end,
                                              beta_schedule=noise_scheduler.config.beta_schedule,
                                              trained_betas=noise_scheduler.config.trained_betas,
                                              clip_sample=noise_scheduler.config.clip_sample,
                                              set_alpha_to_one=noise_scheduler.config.set_alpha_to_one,
                                              prediction_type=prediction_type)

    model = StableDiffusion(
        unet=unet,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        noise_scheduler=noise_scheduler,
        inference_noise_scheduler=inference_noise_scheduler,
        prediction_type=prediction_type,
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

    if clip_qkv is not None:
        if is_xformers_installed:
            attn_processor = ClippedXFormersAttnProcessor(clip_val=clip_qkv)
        else:
            attn_processor = ClippedAttnProcessor2_0(clip_val=clip_qkv)
        model.unet.set_attn_processor(attn_processor)

    return model


def stable_diffusion_xl(
    model_name: str = 'stabilityai/stable-diffusion-xl-base-1.0',
    unet_model_name: str = 'stabilityai/stable-diffusion-xl-base-1.0',
    vae_model_name: str = 'madebyollin/sdxl-vae-fp16-fix',
    pretrained: bool = True,
    prediction_type: str = 'epsilon',
    train_metrics: Optional[List] = None,
    val_metrics: Optional[List] = None,
    val_guidance_scales: Optional[List] = None,
    val_seed: int = 1138,
    loss_bins: Optional[List] = None,
    precomputed_latents: bool = False,
    encode_latents_in_fp16: bool = True,
    fsdp: bool = True,
    clip_qkv: Optional[float] = 6.0,
):
    """Stable diffusion 2 training setup + SDXL UNet and VAE.

    Requires batches of matched images and text prompts to train. Generates images from text
    prompts. Currently uses UNet and VAE config from SDXL, but text encoder/tokenizer from SD2.

    Args:
        model_name (str): Name of the model to load. Determines the text encoders, tokenizers,
            and noise scheduler. Defaults to 'stabilityai/stable-diffusion-xl-base-1.0'.
        unet_model_name (str): Name of the UNet model to load. Defaults to
            'stabilityai/stable-diffusion-xl-base-1.0'.
        vae_model_name (str): Name of the VAE model to load. Defaults to
            'madebyollin/sdxl-vae-fp16-fix' as the official VAE checkpoint (from
            'stabilityai/stable-diffusion-xl-base-1.0') is not compatible with fp16.
        pretrained (bool): Whether to load pretrained weights. Defaults to True.
        prediction_type (str): The type of prediction to use. Must be one of 'sample',
            'epsilon', or 'v_prediction'. Default: `epsilon`.
        train_metrics (list, optional): List of metrics to compute during training. If None, defaults to
            [MeanSquaredError()].
        val_metrics (list, optional): List of metrics to compute during validation. If None, defaults to
            [MeanSquaredError(), FrechetInceptionDistance(normalize=True)].
        val_guidance_scales (list, optional): List of scales to use for validation guidance. If None, defaults to
            [1.0, 3.0, 7.0].
        val_seed (int): Seed to use for generating evaluation images. Defaults to 1138.
        loss_bins (list, optional): List of tuples of (min, max) values to use for loss binning. If None, defaults to
            [(0, 1)].
        precomputed_latents (bool): Whether to use precomputed latents. Defaults to False.
        encode_latents_in_fp16 (bool): Whether to encode latents in fp16. Defaults to True.
        fsdp (bool): Whether to use FSDP. Defaults to True.
        clip_qkv (float, optional): If not None, clip the qkv values to this value. Defaults to 6.0. Improves stability
            of training.
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
        raise NotImplementedError('Full SDXL pipeline not implemented yet.')
    else:
        config = PretrainedConfig.get_config_dict(unet_model_name, subfolder='unet')
        unet = UNet2DConditionModel(**config[0])

        # Zero initialization trick for more stable training
        for name, layer in unet.named_modules():
            # Final conv in ResNet blocks
            if name.endswith('conv2'):
                layer = zero_module(layer)
            # proj_out in attention blocks
            if name.endswith('to_out.0'):
                layer = zero_module(layer)
        # Last conv block out projection
        unet.conv_out = zero_module(unet.conv_out)

    if encode_latents_in_fp16:
        try:
            vae = AutoencoderKL.from_pretrained(vae_model_name, subfolder='vae', torch_dtype=torch.float16)
        except:  # for handling SDXL vae fp16 fixed checkpoint
            vae = AutoencoderKL.from_pretrained(vae_model_name, torch_dtype=torch.float16)
    else:
        try:
            vae = AutoencoderKL.from_pretrained(vae_model_name, subfolder='vae')
        except:  #  for handling SDXL vae fp16 fixed checkpoint
            vae = AutoencoderKL.from_pretrained(vae_model_name)

    tokenizer = SDXLTokenizer(model_name)
    text_encoder = SDXLTextEncoder(model_name, encode_latents_in_fp16)

    noise_scheduler = DDPMScheduler.from_pretrained(model_name, subfolder='scheduler')
    inference_noise_scheduler = EulerDiscreteScheduler.from_pretrained(model_name, subfolder='scheduler')

    model = StableDiffusion(
        unet=unet,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        noise_scheduler=noise_scheduler,
        inference_noise_scheduler=inference_noise_scheduler,
        prediction_type=prediction_type,
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        val_guidance_scales=val_guidance_scales,
        val_seed=val_seed,
        loss_bins=loss_bins,
        precomputed_latents=precomputed_latents,
        encode_latents_in_fp16=encode_latents_in_fp16,
        fsdp=fsdp,
        sdxl=True,
    )
    if torch.cuda.is_available():
        model = DeviceGPU().module_to_device(model)
        if is_xformers_installed:
            model.unet.enable_xformers_memory_efficient_attention()
            model.vae.enable_xformers_memory_efficient_attention()

    if clip_qkv is not None:
        if is_xformers_installed:
            attn_processor = ClippedXFormersAttnProcessor(clip_val=clip_qkv)
        else:
            attn_processor = ClippedAttnProcessor2_0(clip_val=clip_qkv)
        model.unet.set_attn_processor(attn_processor)

    return model


def discrete_pixel_diffusion(clip_model_name: str = 'openai/clip-vit-large-patch14', prediction_type='epsilon'):
    """Discrete pixel diffusion training setup.

    Args:
        clip_model_name (str, optional): Name of the clip model to load. Defaults to 'openai/clip-vit-large-patch14'.
        prediction_type (str, optional): Type of prediction to use. One of 'sample', 'epsilon', 'v_prediction'.
            Defaults to 'epsilon'.
    """
    # Create a pixel space unet
    unet = UNet2DConditionModel(in_channels=3,
                                out_channels=3,
                                attention_head_dim=[5, 10, 20, 20],
                                cross_attention_dim=768,
                                flip_sin_to_cos=True,
                                use_linear_projection=True)
    # Get the CLIP text encoder and tokenizer:
    text_encoder = CLIPTextModel.from_pretrained(clip_model_name)
    tokenizer = CLIPTokenizer.from_pretrained(clip_model_name)
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
    model = PixelDiffusion(unet,
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


def continuous_pixel_diffusion(clip_model_name: str = 'openai/clip-vit-large-patch14',
                               prediction_type='epsilon',
                               use_ode=False,
                               train_t_max=1.570795,
                               inference_t_max=1.56):
    """Continuous pixel diffusion training setup.

    Uses the same clip and unet config as `discrete_pixel_diffusion`, but operates in continous time as in the VP
    process in https://arxiv.org/abs/2011.13456.

    Args:
        clip_model_name (str, optional): Name of the clip model to load. Defaults to 'openai/clip-vit-large-patch14'.
        prediction_type (str, optional): Type of prediction to use. One of 'sample', 'epsilon', 'v_prediction'.
            Defaults to 'epsilon'.
        use_ode (bool, optional): Whether to do generation using the probability flow ODE. If not used, uses the
            reverse diffusion process. Defaults to False.
        train_t_max (float, optional): Maximum timestep during training. Defaults to 1.570795 (pi/2).
        inference_t_max (float, optional): Maximum timestep during inference.
            Defaults to 1.56 (pi/2 - 0.01 for stability).
    """
    # Create a pixel space unet
    unet = UNet2DConditionModel(in_channels=3,
                                out_channels=3,
                                attention_head_dim=[5, 10, 20, 20],
                                cross_attention_dim=768,
                                flip_sin_to_cos=True,
                                use_linear_projection=True)
    # Get the CLIP text encoder and tokenizer:
    text_encoder = CLIPTextModel.from_pretrained(clip_model_name)
    tokenizer = CLIPTokenizer.from_pretrained(clip_model_name)
    # Need to use the continuous time schedulers for training and inference.
    noise_scheduler = ContinuousTimeScheduler(t_max=train_t_max, prediction_type=prediction_type)
    inference_scheduler = ContinuousTimeScheduler(t_max=inference_t_max,
                                                  prediction_type=prediction_type,
                                                  use_ode=use_ode)

    # Create the pixel space diffusion model
    model = PixelDiffusion(unet,
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


class SDXLTextEncoder(torch.nn.Module):
    """Wrapper around HuggingFace text encoders for SDXL.

    Creates two text encoders (a CLIPTextModel and CLIPTextModelWithProjection) that behave like one.

    Args:
        model_name (str): Name of the model's text encoders to load. Defaults to 'stabilityai/stable-diffusion-xl-base-1.0'.
        encode_latents_in_fp16 (bool): Whether to encode latents in fp16. Defaults to True.
    """

    def __init__(self, model_name='stabilityai/stable-diffusion-xl-base-1.0', encode_latents_in_fp16=True):
        super().__init__()
        torch_dtype = torch.float16 if encode_latents_in_fp16 else None
        self.text_encoder = CLIPTextModel.from_pretrained(model_name, subfolder='text_encoder', torch_dtype=torch_dtype)
        self.text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(model_name,
                                                                          subfolder='text_encoder_2',
                                                                          torch_dtype=torch_dtype)

    @property
    def device(self):
        return self.text_encoder.device
    
    def forward(self, tokenized_text):
        # first text encoder
        conditioning = self.text_encoder(tokenized_text[0], output_hidden_states=True).hidden_states[-2]
        # second text encoder
        text_encoder_2_out = self.text_encoder_2(tokenized_text[1], output_hidden_states=True)
        pooled_conditioning = text_encoder_2_out[0]  # (batch_size, 1280)
        conditioning_2 = text_encoder_2_out.hidden_states[-2]  # (batch_size, 77, 1280)

        # # zero out the appropriate things
        # if batch[self.text_key].sum() == 0:
        #     conditioning = torch.zeros_like(conditioning)
        # if batch[self.text_key_2].sum() == 0:
        #     conditioning_2 = torch.zeros_like(conditioning_2)
        #     pooled_conditioning = torch.zeros_like(pooled_conditioning)

        conditioning = torch.concat([conditioning, conditioning_2], dim=-1)
        return conditioning, pooled_conditioning


class SDXLTokenizer:
    """Wrapper around HuggingFace tokenizers for SDXL.

    Tokenizes prompt with two tokenizers and returns the outputs as a list.

    Args:
        model_name (str): Name of the model's text encoders to load. Defaults to 'stabilityai/stable-diffusion-xl-base-1.0'.
    """

    def __init__(self, model_name='stabilityai/stable-diffusion-xl-base-1.0'):
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name, subfolder='tokenizer')
        self.tokenizer_2 = CLIPTokenizer.from_pretrained(model_name, subfolder='tokenizer_2')

    def __call__(self, prompt, padding, truncation, return_tensors, input_ids=False):
        tokenized_output = self.tokenizer(prompt,
                                          padding=padding,
                                          max_length=self.tokenizer.model_max_length,
                                          truncation=truncation,
                                          return_tensors=return_tensors)
        tokenized_output_2 = self.tokenizer_2(prompt,
                                              padding=padding,
                                              max_length=self.tokenizer_2.model_max_length,
                                              truncation=truncation,
                                              return_tensors=return_tensors)
        if input_ids:
            tokenized_output = tokenized_output.input_ids
            tokenized_output_2 = tokenized_output_2.input_ids
        return [tokenized_output, tokenized_output_2]
