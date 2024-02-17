# Copyright 2022 MosaicML Diffusion authors
# SPDX-License-Identifier: Apache-2.0

"""Constructors for diffusion models."""

import logging
import math
from typing import List, Optional, Tuple, Union

import torch
from composer.devices import DeviceGPU
from diffusers import AutoencoderKL, DDIMScheduler, DDPMScheduler, EulerDiscreteScheduler, UNet2DConditionModel
from torchmetrics import MeanSquaredError
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.multimodal.clip_score import CLIPScore
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer, PretrainedConfig

from diffusion.models.autoencoder import (AutoEncoder, AutoEncoderLoss, ComposerAutoEncoder,
                                          ComposerDiffusersAutoEncoder, load_autoencoder)
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

log = logging.getLogger(__name__)


def stable_diffusion_2(
    model_name: str = 'stabilityai/stable-diffusion-2-base',
    pretrained: bool = True,
    autoencoder_path: Optional[str] = None,
    autoencoder_local_path: str = '/tmp/autoencoder_weights.pt',
    prediction_type: str = 'epsilon',
    latent_scale: Union[float, str] = 0.18215,
    offset_noise: Optional[float] = None,
    train_metrics: Optional[List] = None,
    val_metrics: Optional[List] = None,
    val_guidance_scales: Optional[List] = None,
    val_seed: int = 1138,
    loss_bins: Optional[List] = None,
    precomputed_latents: bool = False,
    encode_latents_in_fp16: bool = True,
    mask_pad_tokens: bool = False,
    fsdp: bool = True,
    clip_qkv: Optional[float] = None,
    use_xformers: bool = True,
):
    """Stable diffusion v2 training setup.

    Requires batches of matched images and text prompts to train. Generates images from text
    prompts.

    Args:
        model_name (str): Name of the model to load. Defaults to 'stabilityai/stable-diffusion-2-base'.
        pretrained (bool): Whether to load pretrained weights. Defaults to True.
        autoencoder_path (optional, str): Path to autoencoder weights if using custom autoencoder. If not specified,
            will use the vae from `model_name`. Default `None`.
        autoencoder_local_path (optional, str): Path to autoencoder weights. Default: `/tmp/autoencoder_weights.pt`.
        prediction_type (str): The type of prediction to use. Must be one of 'sample',
            'epsilon', or 'v_prediction'. Default: `epsilon`.
        latent_scale (float, str): The scale of the autoencoder latents. Either a float for the scaling value,
            or `'latent_statistics'` to try to use the value from the autoencoder checkpoint. Defaults to `0.18215`.
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
        offset_noise (float, optional): The scale of the offset noise. If not specified, offset noise will not
            be used. Default `None`.
        encode_latents_in_fp16 (bool): Whether to encode latents in fp16. Defaults to True.
        mask_pad_tokens (bool): Whether to mask pad tokens in cross attention. Defaults to False.
        fsdp (bool): Whether to use FSDP. Defaults to True.
        clip_qkv (float, optional): If not None, clip the qkv values to this value. Defaults to None.
        use_xformers (bool): Whether to use xformers for attention. Defaults to True.
    """
    if isinstance(latent_scale, str):
        latent_scale = latent_scale.lower()
        if latent_scale != 'latent_statistics':
            raise ValueError(f'Invalid latent scale {latent_scale}. Must be a float or "latent_statistics".')

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

    precision = torch.float16 if encode_latents_in_fp16 else None
    # Make the text encoder
    text_encoder = CLIPTextModel.from_pretrained(model_name, subfolder='text_encoder', torch_dtype=precision)
    tokenizer = CLIPTokenizer.from_pretrained(model_name, subfolder='tokenizer')

    # Make the autoencoder
    if autoencoder_path is None:
        if latent_scale == 'latent_statistics':
            raise ValueError('Cannot use tracked latent_statistics when using the pretrained vae.')
        # Use the pretrained vae
        downsample_factor = 8
        vae = AutoencoderKL.from_pretrained(model_name, subfolder='vae', torch_dtype=precision)
    else:
        # Use a custom autoencoder
        vae, latent_statistics = load_autoencoder(autoencoder_path, autoencoder_local_path, torch_dtype=precision)
        if latent_statistics is not None and latent_scale == 'latent_statistics':
            assert isinstance(latent_statistics['global_mean'], float)
            assert isinstance(latent_statistics['global_std'], float)
            second_moment = latent_statistics['global_mean']**2 + latent_statistics['global_std']**2
            latent_scale = 1 / math.sqrt(second_moment)
        else:
            raise ValueError(
                'Must specify latent scale when using a custom autoencoder without tracking latent statistics.')
        downsample_factor = 2**(len(vae.config['channel_multipliers']) - 1)

    # Make the unet
    if pretrained:
        unet = UNet2DConditionModel.from_pretrained(model_name, subfolder='unet')
        if autoencoder_path is not None and vae.config['latent_channels'] != 4:
            raise ValueError(f'Pretrained unet has 4 latent channels but the vae has {vae.latent_channels}.')
    else:
        unet_config = PretrainedConfig.get_config_dict(model_name, subfolder='unet')[0]
        if autoencoder_path is not None:
            # Adapt the unet config to account for differing number of latent channels if necessary
            unet_config['in_channels'] = vae.config['latent_channels']
            unet_config['out_channels'] = vae.config['latent_channels']
        # Init the unet from the config
        unet = UNet2DConditionModel(**unet_config)

    # Make the noise schedulers
    noise_scheduler = DDPMScheduler.from_pretrained(model_name, subfolder='scheduler')
    inference_noise_scheduler = DDIMScheduler(num_train_timesteps=noise_scheduler.config.num_train_timesteps,
                                              beta_start=noise_scheduler.config.beta_start,
                                              beta_end=noise_scheduler.config.beta_end,
                                              beta_schedule=noise_scheduler.config.beta_schedule,
                                              trained_betas=noise_scheduler.config.trained_betas,
                                              clip_sample=noise_scheduler.config.clip_sample,
                                              set_alpha_to_one=noise_scheduler.config.set_alpha_to_one,
                                              prediction_type=prediction_type)

    # Make the composer model
    model = StableDiffusion(
        unet=unet,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        noise_scheduler=noise_scheduler,
        inference_noise_scheduler=inference_noise_scheduler,
        prediction_type=prediction_type,
        latent_scale=latent_scale,
        downsample_factor=downsample_factor,
        offset_noise=offset_noise,
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        val_guidance_scales=val_guidance_scales,
        val_seed=val_seed,
        loss_bins=loss_bins,
        precomputed_latents=precomputed_latents,
        encode_latents_in_fp16=encode_latents_in_fp16,
        mask_pad_tokens=mask_pad_tokens,
        fsdp=fsdp,
    )
    if torch.cuda.is_available():
        model = DeviceGPU().module_to_device(model)
        if is_xformers_installed and use_xformers:
            model.unet.enable_xformers_memory_efficient_attention()
            if hasattr(model.vae, 'enable_xformers_memory_efficient_attention'):
                model.vae.enable_xformers_memory_efficient_attention()

    if clip_qkv is not None:
        if is_xformers_installed and use_xformers:
            attn_processor = ClippedXFormersAttnProcessor(clip_val=clip_qkv)
        else:
            attn_processor = ClippedAttnProcessor2_0(clip_val=clip_qkv)
        log.info('Using %s with clip_val %.1f' % (attn_processor.__class__, clip_qkv))
        model.unet.set_attn_processor(attn_processor)

    return model


def stable_diffusion_xl(
    model_name: str = 'stabilityai/stable-diffusion-xl-base-1.0',
    unet_model_name: str = 'stabilityai/stable-diffusion-xl-base-1.0',
    vae_model_name: str = 'madebyollin/sdxl-vae-fp16-fix',
    pretrained: bool = True,
    autoencoder_path: Optional[str] = None,
    autoencoder_local_path: str = '/tmp/autoencoder_weights.pt',
    prediction_type: str = 'epsilon',
    latent_scale: Union[float, str] = 0.13025,
    offset_noise: Optional[float] = None,
    train_metrics: Optional[List] = None,
    val_metrics: Optional[List] = None,
    val_guidance_scales: Optional[List] = None,
    val_seed: int = 1138,
    loss_bins: Optional[List] = None,
    precomputed_latents: bool = False,
    encode_latents_in_fp16: bool = True,
    mask_pad_tokens: bool = False,
    fsdp: bool = True,
    clip_qkv: Optional[float] = 6.0,
    use_xformers: bool = True,
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
        autoencoder_path (optional, str): Path to autoencoder weights if using custom autoencoder. If not specified,
            will use the vae from `model_name`. Default `None`.
        autoencoder_local_path (optional, str): Path to autoencoder weights. Default: `/tmp/autoencoder_weights.pt`.
        prediction_type (str): The type of prediction to use. Must be one of 'sample',
            'epsilon', or 'v_prediction'. Default: `epsilon`.
        latent_scale (float, str): The scale of the autoencoder latents. Either a float for the scaling value,
            or `'latent_statistics'` to try to use the value from the autoencoder checkpoint. Defaults to `0.13025`.
        offset_noise (float, optional): The scale of the offset noise. If not specified, offset noise will not
            be used. Default `None`.
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
        mask_pad_tokens (bool): Whether to mask pad tokens in cross attention. Defaults to False.
        fsdp (bool): Whether to use FSDP. Defaults to True.
        clip_qkv (float, optional): If not None, clip the qkv values to this value. Defaults to 6.0. Improves stability
            of training.
        use_xformers (bool): Whether to use xformers for attention. Defaults to True.
    """
    if isinstance(latent_scale, str):
        latent_scale = latent_scale.lower()
        if latent_scale != 'latent_statistics':
            raise ValueError(f'Invalid latent scale {latent_scale}. Must be a float or "latent_statistics".')
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

    # Make the text encoder
    tokenizer = SDXLTokenizer(model_name)
    text_encoder = SDXLTextEncoder(model_name, encode_latents_in_fp16)

    precision = torch.float16 if encode_latents_in_fp16 else None
    # Make the autoencoder
    if autoencoder_path is None:
        if latent_scale == 'latent_statistics':
            raise ValueError('Cannot use tracked latent_statistics when using the pretrained vae.')
        downsample_factor = 8
        # Use the pretrained vae
        try:
            vae = AutoencoderKL.from_pretrained(vae_model_name, subfolder='vae', torch_dtype=precision)
        except:  # for handling SDXL vae fp16 fixed checkpoint
            vae = AutoencoderKL.from_pretrained(vae_model_name, torch_dtype=precision)
    else:
        # Use a custom autoencoder
        vae, latent_statistics = load_autoencoder(autoencoder_path, autoencoder_local_path, torch_dtype=precision)
        if latent_statistics is not None and latent_scale == 'latent_statistics':
            assert isinstance(latent_statistics['global_mean'], float)
            assert isinstance(latent_statistics['global_std'], float)
            second_moment = latent_statistics['global_mean']**2 + latent_statistics['global_std']**2
            latent_scale = 1 / math.sqrt(second_moment)
        else:
            raise ValueError(
                'Must specify latent scale when using a custom autoencoder without tracking latent statistics.')
        downsample_factor = 2**(len(vae.config['channel_multipliers']) - 1)

    # Make the unet
    if pretrained:
        unet = UNet2DConditionModel.from_pretrained(unet_model_name, subfolder='unet')
        if autoencoder_path is not None and vae.config['latent_channels'] != 4:
            raise ValueError(f'Pretrained unet has 4 latent channels but the vae has {vae.latent_channels}.')
    else:
        unet_config = PretrainedConfig.get_config_dict(unet_model_name, subfolder='unet')[0]
        if autoencoder_path is not None:
            # Adapt the unet config to account for differing number of latent channels if necessary
            unet_config['in_channels'] = vae.config['latent_channels']
            unet_config['out_channels'] = vae.config['latent_channels']
        # Init the unet from the config
        unet = UNet2DConditionModel(**unet_config)

        # Zero initialization trick
        for name, layer in unet.named_modules():
            # Final conv in ResNet blocks
            if name.endswith('conv2'):
                layer = zero_module(layer)
            # proj_out in attention blocks
            if name.endswith('to_out.0'):
                layer = zero_module(layer)
        # Last conv block out projection
        unet.conv_out = zero_module(unet.conv_out)

    if hasattr(unet.mid_block, 'attentions'):
        for attention in unet.mid_block.attentions:
            attention._fsdp_wrap = True
            attention._activation_checkpointing = True
    if hasattr(unet.mid_block, 'resnets'):
        for resnet in unet.mid_block.resnets:
            resnet._fsdp_wrap = True
    for block in unet.up_blocks:
        if hasattr(block, 'attentions'):
            for attention in block.attentions:
                attention._fsdp_wrap = True
                attention._activation_checkpointing = True
        if hasattr(block, 'resnets'):
            for resnet in block.resnets:
                resnet._fsdp_wrap = True
    for block in unet.down_blocks:
        if hasattr(block, 'attentions'):
            for attention in block.attentions:
                attention._fsdp_wrap = True
                attention._activation_checkpointing = True
        if hasattr(block, 'resnets'):
            for resnet in block.resnets:
                resnet._fsdp_wrap = True

    noise_scheduler = DDPMScheduler.from_pretrained(model_name, subfolder='scheduler')
    inference_noise_scheduler = EulerDiscreteScheduler(num_train_timesteps=1000,
                                                       beta_start=0.00085,
                                                       beta_end=0.012,
                                                       beta_schedule='scaled_linear',
                                                       trained_betas=None,
                                                       prediction_type=prediction_type,
                                                       interpolation_type='linear',
                                                       use_karras_sigmas=False,
                                                       timestep_spacing='leading',
                                                       steps_offset=1)

    # Make the composer model
    model = StableDiffusion(
        unet=unet,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        noise_scheduler=noise_scheduler,
        inference_noise_scheduler=inference_noise_scheduler,
        prediction_type=prediction_type,
        latent_scale=latent_scale,
        downsample_factor=downsample_factor,
        offset_noise=offset_noise,
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        val_guidance_scales=val_guidance_scales,
        val_seed=val_seed,
        loss_bins=loss_bins,
        precomputed_latents=precomputed_latents,
        encode_latents_in_fp16=encode_latents_in_fp16,
        mask_pad_tokens=mask_pad_tokens,
        fsdp=fsdp,
        sdxl=True,
    )
    if torch.cuda.is_available():
        model = DeviceGPU().module_to_device(model)
        if is_xformers_installed and use_xformers:
            model.unet.enable_xformers_memory_efficient_attention()
            if hasattr(model.vae, 'enable_xformers_memory_efficient_attention'):
                model.vae.enable_xformers_memory_efficient_attention()

    if clip_qkv is not None:
        if is_xformers_installed and use_xformers:
            attn_processor = ClippedXFormersAttnProcessor(clip_val=clip_qkv)
        else:
            attn_processor = ClippedAttnProcessor2_0(clip_val=clip_qkv)
        log.info('Using %s with clip_val %.1f' % (attn_processor.__class__, clip_qkv))
        model.unet.set_attn_processor(attn_processor)

    return model


def build_autoencoder(input_channels: int = 3,
                      output_channels: int = 3,
                      hidden_channels: int = 128,
                      latent_channels: int = 4,
                      double_latent_channels: bool = True,
                      channel_multipliers: Tuple[int, ...] = (1, 2, 4, 4),
                      num_residual_blocks: int = 2,
                      use_conv_shortcut: bool = False,
                      dropout_probability: float = 0.0,
                      resample_with_conv: bool = True,
                      zero_init_last: bool = False,
                      use_attention: bool = True,
                      input_key: str = 'image',
                      learn_log_var: bool = True,
                      log_var_init: float = 0.0,
                      kl_divergence_weight: float = 1.0,
                      lpips_weight: float = 0.25,
                      discriminator_weight: float = 0.5,
                      discriminator_num_filters: int = 64,
                      discriminator_num_layers: int = 3):
    """Autoencoder training setup. By default, this config matches the network architecure used in SD2 and SDXL.

    Args:
        input_channels (int): Number of input channels. Default: `3`.
        output_channels (int): Number of output channels. Default: `3`.
        hidden_channels (int): Number of hidden channels. Default: `128`.
        latent_channels (int): Number of latent channels. Default: `4`.
        double_latent_channels (bool): Whether to double the number of latent channels in the decoder. Default: `True`.
        channel_multipliers (tuple): Tuple of channel multipliers for each layer in the encoder and decoder. Default: `(1, 2, 4, 4)`.
        num_residual_blocks (int): Number of residual blocks in the encoder and decoder. Default: `2`.
        use_conv_shortcut (bool): Whether to use a convolutional shortcut in the residual blocks. Default: `False`.
        dropout_probability (float): Dropout probability. Default: `0.0`.
        resample_with_conv (bool): Whether to use a convolutional resampling layer. Default: `True`.
        zero_init_last (bool): Whether to zero initialize the last layer in resblocks+discriminator. Default: `False`.
        use_attention (bool): Whether to use attention in the encoder and decoder. Default: `True`.
        input_key (str): Key to use for the input. Default: `image`.
        learn_log_var (bool): Whether to learn the output log variance in the VAE. Default: `True`.
        log_var_init (float): Initial value for the output log variance. Default: `0.0`.
        kl_divergence_weight (float): Weight for the KL divergence loss. Default: `1.0`.
        lpips_weight (float): Weight for the LPIPS loss. Default: `0.25`.
        discriminator_weight (float): Weight for the discriminator loss. Default: `0.5`.
        discriminator_num_filters (int): Number of filters in the discriminator. Default: `64`.
        discriminator_num_layers (int): Number of layers in the discriminator. Default: `3`.
    """
    # Build the autoencoder
    autoencoder = AutoEncoder(
        input_channels=input_channels,
        output_channels=output_channels,
        hidden_channels=hidden_channels,
        latent_channels=latent_channels,
        double_latent_channels=double_latent_channels,
        channel_multipliers=channel_multipliers,
        num_residual_blocks=num_residual_blocks,
        use_conv_shortcut=use_conv_shortcut,
        dropout_probability=dropout_probability,
        resample_with_conv=resample_with_conv,
        zero_init_last=zero_init_last,
        use_attention=use_attention,
    )

    # Configure the loss function
    autoencoder_loss = AutoEncoderLoss(input_key=input_key,
                                       ae_output_channels=output_channels,
                                       learn_log_var=learn_log_var,
                                       log_var_init=log_var_init,
                                       kl_divergence_weight=kl_divergence_weight,
                                       lpips_weight=lpips_weight,
                                       discriminator_weight=discriminator_weight,
                                       discriminator_num_filters=discriminator_num_filters,
                                       discriminator_num_layers=discriminator_num_layers)

    composer_model = ComposerAutoEncoder(model=autoencoder, autoencoder_loss=autoencoder_loss, input_key=input_key)
    return composer_model


def build_diffusers_autoencoder(model_name: str = 'stabilityai/stable-diffusion-2-base',
                                pretrained: bool = True,
                                vae_subfolder: bool = True,
                                output_channels: int = 3,
                                input_key: str = 'image',
                                learn_log_var: bool = True,
                                log_var_init: float = 0.0,
                                kl_divergence_weight: float = 1.0,
                                lpips_weight: float = 0.25,
                                discriminator_weight: float = 0.5,
                                discriminator_num_filters: int = 64,
                                discriminator_num_layers: int = 3,
                                zero_init_last: bool = False):
    """Diffusers autoencoder training setup.

    Args:
        model_name (str): Name of the Huggingface model. Default: `stabilityai/stable-diffusion-2-base`.
        pretrained (bool): Whether to use a pretrained model. Default: `True`.
        vae_subfolder: (bool): Whether to find the model config in a vae subfolder. Default: `True`.
        output_channels (int): Number of output channels. Default: `3`.
        input_key (str): Key for the input to the model. Default: `image`.
        learn_log_var (bool): Whether to learn the output log variance. Default: `True`.
        log_var_init (float): Initial value for the output log variance. Default: `0.0`.
        kl_divergence_weight (float): Weight for the KL divergence loss. Default: `1.0`.
        lpips_weight (float): Weight for the LPIPs loss. Default: `0.25`.
        discriminator_weight (float): Weight for the discriminator loss. Default: `0.5`.
        discriminator_num_filters (int): Number of filters in the first layer of the discriminator. Default: `64`.
        discriminator_num_layers (int): Number of layers in the discriminator. Default: `3`.
        zero_init_last (bool): Whether to initialize the last conv layer to zero. Default: `False`.
    """
    # Get the model architecture and optionally the pretrained weights.
    if pretrained:
        if vae_subfolder:
            model = AutoencoderKL.from_pretrained(model_name, subfolder='vae')
        else:
            model = AutoencoderKL.from_pretrained(model_name)
    else:
        if vae_subfolder:
            config = PretrainedConfig.get_config_dict(model_name, subfolder='vae')
        else:
            config = PretrainedConfig.get_config_dict(model_name)
        model = AutoencoderKL(**config[0])

    # Configure the loss function
    autoencoder_loss = AutoEncoderLoss(input_key=input_key,
                                       ae_output_channels=output_channels,
                                       learn_log_var=learn_log_var,
                                       log_var_init=log_var_init,
                                       kl_divergence_weight=kl_divergence_weight,
                                       lpips_weight=lpips_weight,
                                       discriminator_weight=discriminator_weight,
                                       discriminator_num_filters=discriminator_num_filters,
                                       discriminator_num_layers=discriminator_num_layers)

    # Make the composer model
    composer_model = ComposerDiffusersAutoEncoder(model=model, autoencoder_loss=autoencoder_loss, input_key=input_key)
    return composer_model


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

        conditioning = torch.concat([conditioning, conditioning_2], dim=-1)
        return conditioning, pooled_conditioning


class SDXLTokenizer:
    """Wrapper around HuggingFace tokenizers for SDXL.

    Tokenizes prompt with two tokenizers and returns the joined output.

    Args:
        model_name (str): Name of the model's text encoders to load. Defaults to 'stabilityai/stable-diffusion-xl-base-1.0'.
    """

    def __init__(self, model_name='stabilityai/stable-diffusion-xl-base-1.0'):
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name, subfolder='tokenizer')
        self.tokenizer_2 = CLIPTokenizer.from_pretrained(model_name, subfolder='tokenizer_2')

    def __call__(self, prompt, padding, truncation, return_tensors, max_length=None):
        tokenized_output = self.tokenizer(
            prompt,
            padding=padding,
            max_length=self.tokenizer.model_max_length if max_length is None else max_length,
            truncation=truncation,
            return_tensors=return_tensors)
        tokenized_output_2 = self.tokenizer_2(
            prompt,
            padding=padding,
            max_length=self.tokenizer_2.model_max_length if max_length is None else max_length,
            truncation=truncation,
            return_tensors=return_tensors)

        # Add second tokenizer output to first tokenizer
        for key in tokenized_output.keys():
            tokenized_output[key] = [tokenized_output[key], tokenized_output_2[key]]
        return tokenized_output
