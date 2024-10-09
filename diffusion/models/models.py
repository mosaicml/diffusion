# Copyright 2022 MosaicML Diffusion authors
# SPDX-License-Identifier: Apache-2.0

"""Constructors for diffusion models."""

import logging
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from composer.devices import DeviceGPU
from diffusers import AutoencoderKL, DDIMScheduler, DDPMScheduler, EulerDiscreteScheduler, UNet2DConditionModel
from peft import LoraConfig
from torchmetrics import MeanSquaredError
from transformers import AutoModel, AutoTokenizer, CLIPTextModel, CLIPTokenizer, PretrainedConfig

from diffusion.models.autoencoder import (AutoEncoder, AutoEncoderLoss, ComposerAutoEncoder,
                                          ComposerDiffusersAutoEncoder, load_autoencoder)
from diffusion.models.layers import ClippedAttnProcessor2_0, ClippedXFormersAttnProcessor, zero_module
from diffusion.models.pixel_diffusion import PixelDiffusion
from diffusion.models.precomputed_text_latent_diffusion import PrecomputedTextLatentDiffusion
from diffusion.models.stable_diffusion import StableDiffusion
from diffusion.models.t2i_transformer import ComposerTextToImageMMDiT
from diffusion.models.text_encoder import MultiTextEncoder, MultiTokenizer
from diffusion.models.transformer import DiffusionTransformer
from diffusion.schedulers.schedulers import ContinuousTimeScheduler
from diffusion.schedulers.utils import shift_noise_schedule

try:
    import xformers  # type: ignore
    del xformers
    is_xformers_installed = True
except:
    is_xformers_installed = False

log = logging.getLogger(__name__)


def _parse_latent_statistics(latent_stat: Union[float, Tuple, str]) -> Union[float, Tuple, str]:
    if isinstance(latent_stat, str):
        latent_stat = latent_stat.lower()
        if latent_stat != 'latent_statistics':
            raise ValueError(f'Invalid latent statistic {latent_stat}. Must be a float, tuple or "latent_statistics".')
    elif type(latent_stat).__name__ == 'ListConfig' and not isinstance(latent_stat, float):
        latent_stat = tuple(latent_stat)
    return latent_stat


def stable_diffusion_2(
    model_name: str = 'stabilityai/stable-diffusion-2-base',
    pretrained: bool = True,
    autoencoder_path: Optional[str] = None,
    autoencoder_local_path: str = '/tmp/autoencoder_weights.pt',
    prediction_type: str = 'epsilon',
    latent_mean: Union[float, Tuple, str] = 0.0,
    latent_std: Union[float, Tuple, str] = 5.489980785067252,
    beta_schedule: str = 'scaled_linear',
    zero_terminal_snr: bool = False,
    offset_noise: Optional[float] = None,
    scheduler_shift_resolution: int = 256,
    train_metrics: Optional[List] = None,
    val_metrics: Optional[List] = None,
    quasirandomness: bool = False,
    train_seed: int = 42,
    val_seed: int = 1138,
    precomputed_latents: bool = False,
    encode_latents_in_fp16: bool = True,
    mask_pad_tokens: bool = False,
    fsdp: bool = True,
    clip_qkv: Optional[float] = None,
    use_xformers: bool = True,
    lora_rank: Optional[int] = None,
    lora_alpha: Optional[int] = None,
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
        latent_mean (float, list, str): The mean of the autoencoder latents. Either a float for a single value,
            a tuple of means, or or `'latent_statistics'` to try to use the value from the autoencoder
            checkpoint. Defaults to `0.0`.
        latent_std (float, list, str): The std. dev. of the autoencoder latents. Either a float for a single value,
            a tuple of std_devs, or or `'latent_statistics'` to try to use the value from the autoencoder
            checkpoint. Defaults to `1/0.18215`.
        beta_schedule (str): The beta schedule to use. Must be one of 'scaled_linear', 'linear', or 'squaredcos_cap_v2'.
            Default: `scaled_linear`.
        zero_terminal_snr (bool): Whether to enforce zero terminal SNR. Default: `False`.
        train_metrics (list, optional): List of metrics to compute during training. If None, defaults to
            [MeanSquaredError()].
        val_metrics (list, optional): List of metrics to compute during validation. If None, defaults to
            [MeanSquaredError()].
        quasirandomness (bool): Whether to use quasirandomness for generating diffusion process noise.
            Default: `False`.
        train_seed (int): Seed to use for generating diffusion process noise during training if using
            quasirandomness. Default: `42`.
        val_seed (int): Seed to use for generating evaluation images. Defaults to 1138.
        precomputed_latents (bool): Whether to use precomputed latents. Defaults to False.
        offset_noise (float, optional): The scale of the offset noise. If not specified, offset noise will not
            be used. Default `None`.
        scheduler_shift_resolution (int): The resolution to shift the noise scheduler to. Default: `256`.
        encode_latents_in_fp16 (bool): Whether to encode latents in fp16. Defaults to True.
        mask_pad_tokens (bool): Whether to mask pad tokens in cross attention. Defaults to False.
        fsdp (bool): Whether to use FSDP. Defaults to True.
        clip_qkv (float, optional): If not None, clip the qkv values to this value. Defaults to None.
        use_xformers (bool): Whether to use xformers for attention. Defaults to True.
        lora_rank (int, optional): If not None, the rank to use for LoRA finetuning. Defaults to None.
        lora_alpha (int, optional): If not None, the alpha to use for LoRA finetuning. Defaults to None.
    """
    latent_mean, latent_std = _parse_latent_statistics(latent_mean), _parse_latent_statistics(latent_std)

    if train_metrics is None:
        train_metrics = [MeanSquaredError()]
    if val_metrics is None:
        val_metrics = [MeanSquaredError()]

    precision = torch.float16 if encode_latents_in_fp16 else None
    # Make the text encoder
    text_encoder = CLIPTextModel.from_pretrained(model_name, subfolder='text_encoder', torch_dtype=precision)
    tokenizer = CLIPTokenizer.from_pretrained(model_name, subfolder='tokenizer', clean_up_tokenization_spaces=True)

    # Make the autoencoder
    if autoencoder_path is None:
        if latent_mean == 'latent_statistics' or latent_std == 'latent_statistics':
            raise ValueError('Cannot use tracked latent_statistics when using the pretrained vae.')
        # Use the pretrained vae
        downsample_factor = 8
        vae = AutoencoderKL.from_pretrained(model_name, subfolder='vae', torch_dtype=precision)
    else:
        # Use a custom autoencoder
        vae, latent_statistics = load_autoencoder(autoencoder_path, autoencoder_local_path, torch_dtype=precision)
        if latent_statistics is None and (latent_mean == 'latent_statistics' or latent_std == 'latent_statistics'):
            raise ValueError(
                'Must specify latent scale when using a custom autoencoder without tracking latent statistics.')
        if isinstance(latent_mean, str) and latent_mean == 'latent_statistics':
            assert isinstance(latent_statistics, dict)
            latent_mean = tuple(latent_statistics['latent_channel_means'])
        if isinstance(latent_std, str) and latent_std == 'latent_statistics':
            assert isinstance(latent_statistics, dict)
            latent_std = tuple(latent_statistics['latent_channel_stds'])
        downsample_factor = 2**(len(vae.config['channel_multipliers']) - 1)

    # Make the unet
    unet_config = PretrainedConfig.get_config_dict(model_name, subfolder='unet')[0]
    if pretrained:
        unet = UNet2DConditionModel.from_pretrained(model_name, subfolder='unet')
        if isinstance(vae, AutoEncoder) and vae.config['latent_channels'] != 4:
            raise ValueError(f'Pretrained unet has 4 latent channels but the vae has {vae.latent_channels}.')
    else:
        if isinstance(vae, AutoEncoder):
            # Adapt the unet config to account for differing number of latent channels if necessary
            unet_config['in_channels'] = vae.config['latent_channels']
            unet_config['out_channels'] = vae.config['latent_channels']
        # Init the unet from the config
        unet = UNet2DConditionModel(**unet_config)
    if isinstance(latent_mean, float):
        latent_mean = (latent_mean,) * unet_config['in_channels']
    if isinstance(latent_std, float):
        latent_std = (latent_std,) * unet_config['in_channels']
    assert isinstance(latent_mean, tuple) and isinstance(latent_std, tuple)

    # Make the noise schedulers
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000,
                                    beta_start=0.00085,
                                    beta_end=0.012,
                                    beta_schedule=beta_schedule,
                                    trained_betas=None,
                                    variance_type='fixed_small',
                                    clip_sample=False,
                                    prediction_type=prediction_type,
                                    sample_max_value=1.0,
                                    timestep_spacing='leading',
                                    steps_offset=1,
                                    rescale_betas_zero_snr=zero_terminal_snr)

    inference_noise_scheduler = DDIMScheduler(num_train_timesteps=1000,
                                              beta_start=0.00085,
                                              beta_end=0.012,
                                              beta_schedule=beta_schedule,
                                              trained_betas=None,
                                              clip_sample=False,
                                              set_alpha_to_one=False,
                                              prediction_type=prediction_type)

    # Shift noise scheduler to correct for resolution changes
    noise_scheduler = shift_noise_schedule(noise_scheduler,
                                           base_dim=32,
                                           shift_dim=scheduler_shift_resolution // downsample_factor)
    inference_noise_scheduler = shift_noise_schedule(inference_noise_scheduler,
                                                     base_dim=32,
                                                     shift_dim=scheduler_shift_resolution // downsample_factor)

    # Make the composer model
    model = StableDiffusion(
        unet=unet,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        noise_scheduler=noise_scheduler,
        inference_noise_scheduler=inference_noise_scheduler,
        prediction_type=prediction_type,
        latent_mean=latent_mean,
        latent_std=latent_std,
        downsample_factor=downsample_factor,
        offset_noise=offset_noise,
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        quasirandomness=quasirandomness,
        train_seed=train_seed,
        val_seed=val_seed,
        precomputed_latents=precomputed_latents,
        encode_latents_in_fp16=encode_latents_in_fp16,
        mask_pad_tokens=mask_pad_tokens,
        fsdp=fsdp,
    )
    if lora_rank is not None:
        assert lora_alpha is not None
        model.unet.requires_grad_(False)
        for param in model.unet.parameters():
            param.requires_grad_(False)

        unet_lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            init_lora_weights='gaussian',
            target_modules=['to_k', 'to_q', 'to_v', 'to_out.0'],
        )
        model.unet.add_adapter(unet_lora_config)
        model.unet._fsdp_wrap = True
        if hasattr(model.unet, 'mid_block') and model.unet.mid_block is not None:
            for attention in model.unet.mid_block.attentions:
                attention._fsdp_wrap = True
            for resnet in model.unet.mid_block.resnets:
                resnet._fsdp_wrap = True
        for block in model.unet.up_blocks:
            if hasattr(block, 'attentions'):
                for attention in block.attentions:
                    attention._fsdp_wrap = True
            if hasattr(block, 'resnets'):
                for resnet in block.resnets:
                    resnet._fsdp_wrap = True
        for block in model.unet.down_blocks:
            if hasattr(block, 'attentions'):
                for attention in block.attentions:
                    attention._fsdp_wrap = True
            if hasattr(block, 'resnets'):
                for resnet in block.resnets:
                    resnet._fsdp_wrap = True

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
    tokenizer_names: Union[str, Tuple[str, ...]] = ('stabilityai/stable-diffusion-xl-base-1.0/tokenizer',
                                                    'stabilityai/stable-diffusion-xl-base-1.0/tokenizer_2'),
    text_encoder_names: Union[str, Tuple[str, ...]] = ('stabilityai/stable-diffusion-xl-base-1.0/text_encoder',
                                                       'stabilityai/stable-diffusion-xl-base-1.0/text_encoder_2'),
    unet_model_name: str = 'stabilityai/stable-diffusion-xl-base-1.0',
    vae_model_name: str = 'madebyollin/sdxl-vae-fp16-fix',
    pretrained: bool = True,
    autoencoder_path: Optional[str] = None,
    autoencoder_local_path: str = '/tmp/autoencoder_weights.pt',
    prediction_type: str = 'epsilon',
    latent_mean: Union[float, Tuple, str] = 0.0,
    latent_std: Union[float, Tuple, str] = 7.67754318618,
    beta_schedule: str = 'scaled_linear',
    beta_start: float = 0.00085,
    beta_end: float = 0.012,
    zero_terminal_snr: bool = False,
    use_karras_sigmas: bool = False,
    offset_noise: Optional[float] = None,
    scheduler_shift_resolution: int = 256,
    train_metrics: Optional[List] = None,
    val_metrics: Optional[List] = None,
    quasirandomness: bool = False,
    train_seed: int = 42,
    val_seed: int = 1138,
    precomputed_latents: bool = False,
    encode_latents_in_fp16: bool = True,
    mask_pad_tokens: bool = False,
    fsdp: bool = True,
    clip_qkv: Optional[float] = None,
    use_xformers: bool = True,
    lora_rank: Optional[int] = None,
    lora_alpha: Optional[int] = None,
):
    """Stable diffusion 2 training setup + SDXL UNet and VAE.

    Requires batches of matched images and text prompts to train. Generates images from text
    prompts. Currently uses UNet and VAE config from SDXL, but text encoder/tokenizer from SD2.

    Args:
        tokenizer_names (str, Tuple[str, ...]): HuggingFace name(s) of the tokenizer(s) to load.
            Default: ``('stabilityai/stable-diffusion-xl-base-1.0/tokenizer',
            'stabilityai/stable-diffusion-xl-base-1.0/tokenizer_2')``.
        text_encoder_names (str, Tuple[str, ...]): HuggingFace name(s) of the text encoder(s) to load.
            Default: ``('stabilityai/stable-diffusion-xl-base-1.0/text_encoder',
            'stabilityai/stable-diffusion-xl-base-1.0/text_encoder_2')``.
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
        latent_mean (float, Tuple, str): The mean of the autoencoder latents. Either a float for a single value,
            a tuple of means, or or `'latent_statistics'` to try to use the value from the autoencoder
            checkpoint. Defaults to `0.0`.
        latent_std (float, Tuple, str): The std. dev. of the autoencoder latents. Either a float for a single value,
            a tuple of std_devs, or or `'latent_statistics'` to try to use the value from the autoencoder
            checkpoint. Defaults to `1/0.13025`.
        beta_schedule (str): The beta schedule to use. Must be one of 'scaled_linear', 'linear', or 'squaredcos_cap_v2'.
            Default: `scaled_linear`.
        beta_start (float): The starting beta value. Default: `0.00085`.
        beta_end (float): The ending beta value. Default: `0.012`.
        zero_terminal_snr (bool): Whether to enforce zero terminal SNR. Default: `False`.
        use_karras_sigmas (bool): Whether to use the Karras sigmas for the diffusion process noise. Default: `False`.
        offset_noise (float, optional): The scale of the offset noise. If not specified, offset noise will not
            be used. Default `None`.
        scheduler_shift_resolution (int): The resolution to shift the noise scheduler to. Default: `256`.
        train_metrics (list, optional): List of metrics to compute during training. If None, defaults to
            [MeanSquaredError()].
        val_metrics (list, optional): List of metrics to compute during validation. If None, defaults to
            [MeanSquaredError()].
        quasirandomness (bool): Whether to use quasirandomness for generating diffusion process noise.
            Default: `False`.
        train_seed (int): Seed to use for generating diffusion process noise during training if using
            quasirandomness. Default: `42`.
        val_seed (int): Seed to use for generating evaluation images. Defaults to 1138.
        precomputed_latents (bool): Whether to use precomputed latents. Defaults to False.
        encode_latents_in_fp16 (bool): Whether to encode latents in fp16. Defaults to True.
        mask_pad_tokens (bool): Whether to mask pad tokens in cross attention. Defaults to False.
        fsdp (bool): Whether to use FSDP. Defaults to True.
        clip_qkv (float, optional): If not None, clip the qkv values to this value. Improves stability of training.
            Default: ``None``.
        use_xformers (bool): Whether to use xformers for attention. Defaults to True.
        lora_rank (int, optional): If not None, the rank to use for LoRA finetuning. Defaults to None.
        lora_alpha (int, optional): If not None, the alpha to use for LoRA finetuning. Defaults to None.
    """
    latent_mean, latent_std = _parse_latent_statistics(latent_mean), _parse_latent_statistics(latent_std)

    if (isinstance(tokenizer_names, tuple) or
            isinstance(text_encoder_names, tuple)) and len(tokenizer_names) != len(text_encoder_names):
        raise ValueError('Number of tokenizer_names and text_encoder_names must be equal')

    if train_metrics is None:
        train_metrics = [MeanSquaredError()]
    if val_metrics is None:
        val_metrics = [MeanSquaredError()]

    # Make the tokenizer and text encoder
    tokenizer = MultiTokenizer(tokenizer_names_or_paths=tokenizer_names)
    text_encoder = MultiTextEncoder(model_names=text_encoder_names,
                                    encode_latents_in_fp16=encode_latents_in_fp16,
                                    pretrained_sdxl=pretrained)

    precision = torch.float16 if encode_latents_in_fp16 else None
    # Make the autoencoder
    if autoencoder_path is None:
        if latent_mean == 'latent_statistics' or latent_std == 'latent_statistics':
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
        if latent_statistics is None and (latent_mean == 'latent_statistics' or latent_std == 'latent_statistics'):
            raise ValueError(
                'Must specify latent scale when using a custom autoencoder without tracking latent statistics.')
        if isinstance(latent_mean, str) and latent_mean == 'latent_statistics':
            assert isinstance(latent_statistics, dict)
            latent_mean = tuple(latent_statistics['latent_channel_means'])
        if isinstance(latent_std, str) and latent_std == 'latent_statistics':
            assert isinstance(latent_statistics, dict)
            latent_std = tuple(latent_statistics['latent_channel_stds'])
        downsample_factor = 2**(len(vae.config['channel_multipliers']) - 1)

    # Make the unet
    unet_config = PretrainedConfig.get_config_dict(unet_model_name, subfolder='unet')[0]
    if pretrained:
        unet = UNet2DConditionModel.from_pretrained(unet_model_name, subfolder='unet')
        if isinstance(vae, AutoEncoder) and vae.config['latent_channels'] != 4:
            raise ValueError(f'Pretrained unet has 4 latent channels but the vae has {vae.latent_channels}.')
    else:
        if isinstance(vae, AutoEncoder):
            # Adapt the unet config to account for differing number of latent channels if necessary
            unet_config['in_channels'] = vae.config['latent_channels']
            unet_config['out_channels'] = vae.config['latent_channels']
        unet_config['cross_attention_dim'] = text_encoder.text_encoder_dim
        # This config variable is the sum of the text encoder projection dimension and
        # the number of additional time embeddings (6) * addition_time_embed_dim (256)
        unet_config['projection_class_embeddings_input_dim'] = text_encoder.text_encoder_proj_dim + 1536
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
    if isinstance(latent_mean, float):
        latent_mean = (latent_mean,) * unet_config['in_channels']
    if isinstance(latent_std, float):
        latent_std = (latent_std,) * unet_config['in_channels']
    assert isinstance(latent_mean, tuple) and isinstance(latent_std, tuple)

    assert isinstance(unet, UNet2DConditionModel)
    if hasattr(unet, 'mid_block') and unet.mid_block is not None:
        for attention in unet.mid_block.attentions:
            attention._fsdp_wrap = True
        for resnet in unet.mid_block.resnets:
            resnet._fsdp_wrap = True
    for block in unet.up_blocks:
        if hasattr(block, 'attentions'):
            for attention in block.attentions:
                attention._fsdp_wrap = True
        if hasattr(block, 'resnets'):
            for resnet in block.resnets:
                resnet._fsdp_wrap = True
    for block in unet.down_blocks:
        if hasattr(block, 'attentions'):
            for attention in block.attentions:
                attention._fsdp_wrap = True
        if hasattr(block, 'resnets'):
            for resnet in block.resnets:
                resnet._fsdp_wrap = True

    # Make the noise schedulers
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000,
                                    beta_start=beta_start,
                                    beta_end=beta_end,
                                    beta_schedule=beta_schedule,
                                    trained_betas=None,
                                    variance_type='fixed_small',
                                    clip_sample=False,
                                    prediction_type=prediction_type,
                                    sample_max_value=1.0,
                                    timestep_spacing='leading',
                                    steps_offset=1,
                                    rescale_betas_zero_snr=zero_terminal_snr)
    if beta_schedule == 'squaredcos_cap_v2':
        inference_noise_scheduler = DDIMScheduler(num_train_timesteps=1000,
                                                  beta_start=beta_start,
                                                  beta_end=beta_end,
                                                  beta_schedule=beta_schedule,
                                                  trained_betas=None,
                                                  clip_sample=False,
                                                  set_alpha_to_one=False,
                                                  prediction_type=prediction_type,
                                                  rescale_betas_zero_snr=zero_terminal_snr)
    else:
        inference_noise_scheduler = EulerDiscreteScheduler(num_train_timesteps=1000,
                                                           beta_start=beta_start,
                                                           beta_end=beta_end,
                                                           beta_schedule=beta_schedule,
                                                           trained_betas=None,
                                                           prediction_type=prediction_type,
                                                           interpolation_type='linear',
                                                           use_karras_sigmas=use_karras_sigmas,
                                                           timestep_spacing='leading',
                                                           steps_offset=1,
                                                           rescale_betas_zero_snr=zero_terminal_snr)

    # Shift noise scheduler to correct for resolution changes
    noise_scheduler = shift_noise_schedule(noise_scheduler,
                                           base_dim=32,
                                           shift_dim=scheduler_shift_resolution // downsample_factor)
    inference_noise_scheduler = shift_noise_schedule(inference_noise_scheduler,
                                                     base_dim=32,
                                                     shift_dim=scheduler_shift_resolution // downsample_factor)

    # Make the composer model
    model = StableDiffusion(
        unet=unet,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        noise_scheduler=noise_scheduler,
        inference_noise_scheduler=inference_noise_scheduler,
        prediction_type=prediction_type,
        latent_mean=latent_mean,
        latent_std=latent_std,
        downsample_factor=downsample_factor,
        offset_noise=offset_noise,
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        quasirandomness=quasirandomness,
        train_seed=train_seed,
        val_seed=val_seed,
        precomputed_latents=precomputed_latents,
        encode_latents_in_fp16=encode_latents_in_fp16,
        mask_pad_tokens=mask_pad_tokens,
        fsdp=fsdp,
        sdxl=True,
    )

    if lora_rank is not None:
        assert lora_alpha is not None
        model.unet.requires_grad_(False)
        for param in model.unet.parameters():
            param.requires_grad_(False)

        unet_lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            init_lora_weights='gaussian',
            target_modules=['to_k', 'to_q', 'to_v', 'to_out.0'],
        )
        model.unet.add_adapter(unet_lora_config)
        model.unet._fsdp_wrap = True
        if hasattr(model.unet, 'mid_block') and model.unet.mid_block is not None:
            for attention in model.unet.mid_block.attentions:
                attention._fsdp_wrap = True
            for resnet in model.unet.mid_block.resnets:
                resnet._fsdp_wrap = True
        for block in model.unet.up_blocks:
            if hasattr(block, 'attentions'):
                for attention in block.attentions:
                    attention._fsdp_wrap = True
            if hasattr(block, 'resnets'):
                for resnet in block.resnets:
                    resnet._fsdp_wrap = True
        for block in model.unet.down_blocks:
            if hasattr(block, 'attentions'):
                for attention in block.attentions:
                    attention._fsdp_wrap = True
            if hasattr(block, 'resnets'):
                for resnet in block.resnets:
                    resnet._fsdp_wrap = True
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


def precomputed_text_latent_diffusion(
    unet_model_name: str = 'stabilityai/stable-diffusion-xl-base-1.0',
    vae_model_name: str = 'madebyollin/sdxl-vae-fp16-fix',
    autoencoder_path: Optional[str] = None,
    autoencoder_local_path: str = '/tmp/autoencoder_weights.pt',
    include_text_encoders: bool = False,
    text_encoder_dtype: str = 'bfloat16',
    cache_dir: str = '/tmp/hf_files',
    prediction_type: str = 'epsilon',
    image_key: str = 'image',
    t5_latent_key: str = 'T5_LATENTS',
    t5_mask_key: str = 'T5_ATTENTION_MASK',
    clip_latent_key: str = 'CLIP_LATENTS',
    clip_mask_key: str = 'CLIP_ATTENTION_MASK',
    clip_pooled_key: str = 'CLIP_POOLED',
    latent_mean: Union[float, Tuple, str] = 0.0,
    latent_std: Union[float, Tuple, str] = 7.67754318618,
    text_embed_dim: int = 4096,
    train_noise_scheduler_params: Optional[Dict[str, Any]] = None,
    inference_noise_scheduler_params: Optional[Dict[str, Any]] = None,
    scheduler_shift_resolution: int = 256,
    train_metrics: Optional[List] = None,
    val_metrics: Optional[List] = None,
    quasirandomness: bool = False,
    train_seed: int = 42,
    val_seed: int = 1138,
    fsdp: bool = True,
    use_xformers: bool = True,
    lora_rank: Optional[int] = None,
    lora_alpha: Optional[int] = None,
):
    """Latent diffusion model training using precomputed text latents from T5-XXL and CLIP.

    Args:
        unet_model_name (str): Name of the UNet model to load. Defaults to
            'stabilityai/stable-diffusion-xl-base-1.0'.
        vae_model_name (str): Name of the VAE model to load. Defaults to
            'madebyollin/sdxl-vae-fp16-fix' as the official VAE checkpoint (from
            'stabilityai/stable-diffusion-xl-base-1.0') is not compatible with fp16.
        autoencoder_path (optional, str): Path to autoencoder weights if using custom autoencoder. If not specified,
            will use the vae from `model_name`. Default `None`.
        autoencoder_local_path (optional, str): Path to autoencoder weights. Default: `/tmp/autoencoder_weights.pt`.
        include_text_encoders (bool): Whether to include text encoders in the model. Should only do this for running
            inference. Default: `False`.
        text_encoder_dtype (str): The dtype to use for the text encoder. One of [`float32`, `float16`, `bfloat16`].
            Default: `bfloat16`.
        cache_dir (str): Directory to cache the model in if using `include_text_encoders`. Default: `'/tmp/hf_files'`.
        prediction_type (str): The type of prediction to use. Must be one of 'sample',
            'epsilon', or 'v_prediction'. Default: `epsilon`.
        image_key (str): The key to use for the image in the precomputed latents. Default: `'image'`.
        t5_latent_key (str): The key to use for the T5 latents in the precomputed latents. Default: `'T5_LATENTS'`.
        t5_mask_key (str): The key to use for the T5 attention mask in the precomputed latents. Default: `'T5_ATTENTION_MASK'`.
        clip_latent_key (str): The key to use for the CLIP latents in the precomputed latents. Default: `'CLIP_LATENTS'`.
        clip_mask_key (str): The key to use for the CLIP attention mask in the precomputed latents. Default: `'CLIP_ATTENTION_MASK'`.
        clip_pooled_key (str): The key to use for the CLIP pooled in the precomputed latents. Default: `'CLIP_POOLED'`.
        latent_mean (float, Tuple, str): The mean of the autoencoder latents. Either a float for a single value,
            a tuple of means, or or `'latent_statistics'` to try to use the value from the autoencoder
            checkpoint. Defaults to `0.0`.
        latent_std (float, Tuple, str): The std. dev. of the autoencoder latents. Either a float for a single value,
            a tuple of std_devs, or or `'latent_statistics'` to try to use the value from the autoencoder
            checkpoint. Defaults to `1/0.13025`.
        text_embed_dim (int): The dimension to project the text embeddings to. Default: `4096`.
        train_noise_scheduler_params (Dict): Parameters to overried in the training noise scheduler. Anything not
            specified will default to SDXL values. Default: `None`.
        inference_noise_scheduler_params (Dict): Parameters to overried in the inference noise scheduler. Anything
            not specified will default to SDXL values. Default: `None`.
        scheduler_shift_resolution (int): The resolution to shift the noise scheduler to. Default: `256`.
        train_metrics (list, optional): List of metrics to compute during training. If None, defaults to
            [MeanSquaredError()].
        val_metrics (list, optional): List of metrics to compute during validation. If None, defaults to
            [MeanSquaredError()].
        quasirandomness (bool): Whether to use quasirandomness for generating diffusion process noise.
            Default: `False`.
        train_seed (int): Seed to use for generating diffusion process noise during training if using
            quasirandomness. Default: `42`.
        val_seed (int): Seed to use for generating evaluation images. Defaults to 1138.
        fsdp (bool): Whether to use FSDP. Defaults to True.
        use_xformers (bool): Whether to use xformers for attention. Defaults to True.
        lora_rank (int, optional): If not None, the rank to use for LoRA finetuning. Defaults to None.
        lora_alpha (int, optional): If not None, the alpha to use for LoRA finetuning. Defaults to None.
    """
    latent_mean, latent_std = _parse_latent_statistics(latent_mean), _parse_latent_statistics(latent_std)

    if train_metrics is None:
        train_metrics = [MeanSquaredError()]
    if val_metrics is None:
        val_metrics = [MeanSquaredError()]

    # Make the autoencoder
    if autoencoder_path is None:
        if latent_mean == 'latent_statistics' or latent_std == 'latent_statistics':
            raise ValueError('Cannot use tracked latent_statistics when using the pretrained vae.')
        downsample_factor = 8
        # Use the pretrained vae
        try:
            vae = AutoencoderKL.from_pretrained(vae_model_name, subfolder='vae', torch_dtype=torch.float16)
        except:  # for handling SDXL vae fp16 fixed checkpoint
            vae = AutoencoderKL.from_pretrained(vae_model_name, torch_dtype=torch.float16)
    else:
        # Use a custom autoencoder
        vae, latent_statistics = load_autoencoder(autoencoder_path, autoencoder_local_path, torch_dtype=torch.float16)
        if latent_statistics is None and (latent_mean == 'latent_statistics' or latent_std == 'latent_statistics'):
            raise ValueError(
                'Must specify latent scale when using a custom autoencoder without tracking latent statistics.')
        if isinstance(latent_mean, str) and latent_mean == 'latent_statistics':
            assert isinstance(latent_statistics, dict)
            latent_mean = tuple(latent_statistics['latent_channel_means'])
        if isinstance(latent_std, str) and latent_std == 'latent_statistics':
            assert isinstance(latent_statistics, dict)
            latent_std = tuple(latent_statistics['latent_channel_stds'])
        downsample_factor = 2**(len(vae.config['channel_multipliers']) - 1)

    # Make the unet
    unet_config = PretrainedConfig.get_config_dict(unet_model_name, subfolder='unet')[0]

    if isinstance(vae, AutoEncoder):
        # Adapt the unet config to account for differing number of latent channels if necessary
        unet_config['in_channels'] = vae.config['latent_channels']
        unet_config['out_channels'] = vae.config['latent_channels']
    unet_config['cross_attention_dim'] = text_embed_dim
    # This config variable is the sum of the text encoder projection dimension and
    # the number of additional time embeddings (6) * addition_time_embed_dim (256)
    unet_config['projection_class_embeddings_input_dim'] = 2304
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

    if isinstance(latent_mean, float):
        latent_mean = (latent_mean,) * unet_config['in_channels']
    if isinstance(latent_std, float):
        latent_std = (latent_std,) * unet_config['in_channels']
    assert isinstance(latent_mean, tuple) and isinstance(latent_std, tuple)

    # FSDP Wrapping Scheme
    if hasattr(unet, 'mid_block') and unet.mid_block is not None:
        for attention in unet.mid_block.attentions:
            attention._fsdp_wrap = True
        for resnet in unet.mid_block.resnets:
            resnet._fsdp_wrap = True
    for block in unet.up_blocks:
        if hasattr(block, 'attentions'):
            for attention in block.attentions:
                attention._fsdp_wrap = True
        if hasattr(block, 'resnets'):
            for resnet in block.resnets:
                resnet._fsdp_wrap = True
    for block in unet.down_blocks:
        if hasattr(block, 'attentions'):
            for attention in block.attentions:
                attention._fsdp_wrap = True
        if hasattr(block, 'resnets'):
            for resnet in block.resnets:
                resnet._fsdp_wrap = True

    # Make the noise schedulers
    train_scheduler_params: Dict[str, Any] = {
        'num_train_timesteps': 1000,
        'beta_start': 0.00085,
        'beta_end': 0.012,
        'beta_schedule': 'scaled_linear',
        'variance_type': 'fixed_small',
        'clip_sample': False,
        'prediction_type': prediction_type,
        'sample_max_value': 1.0,
        'timestep_spacing': 'leading',
        'steps_offset': 1,
        'rescale_betas_zero_snr': False,
    }
    if train_noise_scheduler_params is not None:
        train_scheduler_params.update(train_noise_scheduler_params)
    noise_scheduler = DDPMScheduler(**train_scheduler_params)

    inference_scheduler_params: Dict[str, Any] = {
        'num_train_timesteps': 1000,
        'beta_start': 0.00085,
        'beta_end': 0.012,
        'beta_schedule': 'scaled_linear',
        'trained_betas': None,
        'prediction_type': prediction_type,
        'interpolation_type': 'linear',
        'use_karras_sigmas': False,
        'timestep_spacing': 'leading',
        'steps_offset': 1,
        'rescale_betas_zero_snr': False,
    }

    if inference_noise_scheduler_params is not None:
        inference_scheduler_params.update(inference_noise_scheduler_params)
    inference_noise_scheduler = EulerDiscreteScheduler(**inference_scheduler_params)

    # Shift noise scheduler to correct for resolution changes
    noise_scheduler = shift_noise_schedule(noise_scheduler,
                                           base_dim=32,
                                           shift_dim=scheduler_shift_resolution // downsample_factor)
    inference_noise_scheduler = shift_noise_schedule(inference_noise_scheduler,
                                                     base_dim=32,
                                                     shift_dim=scheduler_shift_resolution // downsample_factor)

    # Optionally load the tokenizers and text encoders
    t5_tokenizer, t5_encoder, clip_tokenizer, clip_encoder = None, None, None, None
    if include_text_encoders:
        dtype_map = {'float32': torch.float32, 'float16': torch.float16, 'bfloat16': torch.bfloat16}
        dtype = dtype_map[text_encoder_dtype]
        t5_tokenizer = AutoTokenizer.from_pretrained('google/t5-v1_1-xxl', cache_dir=cache_dir, local_files_only=True)
        clip_tokenizer = AutoTokenizer.from_pretrained('stabilityai/stable-diffusion-xl-base-1.0',
                                                       subfolder='tokenizer',
                                                       cache_dir=cache_dir,
                                                       local_files_only=False)
        t5_encoder = AutoModel.from_pretrained('google/t5-v1_1-xxl',
                                               torch_dtype=dtype,
                                               cache_dir=cache_dir,
                                               local_files_only=False).encoder.eval()
        clip_encoder = CLIPTextModel.from_pretrained('stabilityai/stable-diffusion-xl-base-1.0',
                                                     subfolder='text_encoder',
                                                     torch_dtype=dtype,
                                                     cache_dir=cache_dir,
                                                     local_files_only=False).cuda().eval()
    # Make the composer model
    model = PrecomputedTextLatentDiffusion(
        unet=unet,
        vae=vae,
        t5_tokenizer=t5_tokenizer,
        t5_encoder=t5_encoder,
        clip_tokenizer=clip_tokenizer,
        clip_encoder=clip_encoder,
        noise_scheduler=noise_scheduler,
        inference_noise_scheduler=inference_noise_scheduler,
        prediction_type=prediction_type,
        image_key=image_key,
        t5_latent_key=t5_latent_key,
        t5_mask_key=t5_mask_key,
        clip_latent_key=clip_latent_key,
        clip_mask_key=clip_mask_key,
        clip_pooled_key=clip_pooled_key,
        latent_mean=latent_mean,
        latent_std=latent_std,
        downsample_factor=downsample_factor,
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        quasirandomness=quasirandomness,
        train_seed=train_seed,
        val_seed=val_seed,
        text_embed_dim=text_embed_dim,
        fsdp=fsdp,
    )

    if lora_rank is not None:
        assert lora_alpha is not None
        model.unet.requires_grad_(False)
        for param in model.unet.parameters():
            param.requires_grad_(False)

        unet_lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            init_lora_weights='gaussian',
            target_modules=['to_k', 'to_q', 'to_v', 'to_out.0'],
        )
        model.unet.add_adapter(unet_lora_config)
        model.unet._fsdp_wrap = True
        if hasattr(model.unet, 'mid_block') and model.unet.mid_block is not None:
            for attention in model.unet.mid_block.attentions:
                attention._fsdp_wrap = True
            for resnet in model.unet.mid_block.resnets:
                resnet._fsdp_wrap = True
        for block in model.unet.up_blocks:
            if hasattr(block, 'attentions'):
                for attention in block.attentions:
                    attention._fsdp_wrap = True
            if hasattr(block, 'resnets'):
                for resnet in block.resnets:
                    resnet._fsdp_wrap = True
        for block in model.unet.down_blocks:
            if hasattr(block, 'attentions'):
                for attention in block.attentions:
                    attention._fsdp_wrap = True
            if hasattr(block, 'resnets'):
                for resnet in block.resnets:
                    resnet._fsdp_wrap = True

    if torch.cuda.is_available():
        model = DeviceGPU().module_to_device(model)
        if is_xformers_installed and use_xformers:
            model.unet.enable_xformers_memory_efficient_attention()
            if hasattr(model.vae, 'enable_xformers_memory_efficient_attention'):
                model.vae.enable_xformers_memory_efficient_attention()

    return model


def text_to_image_transformer(
    tokenizer_names: Union[str, Tuple[str, ...]] = ('stabilityai/stable-diffusion-xl-base-1.0/tokenizer'),
    text_encoder_names: Union[str, Tuple[str, ...]] = ('stabilityai/stable-diffusion-xl-base-1.0/text_encoder'),
    vae_model_name: str = 'madebyollin/sdxl-vae-fp16-fix',
    autoencoder_path: Optional[str] = None,
    autoencoder_local_path: str = '/tmp/autoencoder_weights.pt',
    num_layers: int = 28,
    max_image_side: int = 1280,
    conditioning_features: int = 768,
    conditioning_max_sequence_length: int = 77,
    patch_size: int = 2,
    latent_mean: Union[float, Tuple, str] = 0.0,
    latent_std: Union[float, Tuple, str] = 7.67754318618,
    timestep_mean: float = 0.0,
    timestep_std: float = 1.0,
    timestep_shift: float = 1.0,
    image_key: str = 'image',
    caption_key: str = 'captions',
    caption_mask_key: str = 'attention_mask',
    pretrained: bool = False,
):
    """Text to image transformer training setup.

    Args:
        tokenizer_names (str, Tuple[str, ...]): HuggingFace name(s) of the tokenizer(s) to load.
            Default: ``('stabilityai/stable-diffusion-xl-base-1.0/tokenizer')``.
        text_encoder_names (str, Tuple[str, ...]): HuggingFace name(s) of the text encoder(s) to load.
            Default: ``('stabilityai/stable-diffusion-xl-base-1.0/text_encoder')``.
        vae_model_name (str): Name of the VAE model to load. Defaults to 'madebyollin/sdxl-vae-fp16-fix'.
        autoencoder_path (optional, str): Path to autoencoder weights if using custom autoencoder. If not specified,
            will use the vae from `model_name`. Default `None`.
        autoencoder_local_path (optional, str): Path to autoencoder weights. Default: `/tmp/autoencoder_weights.pt`.
        num_layers (int): Number of layers in the transformer. Number of heads and layer width are determined by
            this according to `num_features = 64 * num_layers`, and `num_heads = num_layers`. Default: `28`.
        max_image_side (int): Maximum side length of the image. Default: `1280`.
        conditioning_features (int): Number of features in the conditioning transformer. Default: `768`.
        conditioning_max_sequence_length (int): Maximum sequence length for the conditioning transformer. Default: `77`.
        patch_size (int): Patch size for the transformer. Default: `2`.
        latent_mean (float, Tuple, str): The mean of the autoencoder latents. Either a float for a single value,
            a tuple of means, or or `'latent_statistics'` to try to use the value from the autoencoder
            checkpoint. Defaults to `0.0`.
        latent_std (float, Tuple, str): The std. dev. of the autoencoder latents. Either a float for a single value,
            a tuple of std_devs, or or `'latent_statistics'` to try to use the value from the autoencoder
            checkpoint. Defaults to `1/0.13025`.
        timestep_mean (float): The mean of the timesteps. Default: `0.0`.
        timestep_std (float): The std. dev. of the timesteps. Default: `1.0`.
        timestep_shift (float): The shift of the timesteps. Default: `1.0`.
        image_key (str): The key for the image in the batch. Default: `image`.
        caption_key (str): The key for the captions in the batch. Default: `captions`.
        caption_mask_key (str): The key for the caption mask in the batch. Default: `attention_mask`.
        pretrained (bool): Whether to load pretrained weights. Not used. Defaults to False.
    """
    latent_mean, latent_std = _parse_latent_statistics(latent_mean), _parse_latent_statistics(latent_std)

    if (isinstance(tokenizer_names, tuple) or
            isinstance(text_encoder_names, tuple)) and len(tokenizer_names) != len(text_encoder_names):
        raise ValueError('Number of tokenizer_names and text_encoder_names must be equal')

    # Make the tokenizer and text encoder
    tokenizer = MultiTokenizer(tokenizer_names_or_paths=tokenizer_names)
    text_encoder = MultiTextEncoder(model_names=text_encoder_names, encode_latents_in_fp16=True, pretrained_sdxl=False)

    precision = torch.float16
    # Make the autoencoder
    if autoencoder_path is None:
        if latent_mean == 'latent_statistics' or latent_std == 'latent_statistics':
            raise ValueError('Cannot use tracked latent_statistics when using the pretrained vae.')
        downsample_factor = 8
        autoencoder_channels = 4
        # Use the pretrained vae
        try:
            vae = AutoencoderKL.from_pretrained(vae_model_name, subfolder='vae', torch_dtype=precision)
        except:  # for handling SDXL vae fp16 fixed checkpoint
            vae = AutoencoderKL.from_pretrained(vae_model_name, torch_dtype=precision)
    else:
        # Use a custom autoencoder
        vae, latent_statistics = load_autoencoder(autoencoder_path, autoencoder_local_path, torch_dtype=precision)
        if latent_statistics is None and (latent_mean == 'latent_statistics' or latent_std == 'latent_statistics'):
            raise ValueError(
                'Must specify latent scale when using a custom autoencoder without tracking latent statistics.')
        if isinstance(latent_mean, str) and latent_mean == 'latent_statistics':
            assert isinstance(latent_statistics, dict)
            latent_mean = tuple(latent_statistics['latent_channel_means'])
        if isinstance(latent_std, str) and latent_std == 'latent_statistics':
            assert isinstance(latent_statistics, dict)
            latent_std = tuple(latent_statistics['latent_channel_stds'])
        downsample_factor = 2**(len(vae.config['channel_multipliers']) - 1)
        autoencoder_channels = vae.config['latent_channels']
    assert isinstance(vae, torch.nn.Module)
    if isinstance(latent_mean, float):
        latent_mean = (latent_mean,) * autoencoder_channels
    if isinstance(latent_std, float):
        latent_std = (latent_std,) * autoencoder_channels
    assert isinstance(latent_mean, tuple) and isinstance(latent_std, tuple)
    # Figure out the maximum input sequence length
    input_max_sequence_length = math.ceil(max_image_side / (downsample_factor * patch_size))
    # Make the transformer model
    transformer = DiffusionTransformer(num_features=64 * num_layers,
                                       num_heads=num_layers,
                                       num_layers=num_layers,
                                       input_features=autoencoder_channels * (patch_size**2),
                                       input_max_sequence_length=input_max_sequence_length,
                                       input_dimension=2,
                                       conditioning_features=conditioning_features,
                                       conditioning_max_sequence_length=conditioning_max_sequence_length,
                                       conditioning_dimension=1,
                                       expansion_factor=4)
    # Make the composer model
    model = ComposerTextToImageMMDiT(model=transformer,
                                     autoencoder=vae,
                                     text_encoder=text_encoder,
                                     tokenizer=tokenizer,
                                     latent_mean=latent_mean,
                                     latent_std=latent_std,
                                     patch_size=patch_size,
                                     downsample_factor=downsample_factor,
                                     latent_channels=autoencoder_channels,
                                     timestep_mean=timestep_mean,
                                     timestep_std=timestep_std,
                                     timestep_shift=timestep_shift,
                                     image_key=image_key,
                                     caption_key=caption_key,
                                     caption_mask_key=caption_mask_key)

    if torch.cuda.is_available():
        model = DeviceGPU().module_to_device(model)
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
    assert isinstance(model, AutoencoderKL)

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
    unet = UNet2DConditionModel(
        in_channels=3,
        out_channels=3,
        attention_head_dim=[5, 10, 20, 20],  # type: ignore
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
    unet = UNet2DConditionModel(
        in_channels=3,
        out_channels=3,
        attention_head_dim=[5, 10, 20, 20],  # type: ignore
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
