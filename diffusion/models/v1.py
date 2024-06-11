# Copyright 2022 MosaicML Diffusion authors
# SPDX-License-Identifier: Apache-2.0

"""Diffusion models."""

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from composer.devices import DeviceGPU
from composer.models import ComposerModel
from composer.utils import dist
from diffusers import AutoencoderKL, DDIMScheduler, DDPMScheduler, EulerDiscreteScheduler, UNet2DConditionModel
from scipy.stats import qmc
from torchmetrics import MeanSquaredError
from transformers import PretrainedConfig
from tqdm.auto import tqdm

from diffusion.models.autoencoder import AutoEncoder, load_autoencoder
from diffusion.models.layers import zero_module
from diffusion.models.models import _parse_latent_statistics

try:
    import xformers  # type: ignore
    del xformers
    is_xformers_installed = True
except:
    is_xformers_installed = False


class DiffusionV1(ComposerModel):
    """Stable Diffusion ComposerModel.

    This is a Latent Diffusion model conditioned on text prompts that are run through
    a pre-trained CLIP or LLM model. The CLIP outputs are then passed to as an
    additional input to our Unet during training and can later be used to guide
    the image generation process.

    Args:
        unet (torch.nn.Module): HuggingFace conditional unet, must accept a
            (B, C, H, W) input, (B,) timestep array of noise timesteps,
            and (B, 77, 768) text conditioning vectors.
        vae (torch.nn.Module): HuggingFace or compatible vae.
            must support `.encode()` and `decode()` functions.
        noise_scheduler (diffusers.SchedulerMixin): HuggingFace diffusers
            noise scheduler. Used during the forward diffusion process (training).
        inference_scheduler (diffusers.SchedulerMixin): HuggingFace diffusers
            noise scheduler. Used during the backward diffusion process (inference).
        loss_fn (torch.nn.Module): torch loss function. Default: `F.mse_loss`.
        prediction_type (str): The type of prediction to use. Must be one of 'sample',
            'epsilon', or 'v_prediction'. Default: `epsilon`.
        latent_mean (Optional[tuple[float]]): The means of the latent space. If not specified, defaults to
            . Default: ``(0.0,) * 4``.
        latent_std (Optional[tuple[float]]): The standard deviations of the latent space. Default: ``(1/0.13025,)*4``.
        downsample_factor (int): The factor by which the image is downsampled by the autoencoder. Default `8`.
        train_metrics (list): List of torchmetrics to calculate during training.
            Default: `None`.
        val_metrics (list): List of torchmetrics to calculate during validation.
            Default: `None`.
        quasirandomness (bool): Whether to use quasirandomness for generating diffusion process noise.
            Default: `False`.
        train_seed (int): Seed to use for generating diffusion process noise during training if using
            quasirandomness. Default: `42`.
        val_seed (int): Seed to use for generating eval images. Default: `1138`.
        fsdp (bool): whether to use FSDP, Default: `False`.
    """

    def __init__(self,
                 unet,
                 vae,
                 noise_scheduler,
                 inference_noise_scheduler,
                 loss_fn=F.mse_loss,
                 prediction_type: str = 'epsilon',
                 latent_mean: Tuple[float] = (0.0,) * 4,
                 latent_std: Tuple[float] = (1 / 0.13025,) * 4,
                 downsample_factor: int = 8,
                 train_metrics: Optional[List] = None,
                 val_metrics: Optional[List] = None,
                 quasirandomness: bool = False,
                 train_seed: int = 42,
                 val_seed: int = 1138,
                 text_embed_dim: int = 4096,
                 fsdp: bool = False,
                 ):
        super().__init__()
        self.unet = unet
        self.vae = vae
        self.noise_scheduler = noise_scheduler
        self.loss_fn = loss_fn
        self.prediction_type = prediction_type.lower()
        if self.prediction_type not in ['sample', 'epsilon', 'v_prediction']:
            raise ValueError(f'prediction type must be one of sample, epsilon, or v_prediction. Got {prediction_type}')
        self.downsample_factor = downsample_factor
        self.quasirandomness = quasirandomness
        self.train_seed = train_seed
        self.val_seed = val_seed
        self.latent_mean = latent_mean
        self.latent_std = latent_std
        self.latent_mean = torch.tensor(latent_mean).view(1, -1, 1, 1)
        self.latent_std = torch.tensor(latent_std).view(1, -1, 1, 1)
        self.train_metrics = train_metrics if train_metrics is not None else [MeanSquaredError()]
        self.val_metrics = val_metrics if val_metrics is not None else [MeanSquaredError()]
        self.inference_scheduler = inference_noise_scheduler
        # freeze VAE during diffusion training
        self.vae.requires_grad_(False)
        self.vae = self.vae.half()
        if fsdp:
            # only wrap models we are training
            self.vae._fsdp_wrap = False
            self.unet._fsdp_wrap = True

        # Optional rng generator
        self.rng_generator: Optional[torch.Generator] = None
        if self.quasirandomness:
            self.sobol = qmc.Sobol(d=1, scramble=True, seed=self.train_seed)

        self.clip_proj = nn.Linear(768, text_embed_dim)
        self.t5_proj = nn.Linear(4096, text_embed_dim)

    def _apply(self, fn):
        super(DiffusionV1, self)._apply(fn)
        self.latent_mean = fn(self.latent_mean)
        self.latent_std = fn(self.latent_std)
        return self

    def _generate_timesteps(self, latents: torch.Tensor):
        if self.quasirandomness:
            # Generate a quasirandom sequence of timesteps equal to the global batch size
            global_batch_size = latents.shape[0] * dist.get_world_size()
            sampled_fractions = torch.tensor(self.sobol.random(global_batch_size), device=latents.device)
            timesteps = (len(self.noise_scheduler) * sampled_fractions).squeeze()
            timesteps = torch.floor(timesteps).long()
            # Get this device's subset of all the timesteps
            idx_offset = dist.get_global_rank() * latents.shape[0]
            timesteps = timesteps[idx_offset:idx_offset + latents.shape[0]].to(latents.device)
        else:
            timesteps = torch.randint(0,
                                      len(self.noise_scheduler), (latents.shape[0],),
                                      device=latents.device,
                                      generator=self.rng_generator)
        return timesteps

    def set_rng_generator(self, rng_generator: torch.Generator):
        """Sets the rng generator for the model."""
        self.rng_generator = rng_generator

    def forward(self, batch):
        latents, text_embeds, text_pooled_embeds, encoder_attention_mask = None, None, None, None

        inputs = batch['image']
        with torch.cuda.amp.autocast(enabled=False):
            latents = self.vae.encode(inputs.half())['latent_dist'].sample().data
        latents = (latents - self.latent_mean) / self.latent_std # scale latents

        t5_embed = self.t5_proj(batch['T5_LATENTS'])
        clip_embed = self.clip_proj(batch['CLIP_LATENTS'])
        text_embeds = torch.cat([t5_embed, clip_embed], dim=1)
        text_pooled_embeds = batch['CLIP_POOLED']

        encoder_attention_mask = torch.cat([batch['T5_ATTENTION_MASK'], batch['CLIP_ATTENTION_MASK']], dim=1)

        # Sample the diffusion timesteps
        timesteps = self._generate_timesteps(latents)
        # Add noise to the inputs (forward diffusion)
        noise = torch.randn(*latents.shape, device=latents.device, generator=self.rng_generator)
        noised_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
        # Generate the targets
        if self.prediction_type == 'epsilon':
            targets = noise
        elif self.prediction_type == 'sample':
            targets = latents
        elif self.prediction_type == 'v_prediction':
            targets = self.noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(
                f'prediction type must be one of sample, epsilon, or v_prediction. Got {self.prediction_type}')

        # Prepare added time ids & embeddings
        add_time_ids = torch.cat(
            [batch['cond_original_size'], batch['cond_crops_coords_top_left'], batch['cond_target_size']], dim=1)
        added_cond_kwargs = {'text_embeds': text_pooled_embeds, 'time_ids': add_time_ids}

        # Forward through the model
        return self.unet(noised_latents,
                         timesteps,
                         text_embeds,
                         encoder_attention_mask=encoder_attention_mask,
                         added_cond_kwargs=added_cond_kwargs)['sample'], targets, timesteps

    def loss(self, outputs, batch):
        """Loss between unet output and added noise, typically mse."""
        return self.loss_fn(outputs[0], outputs[1])

    def eval_forward(self, batch, outputs=None):
        """For stable diffusion, eval forward computes unet outputs as well as some samples."""
        # Skip this if outputs have already been computed, e.g. during training
        if outputs is not None:
            return outputs
        return self.forward(batch)

    def get_metrics(self, is_train: bool = False):
        if is_train:
            metrics = self.train_metrics
        else:
            metrics = self.val_metrics
        metrics_dict = {metric.__class__.__name__: metric for metric in metrics}
        return metrics_dict

    def update_metric(self, batch, outputs, metric):
        metric.update(outputs[0], outputs[1])

    @torch.no_grad()
    def generate(
        self,
        prompt_embeds: torch.FloatTensor,
        pooled_prompt: torch.FloatTensor,
        prompt_mask: torch.LongTensor,
        neg_prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_neg_prompt: Optional[torch.FloatTensor] = None,
        neg_prompt_mask: Optional[torch.LongTensor] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: Optional[int] = 50,
        guidance_scale: Optional[float] = 3.0,
        rescaled_guidance: Optional[float] = None,
        num_images_per_prompt: Optional[int] = 1,
        seed: Optional[int] = None,
        progress_bar: Optional[bool] = True,
        crop_params: Optional[torch.Tensor] = None,
        input_size_params: Optional[torch.Tensor] = None,
    ):
        """Generates image from noise.

        Performs the backward diffusion process, each inference step takes
        one forward pass through the unet.

        Args:
            prompt (str or List[str]): The prompt or prompts to guide the image generation.
            negative_prompt (str or List[str]): The prompt or prompts to guide the
                image generation away from. Ignored when not using guidance
                (i.e., ignored if guidance_scale is less than 1).
                Must be the same length as list of prompts. Default: `None`.
            tokenized_prompts (torch.LongTensor): Optionally pass pre-tokenized prompts instead
                of string prompts. If SDXL, this will be a tensor of size [B, 2, max_length],
                otherwise will be of shape [B, max_length]. Default: `None`.
            tokenized_negative_prompts (torch.LongTensor): Optionally pass pre-tokenized negative
                prompts instead of string prompts. Default: `None`.
            tokenized_prompts_pad_mask (torch.LongTensor): Optionally pass padding mask for
                pre-tokenized prompts. Default `None`.
            tokenized_negative_prompts_pad_mask (torch.LongTensor): Optionall pass padding mask for
                pre-tokenized negative prompts. Default `None`.
            prompt_embeds (torch.FloatTensor): Optionally pass pre-tokenized prompts instead
                of string prompts. If both prompt and prompt_embeds
                are passed, prompt_embeds will be used. Default: `None`.
            negative_prompt_embeds (torch.FloatTensor): Optionally pass pre-embedded negative
                prompts instead of string negative prompts. If both negative_prompt and
                negative_prompt_embeds are passed, prompt_embeds will be used.  Default: `None`.
            height (int, optional): The height in pixels of the generated image.
                Default: `self.unet.config.sample_size * 8)`.
            width (int, optional): The width in pixels of the generated image.
                Default: `self.unet.config.sample_size * 8)`.
            num_inference_steps (int): The number of denoising steps.
                More denoising steps usually lead to a higher quality image at the expense
                of slower inference. Default: `50`.
            guidance_scale (float): Guidance scale as defined in
                Classifier-Free Diffusion Guidance. guidance_scale is defined as w of equation
                2. of Imagen Paper. Guidance scale is enabled by setting guidance_scale > 1.
                Higher guidance scale encourages to generate images that are closely linked
                to the text prompt, usually at the expense of lower image quality.
                Default: `3.0`.
            rescaled_guidance (float, optional): Rescaled guidance scale. If not specified, rescaled guidance will
                not be used. Default: `None`.
            num_images_per_prompt (int): The number of images to generate per prompt.
                 Default: `1`.
            progress_bar (bool): Whether to use the tqdm progress bar during generation.
                Default: `True`.
            seed (int): Random seed to use for generation. Set a seed for reproducible generation.
                Default: `None`.
            crop_params (torch.FloatTensor of size [Bx2], optional): Crop parameters to use
                when generating images with SDXL. Default: `None`.
            input_size_params (torch.FloatTensor of size [Bx2], optional): Size parameters
                (representing original size of input image) to use when generating images with SDXL.
                Default: `None`.
        """

        # TODO: do checks
        # if prompt_embeds.shape[:2] == prompt_mask.shape[:2]:
        #     raise ValueError(' ')
        
        # Check all parts of negative prompts exist and are equal length
        # if neg_prompt_embeds is not None or neg_prompt_mask is not None or pooled_neg_prompt is not None:

        # if negative_negative_embedlen(prompt_embeds) != len(negative_prompt_embeds):
        #     raise ValueError('len(prompts) and len(negative_prompts) must be the same. \
        #             A negative prompt must be provided for each given prompt.')        

        # Create rng for the generation
        device = self.vae.device
        rng_generator = torch.Generator(device=device)
        if seed:
            rng_generator = rng_generator.manual_seed(seed)  # type: ignore

        if height is None:
            height = self.unet.config.sample_size * self.downsample_factor
        if width is None:
            width = self.unet.config.sample_size * self.downsample_factor

        do_classifier_free_guidance = guidance_scale > 1.0  # type: ignore

        text_embeddings = _duplicate_tensor(prompt_embeds, num_images_per_prompt)
        pooled_embeddings = _duplicate_tensor(pooled_prompt, num_images_per_prompt)
        encoder_attn_mask = _duplicate_tensor(prompt_mask, num_images_per_prompt)

        batch_size = len(prompt_embeds)  # len prompts * num_images_per_prompt
        # classifier free guidance + negative prompts
        # negative prompt is given in place of the unconditional input in classifier free guidance
        if do_classifier_free_guidance:
            if not neg_prompt_embeds:
                # Negative prompt is empty and we want to zero it out
                neg_prompt_embeds = torch.zeros_like(text_embeddings)
                pooled_neg_prompt = torch.zeros_like(pooled_embeddings)
                neg_prompt_mask = torch.zeros_like(encoder_attn_mask)
            else:
                neg_prompt_embeds = _duplicate_tensor(neg_prompt_embeds, num_images_per_prompt)
                pooled_neg_prompt = _duplicate_tensor(pooled_neg_prompt, num_images_per_prompt)
                neg_prompt_mask = _duplicate_tensor(neg_prompt_mask, num_images_per_prompt)
            
            # concat uncond + prompt
            text_embeddings = torch.cat([neg_prompt_embeds, text_embeddings])
            pooled_embeddings = torch.cat([pooled_neg_prompt, pooled_embeddings])
            encoder_attn_mask = torch.cat([neg_prompt_mask, encoder_attn_mask])

        # prepare for diffusion generation process
        latents = torch.randn(
            (batch_size, self.unet.config.in_channels, height // self.downsample_factor,
             width // self.downsample_factor),
            device=device,
            dtype=self.unet.dtype,
            generator=rng_generator,
        )

        self.inference_scheduler.set_timesteps(num_inference_steps)
        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.inference_scheduler.init_noise_sigma

        added_cond_kwargs = {}
        # if using SDXL, prepare added time ids & embeddings

        if crop_params is None:
            crop_params = torch.zeros((batch_size, 2), dtype=text_embeddings.dtype)
        if input_size_params is None:
            input_size_params = torch.tensor([[width, height]] * batch_size, dtype=text_embeddings.dtype)
        output_size_params = torch.tensor([[width, height]] * batch_size, dtype=text_embeddings.dtype)

        if do_classifier_free_guidance:
            crop_params = torch.cat([crop_params, crop_params])
            input_size_params = torch.cat([input_size_params, input_size_params])
            output_size_params = torch.cat([output_size_params, output_size_params])

        add_time_ids = torch.cat([input_size_params, crop_params, output_size_params], dim=1).to(device)
        added_cond_kwargs = {'text_embeds': pooled_embeddings, 'time_ids': add_time_ids}

        # backward diffusion process
        for t in tqdm(self.inference_scheduler.timesteps, disable=not progress_bar):
            if do_classifier_free_guidance:
                latent_model_input = torch.cat([latents] * 2)
            else:
                latent_model_input = latents

            latent_model_input = self.inference_scheduler.scale_model_input(latent_model_input, t)
            # Model prediction
            pred = self.unet(latent_model_input,
                             t,
                             encoder_hidden_states=text_embeddings,
                             encoder_attention_mask=encoder_attn_mask,
                             added_cond_kwargs=added_cond_kwargs).sample

            if do_classifier_free_guidance:
                # perform guidance. Note this is only techincally correct for prediction_type 'epsilon'
                pred_uncond, pred_text = pred.chunk(2)
                pred = pred_uncond + guidance_scale * (pred_text - pred_uncond)
                # Optionally rescale the classifer free guidance
                if rescaled_guidance is not None:
                    std_pos = torch.std(pred_text, dim=(1, 2, 3), keepdim=True)
                    std_cfg = torch.std(pred, dim=(1, 2, 3), keepdim=True)
                    pred_rescaled = pred * (std_pos / std_cfg)
                    pred = pred_rescaled * rescaled_guidance + pred * (1 - rescaled_guidance)
            # compute the previous noisy sample x_t -> x_t-1
            latents = self.inference_scheduler.step(pred, t, latents, generator=rng_generator).prev_sample

        # We now use the vae to decode the generated latents back into the image.
        # scale and decode the image latents with vae
        latents = latents * self.latent_std + self.latent_mean
        image = self.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        return image.detach()  # (batch*num_images_per_prompt, channel, h, w)


def _duplicate_tensor(tensor, num_images_per_prompt):
    """Duplicate tensor for multiple generations from a single prompt."""
    batch_size, seq_len = tensor.shape[:2]
    tensor = tensor.repeat(1, num_images_per_prompt, *[
        1,
    ] * len(tensor.shape[2:]))
    return tensor.view(batch_size * num_images_per_prompt, seq_len, *[
        -1,
    ] * len(tensor.shape[2:]))

def build_diffusion_v1(
    unet_model_name: str = 'stabilityai/stable-diffusion-xl-base-1.0',
    vae_model_name: str = 'madebyollin/sdxl-vae-fp16-fix',
    autoencoder_path: Optional[str] = None,
    autoencoder_local_path: str = '/tmp/autoencoder_weights.pt',
    prediction_type: str = 'epsilon',
    latent_mean: Union[float, Tuple, str] = 0.0,
    latent_std: Union[float, Tuple, str] = 7.67754318618,
    text_embed_dim: int = 4096,
    beta_schedule: str = 'scaled_linear',
    zero_terminal_snr: bool = False,
    train_metrics: Optional[List] = None,
    val_metrics: Optional[List] = None,
    quasirandomness: bool = False,
    train_seed: int = 42,
    val_seed: int = 1138,
    fsdp: bool = True,
    use_xformers: bool = True,
):
    """Stable diffusion 2 training setup + SDXL UNet and VAE.

    Requires batches of matched images and text prompts to train. Generates images from text
    prompts. Currently uses UNet and VAE config from SDXL, but text encoder/tokenizer from SD2.

    Args:
        unet_model_name (str): Name of the UNet model to load. Defaults to
            'stabilityai/stable-diffusion-xl-base-1.0'.
        vae_model_name (str): Name of the VAE model to load. Defaults to
            'madebyollin/sdxl-vae-fp16-fix' as the official VAE checkpoint (from
            'stabilityai/stable-diffusion-xl-base-1.0') is not compatible with fp16.
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
        fsdp (bool): Whether to use FSDP. Defaults to True.
        use_xformers (bool): Whether to use xformers for attention. Defaults to True.
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
    # This config variable is the sum of the text encoder projection dimension (768 for CLIP) and
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
    if beta_schedule == 'squaredcos_cap_v2':
        inference_noise_scheduler = DDIMScheduler(num_train_timesteps=1000,
                                                  beta_start=0.00085,
                                                  beta_end=0.012,
                                                  beta_schedule=beta_schedule,
                                                  trained_betas=None,
                                                  clip_sample=False,
                                                  set_alpha_to_one=False,
                                                  prediction_type=prediction_type,
                                                  rescale_betas_zero_snr=zero_terminal_snr)
    else:
        inference_noise_scheduler = EulerDiscreteScheduler(num_train_timesteps=1000,
                                                           beta_start=0.00085,
                                                           beta_end=0.012,
                                                           beta_schedule=beta_schedule,
                                                           trained_betas=None,
                                                           prediction_type=prediction_type,
                                                           interpolation_type='linear',
                                                           use_karras_sigmas=False,
                                                           timestep_spacing='leading',
                                                           steps_offset=1,
                                                           rescale_betas_zero_snr=zero_terminal_snr)

    # Make the composer model
    model = DiffusionV1(
        unet=unet,
        vae=vae,
        noise_scheduler=noise_scheduler,
        inference_noise_scheduler=inference_noise_scheduler,
        prediction_type=prediction_type,
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
    if torch.cuda.is_available():
        model = DeviceGPU().module_to_device(model)
        if is_xformers_installed and use_xformers:
            model.unet.enable_xformers_memory_efficient_attention()
            if hasattr(model.vae, 'enable_xformers_memory_efficient_attention'):
                model.vae.enable_xformers_memory_efficient_attention()

    return model