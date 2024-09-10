# Copyright 2022 MosaicML Diffusion authors
# SPDX-License-Identifier: Apache-2.0

"""Diffusion models."""

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from composer.models import ComposerModel
from composer.utils import dist
from scipy.stats import qmc
from torchmetrics import MeanSquaredError
from tqdm.auto import tqdm
from transformers import PreTrainedTokenizer

try:
    import xformers  # type: ignore
    del xformers
    is_xformers_installed = True
except:
    is_xformers_installed = False


class PrecomputedTextLatentDiffusion(ComposerModel):
    """Diffusion ComposerModel for running with precomputed T5 and CLIP embeddings.

    This is a Latent Diffusion model conditioned on text prompts that are run through
    a pre-trained CLIP and T5 text encoder.

    Args:
        unet (torch.nn.Module): HuggingFace conditional unet, must accept a
            (B, C, H, W) input, (B,) timestep array of noise timesteps,
            and (B, 77, text_embed_dim) text conditioning vectors.
        vae (torch.nn.Module): HuggingFace or compatible vae.
            must support `.encode()` and `decode()` functions.
        noise_scheduler (diffusers.SchedulerMixin): HuggingFace diffusers
            noise scheduler. Used during the forward diffusion process (training).
        inference_scheduler (diffusers.SchedulerMixin): HuggingFace diffusers
            noise scheduler. Used during the backward diffusion process (inference).
        t5_tokenizer (Optional): Tokenizer for T5. Should only be specified during inference. Default: `None`.
        t5_encoder (Optional): T5 text encoder. Should only be specified during inference. Default: `None`.
        clip_tokenizer (Optional): Tokenizer for CLIP. Should only be specified during inference. Default: `None`.
        clip_encoder (Optional): CLIP text encoder. Should only be specified during inference. Default: `None`.
        text_embed_dim (int): The common dimension to project the text embeddings to. Default: `4096`.
        prediction_type (str): The type of prediction to use. Must be one of 'sample',
            'epsilon', or 'v_prediction'. Default: `epsilon`.
        latent_mean (Optional[tuple[float]]): The means of the latent space. If not specified, defaults to
            . Default: ``(0.0,) * 4``.
        latent_std (Optional[tuple[float]]): The standard deviations of the latent space. Default: ``(1/0.13025,)*4``.
        downsample_factor (int): The factor by which the image is downsampled by the autoencoder. Default `8`.
        max_seq_len (int): The maximum sequence length for the text encoder. Default: `256`.
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

    def __init__(
        self,
        unet,
        vae,
        noise_scheduler,
        inference_noise_scheduler,
        t5_tokenizer: Optional[PreTrainedTokenizer] = None,
        t5_encoder: Optional[torch.nn.Module] = None,
        clip_tokenizer: Optional[PreTrainedTokenizer] = None,
        clip_encoder: Optional[torch.nn.Module] = None,
        text_embed_dim: int = 4096,
        prediction_type: str = 'epsilon',
        latent_mean: Tuple[float] = (0.0,) * 4,
        latent_std: Tuple[float] = (1 / 0.13025,) * 4,
        downsample_factor: int = 8,
        max_seq_len: int = 256,
        train_metrics: Optional[List] = None,
        val_metrics: Optional[List] = None,
        quasirandomness: bool = False,
        train_seed: int = 42,
        val_seed: int = 1138,
        fsdp: bool = False,
    ):
        super().__init__()
        self.unet = unet
        self.vae = vae
        self.t5_tokenizer = t5_tokenizer
        self.t5_encoder = t5_encoder
        self.clip_tokenizer = clip_tokenizer
        self.clip_encoder = clip_encoder
        self.noise_scheduler = noise_scheduler
        self.prediction_type = prediction_type.lower()
        if self.prediction_type not in ['sample', 'epsilon', 'v_prediction']:
            raise ValueError(f'prediction type must be one of sample, epsilon, or v_prediction. Got {prediction_type}')
        self.downsample_factor = downsample_factor
        self.max_seq_len = max_seq_len
        self.quasirandomness = quasirandomness
        self.train_seed = train_seed
        self.val_seed = val_seed
        self.latent_mean = torch.tensor(latent_mean).view(1, -1, 1, 1)
        self.latent_std = torch.tensor(latent_std).view(1, -1, 1, 1)
        self.train_metrics = train_metrics if train_metrics is not None else [MeanSquaredError()]
        self.val_metrics = val_metrics if val_metrics is not None else [MeanSquaredError()]
        self.inference_scheduler = inference_noise_scheduler
        # freeze VAE during diffusion training
        self.vae.requires_grad_(False)
        self.vae = self.vae.bfloat16()
        if fsdp:
            # only wrap models we are training
            self.vae._fsdp_wrap = False
            self.unet._fsdp_wrap = True

        # Optional rng generator
        self.rng_generator: Optional[torch.Generator] = None
        if self.quasirandomness:
            self.sobol = qmc.Sobol(d=1, scramble=True, seed=self.train_seed)

        # Projection layers for the text embeddings
        self.clip_proj = nn.Linear(768, text_embed_dim)
        self.t5_proj = nn.Linear(4096, text_embed_dim)
        # Layernorms for the text embeddings
        self.clip_ln = nn.LayerNorm(text_embed_dim)
        self.t5_ln = nn.LayerNorm(text_embed_dim)
        # Learnable position embeddings for the conitioning sequences
        t5_position_embeddings = torch.randn(self.max_seq_len, text_embed_dim)
        t5_position_embeddings /= math.sqrt(text_embed_dim)
        self.t5_position_embedding = torch.nn.Parameter(t5_position_embeddings, requires_grad=True)
        clip_position_embeddings = torch.randn(self.max_seq_len, text_embed_dim)
        clip_position_embeddings /= math.sqrt(text_embed_dim)
        self.clip_position_embedding = torch.nn.Parameter(clip_position_embeddings, requires_grad=True)

    def _apply(self, fn):
        super(PrecomputedTextLatentDiffusion, self)._apply(fn)
        self.latent_mean = fn(self.latent_mean)
        self.latent_std = fn(self.latent_std)
        return self

    def _generate_timesteps(self, latents: torch.Tensor):
        if not self.unet.training:
            # Sample equally spaced timesteps across all devices
            global_batch_size = latents.shape[0] * dist.get_world_size()
            global_timesteps = torch.linspace(0, len(self.noise_scheduler) - 1, global_batch_size).to(torch.int64)
            # Get this device's subset of all the timesteps
            idx_offset = dist.get_global_rank() * latents.shape[0]
            timesteps = global_timesteps[idx_offset:idx_offset + latents.shape[0]].to(latents.device)
        else:
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

    def encode_images(self, inputs, dtype=torch.bfloat16):
        with torch.amp.autocast('cuda', enabled=False):
            latents = self.vae.encode(inputs.to(dtype))['latent_dist'].sample().data
        latents = (latents - self.latent_mean) / self.latent_std  # scale latents
        return latents

    def decode_latents(self, latents):
        latents = latents * self.latent_std + self.latent_mean
        image = self.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        return image

    def encode_text(self, text, device):
        assert self.t5_tokenizer is not None and self.t5_encoder is not None
        assert self.clip_tokenizer is not None and self.clip_encoder is not None
        # Encode with T5
        t5_tokenizer_out = self.t5_tokenizer(text,
                                             padding='max_length',
                                             max_length=self.t5_tokenizer.model_max_length,
                                             truncation=True,
                                             return_tensors='pt')
        tokenized_captions = t5_tokenizer_out['input_ids'].to(device)
        t5_attn_mask = t5_tokenizer_out['attention_mask'].to(torch.bool).to(device)
        t5_embed = self.t5_encoder(input_ids=tokenized_captions, attention_mask=t5_attn_mask)
        # Encode with CLIP
        clip_tokenizer_out = self.clip_tokenizer(text,
                                                 padding='max_length',
                                                 max_length=self.clip_tokenizer.model_max_length,
                                                 truncation=True,
                                                 return_tensors='pt')
        tokenized_captions = clip_tokenizer_out['input_ids'].to(device)
        clip_attn_mask = clip_tokenizer_out['attention_mask'].to(torch.bool).to(device)
        clip_out = self.clip_encoder(input_ids=tokenized_captions,
                                     attention_mask=clip_attn_mask,
                                     output_hidden_states=True)
        clip_embed = clip_out.hidden_states[-2]
        pooled_embeddings = clip_out[1]
        return t5_embed, clip_embed, t5_attn_mask, clip_attn_mask, pooled_embeddings

    def prepare_text_embeddings(self, t5_embed: torch.Tensor, clip_embed: torch.Tensor, t5_mask: torch.Tensor,
                                clip_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if t5_embed.shape[1] > self.max_seq_len:
            t5_embed = t5_embed[:, :self.max_seq_len]
            t5_mask = t5_mask[:, :self.max_seq_len]
        if clip_embed.shape[1] > self.max_seq_len:
            clip_embed = clip_embed[:, :self.max_seq_len]
            clip_mask = clip_mask[:, :self.max_seq_len]
        t5_embed = self.t5_proj(t5_embed)
        clip_embed = self.clip_proj(clip_embed)
        # Add position embeddings
        t5_embed = 0.707 * t5_embed + 0.707 * self.t5_position_embedding[:t5_embed.shape[1]].unsqueeze(0)
        clip_embed = 0.707 * clip_embed + 0.707 * self.clip_position_embedding[:clip_embed.shape[1]].unsqueeze(0)
        # Apply layernorms
        t5_embed = self.t5_ln(t5_embed)
        clip_embed = self.clip_ln(clip_embed)
        # Concatenate the text embeddings
        text_embeds = torch.cat([t5_embed, clip_embed], dim=1)
        encoder_attention_mask = torch.cat([t5_mask, clip_mask], dim=1)
        return text_embeds, encoder_attention_mask

    def diffusion_forward(self, latents, timesteps):
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
        return noised_latents, targets

    def forward(self, batch):
        latents, text_embeds, text_pooled_embeds, encoder_attention_mask = None, None, None, None

        # Encode the images with the autoencoder encoder
        inputs = batch['image']
        latents = self.encode_images(inputs)

        # Text embeddings are shape (B, seq_len, emb_dim), optionally truncate to a max length
        t5_embed = batch['T5_LATENTS']
        t5_mask = batch['T5_ATTENTION_MASK']
        clip_embed = batch['CLIP_LATENTS']
        clip_mask = batch['CLIP_ATTENTION_MASK']
        text_pooled_embeds = batch['CLIP_POOLED']
        text_embeds, encoder_attention_mask = self.prepare_text_embeddings(t5_embed, clip_embed, t5_mask, clip_mask)

        # Sample the diffusion timesteps
        timesteps = self._generate_timesteps(latents)
        noised_latents, targets = self.diffusion_forward(latents, timesteps)

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
        loss = F.mse_loss(outputs[0], outputs[1])
        return loss

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
        prompt: Optional[list] = None,
        negative_prompt: Optional[list] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        pooled_prompt: Optional[torch.Tensor] = None,
        prompt_mask: Optional[torch.Tensor] = None,
        neg_prompt_embeds: Optional[torch.Tensor] = None,
        pooled_neg_prompt: Optional[torch.Tensor] = None,
        neg_prompt_mask: Optional[torch.Tensor] = None,
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 50,
        guidance_scale: float = 3.0,
        rescaled_guidance: Optional[float] = None,
        num_images_per_prompt: int = 1,
        seed: Optional[int] = None,
        progress_bar: bool = True,
        crop_params: Optional[torch.Tensor] = None,
        input_size_params: Optional[torch.Tensor] = None,
    ):
        """Generates image from noise.

        Performs the backward diffusion process, each inference step takes
        one forward pass through the unet.

        Args:
            prompt (List[str]): The prompts to guide the image generation. Only use if not
                using embeddings. Default: `None`.
            negative_prompt (str or List[str]): The prompt or prompts to guide the
                image generation away from. Ignored when not using guidance
                (i.e., ignored if guidance_scale is less than 1). Must be the same length
                as list of prompts. Only use if not using negative embeddings. Default: `None`.
            prompt_embeds (torch.Tensor): Optionally pass pre-tokenized prompts instead
                of string prompts. Default: `None`.
            pooled_prompt (torch.Tensor): Optionally pass a precomputed pooled prompt embedding
                if using embeddings. Default: `None`.
            prompt_mask (torch.Tensor): Optionally pass a precomputed attention mask for the
                prompt embeddings. Default: `None`.
            neg_prompt_embeds (torch.Tensor): Optionally pass pre-embedded negative
                prompts instead of string negative prompts.  Default: `None`.
            pooled_neg_prompt (torch.Tensor): Optionally pass a precomputed pooled negative
                prompt embedding if using embeddings. Default: `None`.
            neg_prompt_mask (torch.Tensor): Optionally pass a precomputed attention mask for the
                negative prompt embeddings. Default: `None`.
            height (int, optional): The height in pixels of the generated image.
                Default: `1024`.
            width (int, optional): The width in pixels of the generated image.
                Default: `1024`.
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
            crop_params (torch.Tensor of size [Bx2], optional): Crop parameters to use
                when generating images with SDXL. Default: `None`.
            input_size_params (torch.Tensor of size [Bx2], optional): Size parameters
                (representing original size of input image) to use when generating images with SDXL.
                Default: `None`.
        """
        # Create rng for the generation
        device = self.vae.device
        rng_generator = torch.Generator(device=device)
        if seed:
            rng_generator = rng_generator.manual_seed(seed)

        # Check that inputs are consistent with all embeddings or text inputs. All embeddings should be provided if using
        # embeddings, and none if using text.
        if (prompt_embeds is None) == (prompt is None):
            raise ValueError('One and only one of prompt or prompt_embeds should be provided.')
        if (pooled_prompt is None) != (prompt_embeds is None):
            raise ValueError('pooled_prompt should be provided if and only if using embeddings')
        if (prompt_mask is None) != (prompt_embeds is None):
            raise ValueError('prompt_mask should be provided if and only if using embeddings')
        if (neg_prompt_mask is None) != (neg_prompt_embeds is None):
            raise ValueError('neg_prompt_mask should be provided if and only if using embeddings')
        if (pooled_neg_prompt is None) != (neg_prompt_embeds is None):
            raise ValueError('pooled_neg_prompt should be provided if and only if using embeddings')

        # If the prompt is specified as text, encode it.
        if prompt is not None:
            t5_embed, clip_embed, t5_attn_mask, clip_attn_mask, pooled_prompt = self.encode_text(
                prompt, self.vae.device)
            prompt_embeds, prompt_mask = self.prepare_text_embeddings(t5_embed, clip_embed, t5_attn_mask,
                                                                      clip_attn_mask)
        # If negative prompt is specified as text, encode it.
        if negative_prompt is not None:
            t5_embed, clip_embed, t5_attn_mask, clip_attn_mask, pooled_neg_prompt = self.encode_text(
                negative_prompt, self.vae.device)
            neg_prompt_embeds, neg_prompt_mask = self.prepare_text_embeddings(t5_embed, clip_embed, t5_attn_mask,
                                                                              clip_attn_mask)

        text_embeddings = _duplicate_tensor(prompt_embeds, num_images_per_prompt)
        pooled_embeddings = _duplicate_tensor(pooled_prompt, num_images_per_prompt)
        encoder_attn_mask = _duplicate_tensor(prompt_mask, num_images_per_prompt)

        batch_size = len(text_embeddings)  # len prompts * num_images_per_prompt
        # classifier free guidance + negative prompts
        # negative prompt is given in place of the unconditional input in classifier free guidance
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

        crop_params = torch.cat([crop_params, crop_params])
        input_size_params = torch.cat([input_size_params, input_size_params])
        output_size_params = torch.cat([output_size_params, output_size_params])

        add_time_ids = torch.cat([input_size_params, crop_params, output_size_params], dim=1).to(device)
        added_cond_kwargs = {'text_embeds': pooled_embeddings, 'time_ids': add_time_ids}

        # backward diffusion process
        for t in tqdm(self.inference_scheduler.timesteps, disable=not progress_bar):
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = self.inference_scheduler.scale_model_input(latent_model_input, t)
            # Model prediction
            pred = self.unet(latent_model_input,
                             t,
                             encoder_hidden_states=text_embeddings,
                             encoder_attention_mask=encoder_attn_mask,
                             added_cond_kwargs=added_cond_kwargs).sample

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
        image = self.decode_latents(latents)
        return image.detach().float()  # (batch*num_images_per_prompt, channel, h, w)


def _duplicate_tensor(tensor, num_images_per_prompt):
    """Duplicate tensor for multiple generations from a single prompt."""
    batch_size, seq_len = tensor.shape[:2]
    tensor = tensor.repeat(1, num_images_per_prompt, *[
        1,
    ] * len(tensor.shape[2:]))
    return tensor.view(batch_size * num_images_per_prompt, seq_len, *[
        -1,
    ] * len(tensor.shape[2:]))
