# Copyright 2022 MosaicML Diffusion authors
# SPDX-License-Identifier: Apache-2.0

"""Diffusion models."""

from contextlib import nullcontext
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from composer.models import ComposerModel
from composer.utils import dist
from scipy.stats import qmc
from torchmetrics import MeanSquaredError
from tqdm.auto import tqdm


class StableDiffusion(ComposerModel):
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
        text_encoder (torch.nn.Module): HuggingFace CLIP or LLM text enoder.
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer used for
            text_encoder. For a `CLIPTextModel` this will be the
            `CLIPTokenizer` from HuggingFace transformers.
        noise_scheduler (diffusers.SchedulerMixin): HuggingFace diffusers
            noise scheduler. Used during the forward diffusion process (training).
        inference_scheduler (diffusers.SchedulerMixin): HuggingFace diffusers
            noise scheduler. Used during the backward diffusion process (inference).
        num_images_per_prompt (int): How many images to generate per prompt
            for evaluation. Default: `1`.
        loss_fn (torch.nn.Module): torch loss function. Default: `F.mse_loss`.
        prediction_type (str): The type of prediction to use. Must be one of 'sample',
            'epsilon', or 'v_prediction'. Default: `epsilon`.
        latent_mean (Optional[tuple[float]]): The means of the latent space. If not specified, defaults to
            4 * (0.0,). Default: `None`.
        latent_std (Optional[tuple[float]]): The standard deviations of the latent space. If not specified,
            defaults to 4 * (1/0.13025,) for SDXL, or 4 * (1/0.18215,) for non-SDXL. Default: `None`.
        downsample_factor (int): The factor by which the image is downsampled by the autoencoder. Default `8`.
        offset_noise (float, optional): The scale of the offset noise. If not specified, offset noise will not
            be used. Default `None`.
        train_metrics (list): List of torchmetrics to calculate during training.
            Default: `None`.
        val_metrics (list): List of torchmetrics to calculate during validation.
            Default: `None`.
        quasirandomness (bool): Whether to use quasirandomness for generating diffusion process noise.
            Default: `False`.
        train_seed (int): Seed to use for generating diffusion process noise during training if using
            quasirandomness. Default: `42`.
        val_seed (int): Seed to use for generating eval images. Default: `1138`.
        image_key (str): The name of the image inputs in the dataloader batch.
            Default: `image_tensor`.
        caption_key (str): The name of the caption inputs in the dataloader batch.
            Default: `input_ids`.
        fsdp (bool): whether to use FSDP, Default: `False`.
        image_latents_key (str): key in batch dict for image latents.
            Default: `image_latents`.
        text_latents_key (str): key in batch dict for text latent.
            Default: `caption_latents`.
        precomputed_latents: whether to use precomputed latents.
            Default: `False`.
        encode_latents_in_fp16 (bool): whether to encode latents in fp16.
            Default: `False`.
        mask_pad_tokens (bool): whether to mask pad tokens in unet cross attention.
            Default: `False`.
        sdxl (bool): Whether or not we're training SDXL. Default: `False`.
    """

    def __init__(self,
                 unet,
                 vae,
                 text_encoder,
                 tokenizer,
                 noise_scheduler,
                 inference_noise_scheduler,
                 loss_fn=F.mse_loss,
                 prediction_type: str = 'epsilon',
                 latent_mean: Optional[Tuple[float]] = None,
                 latent_std: Optional[Tuple[float]] = None,
                 downsample_factor: int = 8,
                 offset_noise: Optional[float] = None,
                 train_metrics: Optional[List] = None,
                 val_metrics: Optional[List] = None,
                 quasirandomness: bool = False,
                 train_seed: int = 42,
                 val_seed: int = 1138,
                 image_key: str = 'image',
                 text_key: str = 'captions',
                 image_latents_key: str = 'image_latents',
                 text_latents_key: str = 'caption_latents',
                 precomputed_latents: bool = False,
                 encode_latents_in_fp16: bool = False,
                 mask_pad_tokens: bool = False,
                 fsdp: bool = False,
                 sdxl: bool = False):
        super().__init__()
        self.unet = unet
        self.vae = vae
        self.noise_scheduler = noise_scheduler
        self.loss_fn = loss_fn
        self.prediction_type = prediction_type.lower()
        if self.prediction_type not in ['sample', 'epsilon', 'v_prediction']:
            raise ValueError(f'prediction type must be one of sample, epsilon, or v_prediction. Got {prediction_type}')
        self.downsample_factor = downsample_factor
        self.offset_noise = offset_noise
        self.quasirandomness = quasirandomness
        self.train_seed = train_seed
        self.val_seed = val_seed
        self.image_key = image_key
        self.image_latents_key = image_latents_key
        self.precomputed_latents = precomputed_latents
        self.mask_pad_tokens = mask_pad_tokens
        self.sdxl = sdxl
        if latent_mean is None:
            self.latent_mean = 4 * (0.0)
        if latent_std is None:
            self.latent_std = 4 * (1 / 0.13025,) if self.sdxl else 4 * (1 / 0.18215,)
        self.latent_mean = torch.tensor(latent_mean).view(1, -1, 1, 1)
        self.latent_std = torch.tensor(latent_std).view(1, -1, 1, 1)
        self.train_metrics = train_metrics if train_metrics is not None else [MeanSquaredError()]
        self.val_metrics = val_metrics if val_metrics is not None else [MeanSquaredError()]
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.inference_scheduler = inference_noise_scheduler
        self.text_key = text_key
        self.text_latents_key = text_latents_key
        self.encode_latents_in_fp16 = encode_latents_in_fp16
        self.mask_pad_tokens = mask_pad_tokens
        # freeze text_encoder during diffusion training
        self.text_encoder.requires_grad_(False)
        self.vae.requires_grad_(False)
        if self.encode_latents_in_fp16:
            self.text_encoder = self.text_encoder.half()
            self.vae = self.vae.half()
        if fsdp:
            # only wrap models we are training
            self.text_encoder._fsdp_wrap = False
            self.vae._fsdp_wrap = False
            self.unet._fsdp_wrap = True

        # Optional rng generator
        self.rng_generator: Optional[torch.Generator] = None
        if self.quasirandomness:
            self.sobol = qmc.Sobol(d=1, scramble=True, seed=self.train_seed)

    def _apply(self, fn):
        super(StableDiffusion, self)._apply(fn)
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
        latents, text_embeds, text_pooled_embeds, attention_mask, encoder_attention_mask = None, None, None, None, None
        if 'attention_mask' in batch:
            attention_mask = batch['attention_mask']  # mask for text encoders
            # text mask for U-Net
            if self.mask_pad_tokens:
                encoder_attention_mask = _create_unet_attention_mask(attention_mask)

        # Use latents if specified and available. When specified, they might not exist during eval
        if self.precomputed_latents and self.image_latents_key in batch and self.text_latents_key in batch:
            if self.sdxl:
                raise NotImplementedError('SDXL not yet supported with precomputed latents')
            latents, text_embeds = batch[self.image_latents_key], batch[self.text_latents_key]
        else:
            inputs, conditionings = batch[self.image_key], batch[self.text_key]

            # If encode_latents_in_fp16, disable autocast context as models are in fp16
            c = torch.cuda.amp.autocast(enabled=False) if self.encode_latents_in_fp16 else nullcontext()  # type: ignore
            with c:
                # Encode the images to the latent space.
                if self.encode_latents_in_fp16:
                    latents = self.vae.encode(inputs.half())['latent_dist'].sample().data
                else:
                    latents = self.vae.encode(inputs)['latent_dist'].sample().data
                # Encode tokenized prompt into embedded text and pooled text embeddings
                text_encoder_out = self.text_encoder(conditionings, attention_mask=attention_mask)
                text_embeds = text_encoder_out[0]
                if self.sdxl:
                    if len(text_encoder_out) <= 1:
                        raise RuntimeError('SDXL requires text encoder output to include a pooled text embedding')
                    text_pooled_embeds = text_encoder_out[1]

        # Scale the latents
        latents = (latents - self.latent_mean) / self.latent_std

        # Zero dropped captions if needed
        if 'drop_caption_mask' in batch.keys():
            text_embeds *= batch['drop_caption_mask'].view(-1, 1, 1)
            if text_pooled_embeds is not None:
                text_pooled_embeds *= batch['drop_caption_mask'].view(-1, 1)

        # Sample the diffusion timesteps
        timesteps = self._generate_timesteps(latents)
        # Add noise to the inputs (forward diffusion)
        noise = torch.randn(*latents.shape, device=latents.device, generator=self.rng_generator)
        if self.offset_noise is not None:
            offset_noise = torch.randn(latents.shape[0],
                                       latents.shape[1],
                                       1,
                                       1,
                                       device=noise.device,
                                       generator=self.rng_generator)
            noise += self.offset_noise * offset_noise
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

        added_cond_kwargs = {}
        # if using SDXL, prepare added time ids & embeddings
        if self.sdxl:
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
        prompt: Optional[list] = None,
        negative_prompt: Optional[list] = None,
        tokenized_prompts: Optional[torch.LongTensor] = None,
        tokenized_negative_prompts: Optional[torch.LongTensor] = None,
        tokenized_prompts_pad_mask: Optional[torch.LongTensor] = None,
        tokenized_negative_prompts_pad_mask: Optional[torch.LongTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: Optional[int] = 50,
        guidance_scale: Optional[float] = 3.0,
        rescaled_guidance: Optional[float] = None,
        num_images_per_prompt: Optional[int] = 1,
        seed: Optional[int] = None,
        progress_bar: Optional[bool] = True,
        zero_out_negative_prompt: bool = True,
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
            zero_out_negative_prompt (bool): Whether or not to zero out negative prompt if it is
                an empty string. Default: `True`.
            crop_params (torch.FloatTensor of size [Bx2], optional): Crop parameters to use
                when generating images with SDXL. Default: `None`.
            input_size_params (torch.FloatTensor of size [Bx2], optional): Size parameters
                (representing original size of input image) to use when generating images with SDXL.
                Default: `None`.
        """
        _check_prompt_given(prompt, tokenized_prompts, prompt_embeds)
        _check_prompt_lenths(prompt, negative_prompt)
        _check_prompt_lenths(tokenized_prompts, tokenized_negative_prompts)
        _check_prompt_lenths(prompt_embeds, negative_prompt_embeds)

        # Create rng for the generation
        device = self.vae.device
        rng_generator = torch.Generator(device=device)
        if seed:
            rng_generator = rng_generator.manual_seed(seed)  # type: ignore

        height = height or self.unet.config.sample_size * self.downsample_factor
        width = width or self.unet.config.sample_size * self.downsample_factor
        assert height is not None  # for type checking
        assert width is not None  # for type checking

        do_classifier_free_guidance = guidance_scale > 1.0  # type: ignore

        text_embeddings, pooled_text_embeddings, pad_attn_mask = self._prepare_text_embeddings(
            prompt, tokenized_prompts, tokenized_prompts_pad_mask, prompt_embeds, num_images_per_prompt)
        batch_size = len(text_embeddings)  # len prompts * num_images_per_prompt
        # classifier free guidance + negative prompts
        # negative prompt is given in place of the unconditional input in classifier free guidance
        pooled_embeddings, encoder_attn_mask = pooled_text_embeddings, pad_attn_mask
        if do_classifier_free_guidance:
            if not negative_prompt and not tokenized_negative_prompts and not negative_prompt_embeds and zero_out_negative_prompt:
                # Negative prompt is empty and we want to zero it out
                unconditional_embeddings = torch.zeros_like(text_embeddings)
                pooled_unconditional_embeddings = torch.zeros_like(pooled_text_embeddings) if self.sdxl else None
                uncond_pad_attn_mask = torch.zeros_like(pad_attn_mask) if pad_attn_mask is not None else None
            else:
                if not negative_prompt:
                    negative_prompt = [''] * (batch_size // num_images_per_prompt)  # type: ignore
                unconditional_embeddings, pooled_unconditional_embeddings, uncond_pad_attn_mask = self._prepare_text_embeddings(
                    negative_prompt, tokenized_negative_prompts, tokenized_negative_prompts_pad_mask,
                    negative_prompt_embeds, num_images_per_prompt)

            # concat uncond + prompt
            text_embeddings = torch.cat([unconditional_embeddings, text_embeddings])
            if self.sdxl:
                pooled_embeddings = torch.cat([pooled_unconditional_embeddings, pooled_text_embeddings])  # type: ignore
            if pad_attn_mask is not None:
                encoder_attn_mask = torch.cat([uncond_pad_attn_mask, pad_attn_mask])  # type: ignore
        else:
            if self.sdxl:
                pooled_embeddings = pooled_text_embeddings

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
        if self.sdxl and pooled_embeddings is not None:
            if crop_params is None:
                crop_params = torch.zeros((batch_size, 2), dtype=text_embeddings.dtype)
            if input_size_params is None:
                input_size_params = torch.tensor([width, height], dtype=text_embeddings.dtype).repeat(batch_size, 1)
            output_size_params = torch.tensor([width, height], dtype=text_embeddings.dtype).repeat(batch_size, 1)

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

    def _prepare_text_embeddings(self, prompt, tokenized_prompts, tokenized_pad_mask, prompt_embeds,
                                 num_images_per_prompt):
        """Tokenizes and embeds prompts if needed, then duplicates embeddings to support multiple generations per prompt."""
        device = self.text_encoder.device
        pooled_text_embeddings = None
        if prompt_embeds is None:
            if tokenized_prompts is None:
                tokenized_out = self.tokenizer(prompt,
                                               padding='max_length',
                                               max_length=self.tokenizer.model_max_length,
                                               truncation=True,
                                               return_tensors='pt')
                tokenized_prompts = tokenized_out['input_ids']
                if self.mask_pad_tokens:
                    tokenized_pad_mask = tokenized_out['attention_mask']
                else:
                    tokenized_pad_mask = None
            if tokenized_pad_mask is not None:
                tokenized_pad_mask = tokenized_pad_mask.to(device)
            text_encoder_out = self.text_encoder(tokenized_prompts.to(device), attention_mask=tokenized_pad_mask)
            prompt_embeds = text_encoder_out[0]
            if self.sdxl:
                if len(text_encoder_out) <= 1:
                    raise RuntimeError('SDXL requires text encoder output to include a pooled text embedding')
                pooled_text_embeddings = text_encoder_out[1]
        else:
            if self.sdxl:
                raise NotImplementedError('SDXL not yet supported with precomputed embeddings')

        # duplicate text embeddings for each generation per prompt
        prompt_embeds = _duplicate_tensor(prompt_embeds, num_images_per_prompt)

        if not self.mask_pad_tokens:
            tokenized_pad_mask = None

        if tokenized_pad_mask is not None:
            tokenized_pad_mask = _create_unet_attention_mask(tokenized_pad_mask)
            tokenized_pad_mask = _duplicate_tensor(tokenized_pad_mask, num_images_per_prompt)

        if self.sdxl and pooled_text_embeddings is not None:
            pooled_text_embeddings = _duplicate_tensor(pooled_text_embeddings, num_images_per_prompt)
        return prompt_embeds, pooled_text_embeddings, tokenized_pad_mask


def _check_prompt_lenths(prompt, negative_prompt):
    if prompt is None and negative_prompt is None:
        return
    batch_size = 1 if isinstance(prompt, str) else len(prompt)
    if negative_prompt:
        negative_prompt_bs = 1 if isinstance(negative_prompt, str) else len(negative_prompt)
        if negative_prompt_bs != batch_size:
            raise ValueError('len(prompts) and len(negative_prompts) must be the same. \
                    A negative prompt must be provided for each given prompt.')


def _check_prompt_given(prompt, tokenized_prompts, prompt_embeds):
    if prompt is None and tokenized_prompts is None and prompt_embeds is None:
        raise ValueError('Must provide one of `prompt`, `tokenized_prompts`, or `prompt_embeds`')


def _create_unet_attention_mask(attention_mask):
    """Takes the union of multiple attention masks if given more than one mask."""
    if len(attention_mask.shape) == 2:
        return attention_mask
    elif len(attention_mask.shape) == 3:
        encoder_attention_mask = attention_mask[:, 0]
        for i in range(1, attention_mask.shape[1]):
            encoder_attention_mask |= attention_mask[:, i]
        return encoder_attention_mask
    else:
        raise ValueError(f'attention_mask should have either 2 or 3 dimensions: {attention_mask.shape}')


def _duplicate_tensor(tensor, num_images_per_prompt):
    """Duplicate tensor for multiple generations from a single prompt."""
    batch_size, seq_len = tensor.shape[:2]
    tensor = tensor.repeat(1, num_images_per_prompt, *[
        1,
    ] * len(tensor.shape[2:]))
    return tensor.view(batch_size * num_images_per_prompt, seq_len, *[
        -1,
    ] * len(tensor.shape[2:]))
