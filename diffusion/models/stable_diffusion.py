# Copyright 2022 MosaicML Diffusion authors
# SPDX-License-Identifier: Apache-2.0

"""Diffusion models."""

from typing import List, Optional

import torch
import torch.nn.functional as F
from composer.models import ComposerModel
from torchmetrics import MeanSquaredError, Metric
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
        train_metrics (list): List of torchmetrics to calculate during training.
            Default: `None`.
        val_metrics (list): List of torchmetrics to calculate during validation.
            Default: `None`.
        val_seed (int): Seed to use for generating eval images. Default: `1138`.
        image_key (str): The name of the image inputs in the dataloader batch.
            Default: `image_tensor`.
        caption_key (str): The name of the caption inputs in the dataloader batch.
            Default: `input_ids`.
        val_guidance_scales (list): list of guidance scales to use during evaluation.
            Default: `[0.0]`.
        loss_bins (list): list of tuples of (min, max) values to bin the loss into.
            For example, [(0,0.5), (0.5, 1)] Will track the loss separately for the
            first and last halves of the diffusion process. Default:`[(0,1)]`.
        fsdp (bool): whether to use FSDP, Default: `False`.
        image_latents_key (str): key in batch dict for image latents.
            Default: `image_latents`.
        text_latents_key (str): key in batch dict for text latent.
            Default: `caption_latents`.
        precomputed_latents: whether to use precomputed latents.
            Default: `False`.
        encode_latents_in_fp16 (bool): whether to encode latents in fp16.
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
                 train_metrics: Optional[List] = None,
                 val_metrics: Optional[List] = None,
                 val_seed: int = 1138,
                 val_guidance_scales: Optional[List] = None,
                 loss_bins: Optional[List] = None,
                 image_key: str = 'image',
                 text_key: str = 'captions',
                 image_latents_key: str = 'image_latents',
                 text_latents_key: str = 'caption_latents',
                 precomputed_latents: bool = False,
                 encode_latents_in_fp16: bool = False,
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
        self.val_seed = val_seed
        self.image_key = image_key
        self.image_latents_key = image_latents_key
        self.precomputed_latents = precomputed_latents
        self.sdxl = sdxl
        if self.sdxl:
            self.latent_scale = 0.13025
        else:
            self.latent_scale = 0.18215

        # setup metrics
        if train_metrics is None:
            self.train_metrics = [MeanSquaredError()]
        else:
            self.train_metrics = train_metrics
        if val_metrics is None:
            val_metrics = [MeanSquaredError()]
        if val_guidance_scales is None:
            val_guidance_scales = [0.0]
        if loss_bins is None:
            loss_bins = [(0, 1)]
        # Create new val metrics for each guidance weight and each loss bin
        self.val_guidance_scales = val_guidance_scales

        # bin metrics
        self.val_metrics = {}
        metrics_to_sweep = ['FrechetInceptionDistance', 'InceptionScore', 'CLIPScore']
        for metric in val_metrics:
            if metric.__class__.__name__ in metrics_to_sweep:
                for scale in val_guidance_scales:
                    new_metric = type(metric)(**vars(metric))
                    # WARNING: ugly hack...
                    new_metric.guidance_scale = scale
                    scale_str = str(scale).replace('.', 'p')
                    self.val_metrics[f'{metric.__class__.__name__}-scale-{scale_str}'] = new_metric
            elif isinstance(metric, MeanSquaredError):
                for bin in loss_bins:
                    new_metric = type(metric)(**vars(metric))
                    # WARNING: ugly hack...
                    new_metric.loss_bin = bin
                    self.val_metrics[f'{metric.__class__.__name__}-bin-{bin[0]}-to-{bin[1]}'.replace('.',
                                                                                                     'p')] = new_metric
            else:
                self.val_metrics[metric.__class__.__name__] = metric
        # Add a mse metric for the full loss
        self.val_metrics['MeanSquaredError'] = MeanSquaredError()

        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.inference_scheduler = inference_noise_scheduler
        self.text_key = text_key
        self.text_latents_key = text_latents_key
        self.encode_latents_in_fp16 = encode_latents_in_fp16
        # freeze text_encoder during diffusion training
        self.text_encoder.requires_grad_(False)
        self.vae.requires_grad_(False)
        if self.encode_latents_in_fp16:
            self.text_encoder.half()
            self.vae.half()
        if fsdp:
            # only wrap models we are training
            self.text_encoder._fsdp_wrap = False
            self.vae._fsdp_wrap = False
            self.unet._fsdp_wrap = True

    def forward(self, batch):
        latents, conditioning, pooled_conditioning = None, None, None
        # Use latents if specified and available. When specified, they might not exist during eval
        if self.precomputed_latents and self.image_latents_key in batch and self.text_latents_key in batch:
            if self.sdxl:
                raise NotImplementedError('SDXL not yet supported with precomputed latents')
            latents, conditioning = batch[self.image_latents_key], batch[self.text_latents_key]
        else:
            inputs, conditioning = batch[self.image_key], batch[self.text_key]
            if self.sdxl:
                conditioning, conditioning_2 = conditioning[:,0,:], conditioning[:,1,:] # [B, 2, 77]
            else:
                conditioning_2 = None
            conditioning = conditioning.view(-1, conditioning.shape[-1])
            if self.encode_latents_in_fp16:
                # Disable autocast context as models are in fp16
                with torch.cuda.amp.autocast(enabled=False):
                    # Encode the images to the latent space.
                    # Encode prompt into conditioning vector
                    latents = self.vae.encode(inputs.half())['latent_dist'].sample().data
                    # if self.sdxl:
                    #     conditioning_2 = batch[self.text_key_2].view(-1, conditioning_2.shape[-1])
                    #     conditioning, pooled_conditioning = self.text_encoder(conditioning, conditioning_2)
                    # else:
                    #     conditioning = self.text_encoder(conditioning)[0]  # Should be (batch_size, 77, 768)
                    #     pooled_conditioning = None
            else:
                latents = self.vae.encode(inputs)['latent_dist'].sample().data

            if self.sdxl:
                assert conditioning_2 is not None
                conditioning_2 = conditioning_2.view(-1, conditioning_2.shape[-1])
                conditioning, pooled_conditioning = self.text_encoder([conditioning, conditioning_2])
            else:
                conditioning = self.text_encoder(conditioning)[0]

            # Magical scaling number (See https://github.com/huggingface/diffusers/issues/437#issuecomment-1241827515)
            latents *= self.latent_scale

        # Sample the diffusion timesteps
        timesteps = torch.randint(0, len(self.noise_scheduler), (latents.shape[0],), device=latents.device)
        # Add noise to the inputs (forward diffusion)
        noise = torch.randn_like(latents)
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
            add_text_embeds = pooled_conditioning
            added_cond_kwargs = {'text_embeds': add_text_embeds, 'time_ids': add_time_ids}

        # Forward through the model
        return self.unet(noised_latents, timesteps, conditioning,
                         added_cond_kwargs=added_cond_kwargs)['sample'], targets, timesteps

    def loss(self, outputs, batch):
        """Loss between unet output and added noise, typically mse."""
        return self.loss_fn(outputs[0], outputs[1])

    def eval_forward(self, batch, outputs=None):
        """For stable diffusion, eval forward computes unet outputs as well as some samples."""
        # Skip this if outputs have already been computed, e.g. during training
        if outputs is not None:
            return outputs
        # Get unet outputs
        unet_out, targets, timesteps = self.forward(batch)
        # Sample images from the prompts in the batch
        prompts = batch[self.text_key]
        height, width = batch[self.image_key].shape[-2], batch[self.image_key].shape[-1]

        # If SDXL, add eval-time micro-conditioning to batch
        if self.sdxl:
            device = self.unet.device
            bsz = batch[self.image_key].shape[0]
            # Set to resolution we are trying to generate
            batch['cond_original_size'] = torch.tensor([[width, height]]).repeat(bsz, 1).to(device)
            # No cropping
            batch['cond_crops_coords_top_left'] = torch.tensor([[0., 0.]]).repeat(bsz, 1).to(device)
            # Set to resolution we are trying to generate
            batch['cond_target_size'] = torch.tensor([[width, height]]).repeat(bsz, 1).to(device)

        generated_images = {}
        for guidance_scale in self.val_guidance_scales:
            gen_images = self.generate(tokenized_prompts=prompts,
                                       height=height,
                                       width=width,
                                       guidance_scale=guidance_scale,
                                       seed=self.val_seed,
                                       progress_bar=False)
            generated_images[guidance_scale] = gen_images
        return unet_out, targets, timesteps, generated_images

    def get_metrics(self, is_train: bool = False):
        if is_train:
            metrics = self.train_metrics
        else:
            metrics = self.val_metrics

        if isinstance(metrics, Metric):
            metrics_dict = {metrics.__class__.__name__: metrics}
        elif isinstance(metrics, list):
            metrics_dict = {metrics.__class__.__name__: metric for metric in metrics}
        else:
            metrics_dict = {}
            for name, metric in metrics.items():
                assert isinstance(metric, Metric)
                metrics_dict[name] = metric

        return metrics_dict

    def update_metric(self, batch, outputs, metric):
        # If A MSE metric is associated with a loss bin, update the metric for the bin
        # Othewise, update the metric for the full loss
        if isinstance(metric, MeanSquaredError) and hasattr(metric, 'loss_bin'):
            # Get the loss bin from the metric
            loss_bin = metric.loss_bin
            # Get the loss for timesteps in the bin
            T_max = self.noise_scheduler.num_train_timesteps
            # Get the indices corresponding to timesteps in the bin
            bin_indices = torch.where(
                (outputs[2] >= loss_bin[0] * T_max) & (outputs[2] < loss_bin[1] * T_max))  # type: ignore
            # Update the metric for items in the bin
            metric.update(outputs[0][bin_indices], outputs[1][bin_indices])
        elif isinstance(metric, MeanSquaredError):
            metric.update(outputs[0], outputs[1])
        # FID metrics should be updated with the generated images at the desired guidance scale
        elif metric.__class__.__name__ == 'FrechetInceptionDistance':
            metric.update(batch[self.image_key], real=True)
            metric.update(outputs[3][metric.guidance_scale], real=False)
        # IS metrics should be updated with the generated images at the desired guidance scale
        elif metric.__class__.__name__ == 'InceptionScore':
            metric.update(outputs[3][metric.guidance_scale])
        # CLIP metrics should be updated with the generated images at the desired guidance scale
        elif metric.__class__.__name__ == 'CLIPScore':
            # Convert the captions to a list of strings
            if self.sdxl:
                # Decode captions with first tokenizer
                captions = [
                    self.tokenizer.tokenizer.decode(caption[0], skip_special_tokens=True)
                    for caption in batch[self.text_key]
                ]
            else:
                captions = [
                    self.tokenizer.decode(caption, skip_special_tokens=True) for caption in batch[self.text_key]
                ]
            generated_images = (outputs[3][metric.guidance_scale] * 255).to(torch.uint8)
            metric.update(generated_images, captions)
        else:
            metric.update(outputs[0], outputs[1])

    @torch.no_grad()
    def generate(
        self,
        prompt: Optional[list] = None,
        negative_prompt: Optional[list] = None,
        tokenized_prompts: Optional[torch.LongTensor] = None,
        tokenized_negative_prompts: Optional[torch.LongTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: Optional[int] = 50,
        guidance_scale: Optional[float] = 3.0,
        num_images_per_prompt: Optional[int] = 1,
        seed: Optional[int] = None,
        progress_bar: Optional[bool] = True,
        zero_out_negative_prompt: bool = True,
        crop_params: Optional[list] = None,
        size_params: Optional[list] = None,
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
            tokenized_prompts (torch.LongTensor or List[torch.LongTensor]): Optionally pass
                pre-tokenized prompts instead of string prompts. If SDXL, this will be a list
                of two pre-tokenized prompts. Default: `None`.
            tokenized_negative_prompts (torch.LongTensor): Optionally pass pre-tokenized negative
                prompts instead of string prompts. Default: `None`.
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
            num_images_per_prompt (int): The number of images to generate per prompt.
                 Default: `1`.
            progress_bar (bool): Whether to use the tqdm progress bar during generation.
                Default: `True`.
            seed (int): Random seed to use for generation. Set a seed for reproducible generation.
                Default: `None`.
            zero_out_negative_prompt (bool): Whether or not to zero out negative prompt if it is
                an empty string. Default: `True`.
            crop_params (list, optional): Crop parameters to use when generating images with SDXL.
                Default: `None`.
            size_params (list, optional): Size parameters to use when generating images with SDXL.
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

        vae_scale_factor = 8
        height = height or self.unet.config.sample_size * vae_scale_factor
        width = width or self.unet.config.sample_size * vae_scale_factor
        assert height is not None  # for type checking
        assert width is not None  # for type checking

        do_classifier_free_guidance = guidance_scale > 1.0  # type: ignore

        text_embeddings, pooled_text_embeddings = self._prepare_text_embeddings(prompt, tokenized_prompts,
                                                                                prompt_embeds, num_images_per_prompt)
        batch_size = len(text_embeddings)  # len prompts * num_images_per_prompt
        # classifier free guidance + negative prompts
        # negative prompt is given in place of the unconditional input in classifier free guidance
        pooled_embeddings = None
        if do_classifier_free_guidance:
            if negative_prompt_embeds is None and zero_out_negative_prompt:
                unconditional_embeddings = torch.zeros_like(text_embeddings)
                if pooled_text_embeddings is not None:
                    pooled_unconditional_embeddings = torch.zeros_like(pooled_text_embeddings)
                else:
                    pooled_unconditional_embeddings = None
            else:
                negative_prompt = negative_prompt or ([''] * (batch_size // num_images_per_prompt))  # type: ignore
                unconditional_embeddings, pooled_unconditional_embeddings = self._prepare_text_embeddings(
                    negative_prompt, tokenized_negative_prompts, negative_prompt_embeds, num_images_per_prompt)

            # concat uncond + prompt
            text_embeddings = torch.cat([unconditional_embeddings, text_embeddings])
            if self.sdxl:
                pooled_embeddings = torch.cat([pooled_unconditional_embeddings, pooled_text_embeddings])  # type: ignore

        # prepare for diffusion generation process
        latents = torch.randn(
            (batch_size, self.unet.config.in_channels, height // vae_scale_factor, width // vae_scale_factor),
            device=device,
            generator=rng_generator,
        )

        self.inference_scheduler.set_timesteps(num_inference_steps)
        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.inference_scheduler.init_noise_sigma

        added_cond_kwargs = {}
        # if using SDXL, prepare added time ids & embeddings
        if self.sdxl and pooled_embeddings is not None:
            if not crop_params:
                crop_params = [0., 0.]
            if not size_params:
                size_params = [width, height]
            cond_original_size = torch.tensor([[width, height]]).repeat(pooled_embeddings.shape[0],
                                                                        1).to(device).float()
            cond_crops_coords_top_left = torch.tensor([crop_params]).repeat(pooled_embeddings.shape[0],
                                                                            1).to(device).float()
            cond_target_size = torch.tensor([size_params]).repeat(pooled_embeddings.shape[0], 1).to(device).float()
            add_time_ids = torch.cat([cond_original_size, cond_crops_coords_top_left, cond_target_size], dim=1).float()
            add_text_embeds = pooled_embeddings

            added_cond_kwargs = {'text_embeds': add_text_embeds, 'time_ids': add_time_ids}

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
                             added_cond_kwargs=added_cond_kwargs).sample

            if do_classifier_free_guidance:
                # perform guidance. Note this is only techincally correct for prediction_type 'epsilon'
                pred_uncond, pred_text = pred.chunk(2)
                pred = pred_uncond + guidance_scale * (pred_text - pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.inference_scheduler.step(pred, t, latents, generator=rng_generator).prev_sample

        # We now use the vae to decode the generated latents back into the image.
        # scale and decode the image latents with vae
        latents = 1 / self.latent_scale * latents
        image = self.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        return image.detach()  # (batch*num_images_per_prompt, channel, h, w)

    def _prepare_text_embeddings(self, prompt, tokenized_prompts, prompt_embeds, num_images_per_prompt):
        """Tokenizes and embeds prompts if needed, then duplicates embeddings to support multiple generations per prompt."""
        device = self.text_encoder.device
        pooled_text_embeddings = None
        if prompt_embeds is None:
            if self.sdxl:
                if tokenized_prompts is None:
                    tokenized_prompts = self.tokenizer(prompt,
                                                       padding='max_length',
                                                       truncation=True,
                                                       return_tensors='pt',
                                                       input_ids=True)
                # TODO implement zero-ing out empty prompts!
                text_embeddings, pooled_text_embeddings = self.text_encoder(
                    [tokenized_prompts[0].to(device), tokenized_prompts[1].to(device)])  # type: ignore
            else:
                if tokenized_prompts is None:
                    tokenized_prompts = self.tokenizer(prompt,
                                                       padding='max_length',
                                                       max_length=self.tokenizer.model_max_length,
                                                       truncation=True,
                                                       return_tensors='pt').input_ids
                text_embeddings = self.text_encoder(tokenized_prompts.to(device))[0]  # type: ignore
        else:
            if self.sdxl:
                raise NotImplementedError('SDXL not yet supported with precomputed embeddings')
            text_embeddings = prompt_embeds

        # duplicate text embeddings for each generation per prompt
        bs_embed, seq_len, _ = text_embeddings.shape
        text_embeddings = text_embeddings.repeat(1, num_images_per_prompt, 1)  # type: ignore
        text_embeddings = text_embeddings.view(bs_embed * num_images_per_prompt, seq_len, -1)

        if self.sdxl and pooled_text_embeddings is not None:
            pooled_text_embeddings = pooled_text_embeddings.repeat(1, num_images_per_prompt)
            pooled_text_embeddings = pooled_text_embeddings.view(bs_embed * num_images_per_prompt, -1)
        return text_embeddings, pooled_text_embeddings


def _check_prompt_lenths(prompt, negative_prompt):
    if prompt is None and negative_prompt is None:
        return
    batch_size = 1 if isinstance(prompt, str) else len(prompt)
    if negative_prompt:
        negative_prompt_bs = 1 if isinstance(negative_prompt, str) else len(negative_prompt)
        if negative_prompt_bs != batch_size:
            raise ValueError(f'len(prompts) and len(negative_prompts) must be the same. \
                    A negative prompt must be provided for each given prompt.')


def _check_prompt_given(prompt, tokenized_prompts, prompt_embeds):
    if prompt is None and tokenized_prompts is None and prompt_embeds is None:
        raise ValueError('Must provide one of `prompt`, `tokenized_prompts`, or `prompt_embeds`')
