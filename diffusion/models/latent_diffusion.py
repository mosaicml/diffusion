# Copyright 2022 MosaicML Diffusion authors
# SPDX-License-Identifier: Apache-2.0

"""Custom latent diffusion model."""

from typing import List, Optional, Union

import torch
import torch.nn.functional as F
from composer.models import ComposerModel
from torch.quasirandom import SobolEngine
from torchmetrics import MeanSquaredError
from tqdm.auto import tqdm

from diffusion.models.autoencoder import AutoEncoder


class LatentDiffusion(ComposerModel):
    """Latent Diffusion ComposerModel.

    This is a Latent Diffusion model for generating images conditioned on text prompts.

    Args:
        model (torch.nn.Module): Core diffusion model. must accept a
            (B, C, H, W) input, (B,) timestep array of noise timesteps,
            and (B, S, D) text conditioning vectors.
        autoencoder (torch.nn.Module): Autoencoder for encoding and decoding latents.
            must support `.encode()` and `decode()` functions.
        text_encoder (torch.nn.Module): HuggingFace CLIP or LLM text enoder.
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer used for
            text_encoder. For a `CLIPTextModel` this will be the
            `CLIPTokenizer` from HuggingFace transformers.
        prediction_type (str): The type of prediction to use. Currently only `epsilon` and `v_prediction` are supported.
        offset_noise (float, optional): The scale of the offset noise. If not specified, offset noise will not
            be used. Default `None`.
        latent_means (tuple[float], optional): The means of the latents. If not specified, the means will be
            set to zero. Default `None`.
        latent_stds (tuple[float], optional): The standard deviations of the latents. If not specified, the
            standard deviations will be set to one. Default `None`.
        T_max (int): The maximum time for the forward diffusion process. Default: `1000`.
        image_key (str): The name of the image inputs in the dataloader batch.
            Default: `image_tensor`.
        caption_key (str): The name of the caption inputs in the dataloader batch.
            Default: `input_ids`.
        fsdp (bool): whether to use FSDP, Default: `False`.
        encode_latents_in_fp16 (bool): whether to encode latents in fp16.
            Default: `False`.
        use_quasirandom_timesteps (bool): whether to use quasirandom timesteps. Default: `False`.
        fourier_feature_transform (nn.Module, optional): Fourier feature transformation layer. Default: `None`.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        autoencoder: AutoEncoder,
        text_encoder,
        tokenizer,
        prediction_type: str = 'epsilon',
        offset_noise: Optional[float] = None,
        latent_means: Optional[tuple[float]] = None,
        latent_stds: Optional[tuple[float]] = None,
        T_max: int = 1000,
        image_key: str = 'image',
        text_key: str = 'captions',
        attention_mask_key: str = 'attention_mask',
        fsdp: bool = False,
        encode_latents_in_fp16: bool = False,
        use_quasirandom_timesteps: bool = False,
        fourier_feature_transform: Optional[torch.nn.Module] = None,
    ):
        super().__init__()
        self.model = model
        self.autoencoder = autoencoder
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.prediction_type = prediction_type.lower()
        if self.prediction_type not in ['epsilon', 'v_prediction']:
            raise ValueError(f'Unrecognized prediction type {self.prediction_type}')
        self.offset_noise = offset_noise
        self.latent_means = latent_means
        self.latent_stds = latent_stds
        self.T_max = T_max
        self.image_key = image_key
        self.text_key = text_key
        self.attention_mask_key = attention_mask_key
        self.fsdp = fsdp
        self.encode_latents_in_fp16 = encode_latents_in_fp16
        self.use_quasirandom_timesteps = use_quasirandom_timesteps
        self.fourier_feature_transform = fourier_feature_transform

        # Set up randomness
        if self.use_quasirandom_timesteps:
            self.sobol_engine = SobolEngine(dimension=1, scramble=True)

        # Set up metrics
        self.train_metrics = [MeanSquaredError()]
        self.val_metrics = [MeanSquaredError()]

        # Optional additional conditioning keys
        self.additional_cond_keys = ['cond_original_size', 'cond_crops_coords_top_left', 'cond_target_size']

        # freeze autoencoder and text_encoder during diffusion training
        self.text_encoder.requires_grad_(False)
        self.autoencoder.requires_grad_(False)
        if self.encode_latents_in_fp16:
            self.text_encoder = self.text_encoder.half()
            self.autoencoder = self.autoencoder.half()
        if fsdp:
            # only wrap models we are training
            self.text_encoder._fsdp_wrap = False
            self.autoencoder._fsdp_wrap = False
            self.model._fsdp_wrap = True

    def embed_images(self, images: torch.Tensor):
        """Get the image latents from the autoencoder."""
        if self.encode_latents_in_fp16:
            # Disable autocast context as models are in fp16
            with torch.cuda.amp.autocast(enabled=False):
                latents = self.autoencoder.encode(images.half()).latent_dist.sample().data
        else:
            latents = self.autoencoder.encode(images).latent_dist.sample().data
        return latents

    def tokenize_text(self, text: Union[str, List[str]]):
        """Tokenize text using the tokenizer."""
        if isinstance(text, str):
            text = [text]
        tokenizer_out = self.tokenizer(text, padding='max_length', truncation=True, return_tensors='pt')
        return tokenizer_out.input_ids, tokenizer_out.attention_mask

    def embed_text(self, tokenized_text, attention_mask):
        """Get the text embeddings from the text encoder(s).

        Tokenized text is expected to be of shape (batch_size, sequence_length) for a single tokenizer,
        or (batch_size, num_tokenizers, sequence_length) for multiple tokenizers.
        """
        # Ensure that the tokenized text is on the same device as the text encoder
        tokenized_text = tokenized_text.to(self.text_encoder.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.text_encoder.device)
        # There are possibly multiple text tokenizations depending on the model. Get them all.
        if tokenized_text.ndim > 2:
            conditionings = [tokenized_text[:, i, :] for i in range(tokenized_text.shape[1])]
            conditionings = [c.view(-1, c.shape[-1]) for c in conditionings]
        elif tokenized_text.ndim <= 2:
            conditionings = [tokenized_text]
        else:
            raise ValueError('Invalid tokenized text shape.')
        # Embed the text. In the event there are multiple tokenizations the text encoder expects a list.
        if len(conditionings) > 1:
            conditioning, pooled_conditioning = self.text_encoder(conditionings, attention_mask=attention_mask)
        elif len(conditionings) == 1:
            attention_mask = attention_mask.squeeze()
            conditioning = self.text_encoder(conditionings[0], attention_mask=attention_mask)[0]
            pooled_conditioning = None
        else:
            raise ValueError('No conditioning vectors were found.')
        return conditioning, pooled_conditioning

    def init_latent_stats(self, latents: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Init the means
        if self.latent_means is None:
            # Default to zero mean latents
            latent_means = torch.zeros(latents.shape[1], device=latents.device)
        elif isinstance(self.latent_means, (tuple, list)):
            latent_means = torch.tensor(self.latent_means, device=latents.device)
        elif isinstance(self.latent_means, torch.Tensor):
            latent_means = self.latent_means
        else:
            raise ValueError(f'Unrecognized latent means type {type(self.latent_means)}')
        # Init the standard deviations
        if self.latent_stds is None:
            # Default to unit variance latents
            latent_stds = torch.ones(latents.shape[1], device=latents.device)
        elif isinstance(self.latent_stds, (tuple, list)):
            latent_stds = torch.tensor(self.latent_stds, device=latents.device)
        elif isinstance(self.latent_stds, torch.Tensor):
            latent_stds = self.latent_stds
        else:
            raise ValueError(f'Unrecognized latent stds type {type(self.latent_stds)}')
        return latent_means, latent_stds

    def scale_latents(self, latents):
        """Scale the latents by their means and standard deviations."""
        self.latent_means, self.latent_stds = self.init_latent_stats(latents)
        latents = (latents - self.latent_means.view(1, -1, 1, 1)) / self.latent_stds.view(1, -1, 1, 1)
        return latents

    def unscale_latents(self, latents):
        """Unscale the latents by their means and standard deviations."""
        self.latent_means, self.latent_stds = self.init_latent_stats(latents)
        latents = latents * self.latent_stds.view(1, -1, 1, 1) + self.latent_means.view(1, -1, 1, 1)
        return latents

    def diffusion_forward_process(self, latents: torch.Tensor):
        """Diffusion forward process."""
        # Sample (continuous) timesteps
        if self.use_quasirandom_timesteps:
            timesteps = self.T_max * self.sobol_engine.draw(latents.shape[0]).squeeze().to(latents.device)
        else:
            timesteps = self.T_max * torch.rand(latents.shape[0], device=latents.device)
        # Generate the noise, optionally shifting it with additional offset noise
        noise = torch.randn_like(latents)
        if self.offset_noise is not None:
            offset_noise = torch.randn(latents.shape[0], latents.shape[1], 1, 1, device=noise.device)
            noise += self.offset_noise * offset_noise
        # Add the noise to the latents according to the natural schedule
        cos_t = torch.cos(timesteps * torch.pi / (2 * self.T_max)).view(-1, 1, 1, 1)
        sin_t = torch.sin(timesteps * torch.pi / (2 * self.T_max)).view(-1, 1, 1, 1)
        noised_latents = cos_t * latents + sin_t * noise
        if self.prediction_type == 'epsilon':
            # Get the (epsilon) targets
            targets = noise
        elif self.prediction_type == 'v_prediction':
            # Get the (velocity) targets
            targets = -sin_t * latents + cos_t * noise
        else:
            raise ValueError(f'Unrecognized prediction type {self.prediction_type}')
        # TODO: Implement other prediction types
        return noised_latents, targets, timesteps

    def forward(self, batch):
        images, tokenized_text = batch[self.image_key], batch[self.text_key]
        # Attention mask if used
        if self.attention_mask_key in batch.keys():
            attention_mask = batch[self.attention_mask_key]
        else:
            attention_mask = None

        # Prep the image latents
        latents = self.embed_images(images)
        # Scale the latents by their means and standard deviations
        latents = self.scale_latents(latents)

        # Prep the text embeddings
        conditioning, pooled_conditioning = self.embed_text(tokenized_text, attention_mask)
        # Zero dropped captions if needed
        if 'drop_caption_mask' in batch:
            conditioning *= batch['drop_caption_mask'].view(-1, 1, 1)
            if pooled_conditioning is not None:
                pooled_conditioning *= batch['drop_caption_mask'].view(-1, 1)

        # Diffusion forward process
        noised_latents, targets, timesteps = self.diffusion_forward_process(latents)

        # Optionally add fourier features
        if self.fourier_feature_transform is not None:
            noised_latents = self.fourier_feature_transform(noised_latents)

        # Optional additional conditioning
        added_cond_kwargs = {}
        if all(k in batch for k in self.additional_cond_keys):
            add_time_ids = torch.cat(
                [batch['cond_original_size'], batch['cond_crops_coords_top_left'], batch['cond_target_size']], dim=1)
            add_text_embeds = pooled_conditioning
            added_cond_kwargs = {'text_embeds': add_text_embeds, 'time_ids': add_time_ids}

        # Forward through the model
        model_out = self.model(noised_latents,
                               timesteps,
                               conditioning,
                               encoder_attention_mask=attention_mask,
                               added_cond_kwargs=added_cond_kwargs)['sample']
        return {'predictions': model_out, 'targets': targets}

    def loss(self, outputs, batch):
        """MSE loss between outputs and targets."""
        return F.mse_loss(outputs['predictions'], outputs['targets'])

    def eval_forward(self, batch, outputs=None):
        # Skip this if outputs have already been computed, e.g. during training
        if outputs is not None:
            return outputs
        return self.forward(batch)

    def get_metrics(self, is_train: bool = False):
        if is_train:
            metrics_dict = {metric.__class__.__name__: metric for metric in self.train_metrics}
        else:
            metrics_dict = {metric.__class__.__name__: metric for metric in self.val_metrics}
        return metrics_dict

    def update_metric(self, batch, outputs, metric):
        if isinstance(metric, MeanSquaredError):
            metric.update(outputs['predictions'], outputs['targets'])
        else:
            raise ValueError(f'Unrecognized metric {metric.__class__.__name__}')

    def update_latents(self, latents, predictions, t, delta_t):
        """Gets the latent update."""
        if self.prediction_type == 'epsilon':
            angle = t * torch.pi / (2 * self.T_max)
            cos_t = torch.cos(angle).view(-1, 1, 1, 1)
            sin_t = torch.sin(angle).view(-1, 1, 1, 1)
            if angle == torch.pi / 2:
                # Optimal update here is to do nothing.
                pass
            elif torch.abs(torch.pi / 2 - angle) < 1e-4:
                # Need to avoid instability near t = T_max
                latents = latents - (predictions - sin_t * latents)
            else:
                latents = latents - (predictions - sin_t * latents) * delta_t / cos_t
        elif self.prediction_type == 'v_prediction':
            latents = latents - delta_t * predictions
        return latents

    @torch.no_grad()
    def generate(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        height: int = 64,
        width: int = 64,
        num_inference_steps: int = 50,
        guidance_scale: float = 3.0,
        rescaled_guidance: Optional[float] = None,
        seed: Optional[int] = None,
        progress_bar: Optional[bool] = True,
        zero_out_negative_prompt: bool = True,
        crop_params: Optional[torch.Tensor] = None,
        input_size_params: Optional[torch.Tensor] = None,
    ):
        """Generates image from noise.

        Performs the backward diffusion process, each inference step takes
        one forward pass through the model.

        Args:
            prompt (str or List[str]): The prompt or prompts to guide the image generation.
            negative_prompt (str or List[str]): The prompt or prompts to guide the
                image generation away from. Ignored when not using guidance
                (i.e., ignored if guidance_scale is less than 1).
                Must be the same length as list of prompts. Default: `None`.
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
        # Ensure that the prompts are in the proper format
        if isinstance(prompt, str):
            prompt = [prompt]
        if isinstance(negative_prompt, str):
            negative_prompt = [negative_prompt]
        # Verify that there are enough negative prompts for all the prompts
        if negative_prompt is not None and len(negative_prompt) != len(prompt):
            raise ValueError('Number of negative prompts must match number of prompts.')
        if negative_prompt is None:
            negative_prompt = [''] * len(prompt)

        # Create rng for the generation
        device = self.autoencoder.device
        rng_generator = torch.Generator(device=device)
        if seed:
            rng_generator = rng_generator.manual_seed(seed)  # type: ignore

        # Create the prompt embeddings
        tokenized_prompts, tokenized_prompts_pad_mask = self.tokenize_text(prompt)
        text_embeddings, pooled_text_embeddings = self.embed_text(tokenized_prompts, tokenized_prompts_pad_mask)
        # Create the negative prompt embeddings
        tokenized_negative_prompts, tokenized_negative_prompts_pad_mask = self.tokenize_text(negative_prompt)
        negative_prompt_embeds, pooled_negative_prompt_embeds = self.embed_text(tokenized_negative_prompts,
                                                                                tokenized_negative_prompts_pad_mask)
        # Concatenate prompt and negative prompt
        text_embeddings = torch.cat([negative_prompt_embeds, text_embeddings])
        if tokenized_prompts_pad_mask is not None:
            attention_masks = torch.cat([tokenized_negative_prompts_pad_mask, tokenized_prompts_pad_mask])
            attention_masks = attention_masks.to(device)
        else:
            attention_masks = None
        if pooled_text_embeddings is not None and pooled_negative_prompt_embeds is not None:
            pooled_embeddings = torch.cat([pooled_negative_prompt_embeds, pooled_text_embeddings])  # type: ignore
        else:
            pooled_embeddings = None

        # Generate initial randomness for the diffusion process
        batch_size = len(prompt)
        latents = torch.randn((batch_size, self.autoencoder.latent_channels, height, width),
                              device=device,
                              generator=rng_generator)

        # Make the additional conditioning params
        added_cond_kwargs = {}
        # Optionally prepare added time ids & embeddings
        if pooled_embeddings is not None:
            added_cond_kwargs['text_embeds'] = pooled_embeddings
        if crop_params is not None and input_size_params is not None:
            crop_params = torch.cat([crop_params, crop_params])
            input_size_params = torch.cat([input_size_params, input_size_params])
            output_size_params = torch.tensor([width, height], dtype=text_embeddings.dtype).repeat(batch_size, 1)
            output_size_params = torch.cat([output_size_params, output_size_params])
            add_time_ids = torch.cat([input_size_params, crop_params, output_size_params], dim=1).to(device)
            added_cond_kwargs['time_ids'] = add_time_ids
        # Make the timesteps
        timesteps = torch.linspace(self.T_max, 0, num_inference_steps + 1, device=device)
        time_deltas = -torch.diff(timesteps) * (torch.pi / (2 * self.T_max))
        timesteps = timesteps[:-1]
        # backward diffusion process
        for i, t in enumerate(tqdm(timesteps, disable=not progress_bar)):
            # Need to duplicate latents for CFG
            latent_model_input = torch.cat([latents] * 2)
            # Optionally add fourier features
            if self.fourier_feature_transform is not None:
                latent_model_input = self.fourier_feature_transform(latent_model_input)
            # Model prediction
            outputs = self.model(latent_model_input,
                                 t,
                                 encoder_hidden_states=text_embeddings,
                                 encoder_attention_mask=attention_masks,
                                 added_cond_kwargs=added_cond_kwargs).sample

            # Perform CFG
            pred_uncond, pred_text = outputs.chunk(2)
            pred = pred_uncond + guidance_scale * (pred_text - pred_uncond)
            # Optionally rescale the classifer free guidance
            if rescaled_guidance is not None:
                std_pos = torch.std(pred_text, dim=(1, 2, 3), keepdim=True)
                std_cfg = torch.std(pred, dim=(1, 2, 3), keepdim=True)
                pred_rescaled = pred * (std_pos / std_cfg)
                pred = pred_rescaled * rescaled_guidance + pred * (1 - rescaled_guidance)
            latents = self.update_latents(latents, pred, t, time_deltas[i])

        # We now use the vae to decode the generated latents back into the image.
        # scale and decode the image latents with vae
        latents = self.unscale_latents(latents)
        image = self.autoencoder.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        return image.detach()  # (batch*num_images_per_prompt, channel, h, w)
