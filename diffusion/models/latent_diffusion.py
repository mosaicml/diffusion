# Copyright 2022 MosaicML Diffusion authors
# SPDX-License-Identifier: Apache-2.0

"""Custom latent diffusion model."""

from typing import List, Optional, Union

import torch
import torch.nn.functional as F
from composer.models import ComposerModel
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
        offset_noise (float, optional): The scale of the offset noise. If not specified, offset noise will not
            be used. Default `None`.
        latent_scale (float): The scaling factor of the latents. Default: `1.0`.
        T_max (int): The maximum time for the forward diffusion process. Default: `1000`.
        image_key (str): The name of the image inputs in the dataloader batch.
            Default: `image_tensor`.
        caption_key (str): The name of the caption inputs in the dataloader batch.
            Default: `input_ids`.
        fsdp (bool): whether to use FSDP, Default: `False`.
        encode_latents_in_fp16 (bool): whether to encode latents in fp16.
            Default: `False`.
    """

    def __init__(self,
                 model: torch.nn.Module,
                 autoencoder: AutoEncoder,
                 text_encoder,
                 tokenizer,
                 offset_noise: Optional[float] = None,
                 latent_scale: float = 1.0,
                 T_max: int = 1000,
                 image_key: str = 'image',
                 text_key: str = 'captions',
                 fsdp: bool = False,
                 encode_latents_in_fp16: bool = False):
        super().__init__()
        self.model = model
        self.autoencoder = autoencoder
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.offset_noise = offset_noise
        self.latent_scale = latent_scale
        self.T_max = T_max
        self.image_key = image_key
        self.text_key = text_key
        self.fsdp = fsdp
        self.encode_latents_in_fp16 = encode_latents_in_fp16

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
                latents = self.autoencoder.encode(images.half())['latents'].data
        else:
            latents = self.autoencoder.encode(images)['latents'].data
        return latents

    def tokenize_text(self, text: Union[str, List[str]]):
        """Tokenize text using the tokenizer."""
        if isinstance(text, str):
            text = [text]
        tokenizer_out = self.tokenizer(text, padding='max_length', truncation=True, return_tensors='pt')
        return tokenizer_out.input_ids, tokenizer_out.attention_mask

    def embed_text(self, tokenized_text):
        """Get the text embeddings from the text encoder(s)."""
        # There are possibly multiple text tokenizations depending on the model. Get them all.
        conditionings = [tokenized_text[:, i, :] for i in range(tokenized_text.shape[1])]
        conditionings = [c.view(-1, c.shape[-1]) for c in conditionings]
        if len(conditionings) > 1:
            conditioning, pooled_conditioning = self.text_encoder(conditionings)
        elif len(conditionings) == 1:
            conditioning = self.text_encoder(conditionings[0])[0]
            pooled_conditioning = None
        else:
            raise ValueError('No conditioning vectors were found.')
        return conditioning, pooled_conditioning

    def diffusion_forward_process(self, latents: torch.Tensor):
        """Diffusion forward process."""
        # Sample (continuous) timesteps
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
        # Get the (velocity) targets
        targets = -sin_t * latents + cos_t * noise
        # TODO: Implement other prediction types
        return noised_latents, targets, timesteps

    def forward(self, batch):
        images, tokenized_text = batch[self.image_key], batch[self.text_key]
        # Prep the image latents
        latents = self.embed_images(images)
        # Scale the latents by the magical number
        latents /= self.latent_scale

        # Prep the text embeddings
        conditioning, pooled_conditioning = self.embed_text(tokenized_text)
        # Zero dropped captions if needed
        if 'drop_caption_mask' in batch:
            conditioning *= batch['drop_caption_mask'].view(-1, 1, 1)
            if pooled_conditioning is not None:
                pooled_conditioning *= batch['drop_caption_mask'].view(-1, 1)

        # Diffusion forward process
        noised_latents, targets, timesteps = self.diffusion_forward_process(latents)

        # Optional additional conditioning
        added_cond_kwargs = {}
        if all(k in batch for k in self.additional_cond_keys):
            add_time_ids = torch.cat(
                [batch['cond_original_size'], batch['cond_crops_coords_top_left'], batch['cond_target_size']], dim=1)
            add_text_embeds = pooled_conditioning
            added_cond_kwargs = {'text_embeds': add_text_embeds, 'time_ids': add_time_ids}

        # Forward through the model
        model_out = self.model(noised_latents, timesteps, conditioning, added_cond_kwargs=added_cond_kwargs)['sample']
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
        one forward pass through the unet.

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
        device = next(self.autoencoder.parameters()).device
        rng_generator = torch.Generator(device=device)
        if seed:
            rng_generator = rng_generator.manual_seed(seed)  # type: ignore

        # Create the prompt embeddings
        tokenized_prompts, tokenized_prompts_pad_mask = self.tokenize_text(prompt)
        text_embeddings, pooled_text_embeddings = self.embed_text(tokenized_prompts)
        # Create the negative prompt embeddings
        tokenized_negative_prompts, tokenized_negative_prompts_pad_mask = self.tokenize_text(negative_prompt)
        negative_prompt_embeds, pooled_negative_prompt_embeds = self.embed_text(tokenized_negative_prompts)
        # Concatenate prompt and negative prompt
        text_embeddings = torch.cat([negative_prompt_embeds, text_embeddings])
        if tokenized_prompts_pad_mask is not None:
            attention_masks = torch.cat([tokenized_negative_prompts_pad_mask, tokenized_prompts_pad_mask])
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
        time_deltas = -torch.diff(timesteps)
        timesteps = timesteps[:-1]
        # backward diffusion process
        for i, t in enumerate(tqdm(timesteps, disable=not progress_bar)):
            # Need to duplicate latents for CFG
            latent_model_input = torch.cat([latents] * 2)
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
            # compute the previous noisy sample x_t -> x_t-1
            latents = latents + time_deltas[i] * outputs

        # We now use the vae to decode the generated latents back into the image.
        # scale and decode the image latents with vae
        latents *= self.latent_scale
        image = self.autoencoder.decode(latents)['x_recon']
        image = (image / 2 + 0.5).clamp(0, 1)
        return image.detach()  # (batch*num_images_per_prompt, channel, h, w)
