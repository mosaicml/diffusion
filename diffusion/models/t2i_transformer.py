# Copyright 2022 MosaicML Diffusion authors
# SPDX-License-Identifier: Apache-2.0

"""Composer model for text to image generation with a multimodal transformer."""

from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from composer.models import ComposerModel
from composer.utils import dist
from torchmetrics import MeanSquaredError
from tqdm.auto import tqdm
from transformers import PreTrainedTokenizer

from diffusion.models.transformer import DiffusionTransformer, VectorEmbedding


def _duplicate_tensor(tensor, num_images_per_prompt):
    """Duplicate tensor for multiple generations from a single prompt."""
    batch_size, seq_len = tensor.shape[:2]
    tensor = tensor.repeat(1, num_images_per_prompt, *[
        1,
    ] * len(tensor.shape[2:]))
    return tensor.view(batch_size * num_images_per_prompt, seq_len, *[
        -1,
    ] * len(tensor.shape[2:]))


def patchify(latents: torch.Tensor, patch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Function to extract non-overlapping patches from image-like latents.

    Converts a tensor of shape [B, C, H, W] to patches of shape [B, num_patches, C * patch_size * patch_size].
    Coordinates of the patches are also returned to allow for unpatching and sequence embedding.

    Args:
        latents (torch.Tensor): Latents of shape [B, C, H, W].
        patch_size (int): Size of the patches.

    Returns:
        torch.Tensor: Patches of shape [B, num_patches, C * patch_size * patch_size].
        torch.Tensor: Coordinates of the patches. Shape [B, num_patches, 2].
    """
    # Assume img is a tensor of shape [B, C, H, W]
    B, C, H, W = latents.shape
    assert H % patch_size == 0 and W % patch_size == 0, 'Image dimensions must be divisible by patch_size'
    # Reshape and permute to get non-overlapping patches
    num_H_patches = H // patch_size
    num_W_patches = W // patch_size
    patches = latents.reshape(B, C, num_H_patches, patch_size, num_W_patches, patch_size)
    patches = patches.permute(0, 2, 4, 1, 3, 5).reshape(B, -1, C * patch_size * patch_size)
    # Generate coordinates for each patch
    y_coords = torch.arange(num_H_patches, device=latents.device).repeat_interleave(num_W_patches)
    x_coords = torch.arange(num_W_patches, device=latents.device).repeat(num_H_patches)
    coords = torch.stack((y_coords, x_coords), dim=-1).unsqueeze(0).expand(B, -1, -1)
    return patches, coords


def unpatchify(patches: torch.Tensor, coords: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Recover an image-like tensor from a sequence of patches and their coordinates.

    Converts a tensor of shape [num_patches, C * patch_size * patch_size] to an image of shape [C, H, W].
    Coordinates are used to place the patches in the correct location in the image.

    Args:
        patches (torch.Tensor): Patches of shape [num_patches, C * patch_size * patch_size].
        coords (torch.Tensor): Coordinates of the patches. Shape [num_patches, 2].
        patch_size (int): Size of the patches.
    """
    # Assume patches is a tensor of shape [num_patches, C * patch_size * patch_size]
    C = patches.shape[1] // (patch_size * patch_size)
    # Calculate the height and width of the original image from the coordinates
    H = coords[:, 0].max() * patch_size + patch_size
    W = coords[:, 1].max() * patch_size + patch_size
    # Initialize an empty tensor for the reconstructed image
    img = torch.zeros((C, H, W), device=patches.device, dtype=patches.dtype)  # type: ignore
    # Iterate over the patches and their coordinates
    for patch, (y, x) in zip(patches, patch_size * coords):
        # Reshape the patch to [C, patch_size, patch_size]
        patch = patch.view(C, patch_size, patch_size)
        # Place the patch in the corresponding location in the image
        img[:, y:y + patch_size, x:x + patch_size] = patch
    return img


class ComposerTextToImageMMDiT(ComposerModel):
    """ComposerModel for text to image with a diffusion transformer.

    Args:
        model (DiffusionTransformer): Core diffusion model.
        autoencoder (torch.nn.Module): HuggingFace or compatible vae.
            must support `.encode()` and `decode()` functions.
        text_encoder (torch.nn.Module): HuggingFace CLIP or LLM text enoder.
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer used for
            text_encoder. For a `CLIPTextModel` this will be the
            `CLIPTokenizer` from HuggingFace transformers.
        latent_mean (Optional[tuple[float]]): The means of the latent space. If not specified, defaults to
            4 * (0.0,). Default: `None`.
        latent_std (Optional[tuple[float]]): The standard deviations of the latent space. If not specified,
            defaults to 4 * (1/0.13025,). Default: `None`.
        patch_size (int): The size of the patches in the image latents. Default: `2`.
        downsample_factor (int): The factor by which the image is downsampled by the autoencoder. Default `8`.
        latent_channels (int): The number of channels in the autoencoder latent space. Default: `4`.
        timestep_mean (float): The mean of the logit-normal distribution for sampling timesteps. Default: `0.0`.
        timestep_std (float): The standard deviation of the logit-normal distribution for sampling timesteps.
            Default: `1.0`.
        timestep_shift (float): The shift parameter for the logit-normal distribution for sampling timesteps.
            A value of `1.0` is no shift. Default: `1.0`.
        image_key (str): The name of the images in the dataloader batch. Default: `image`.
        caption_key (str): The name of the caption in the dataloader batch. Default: `caption`.
        pooled_embedding_features (int): The number of features in the pooled text embeddings. Default: `768`.
    """

    def __init__(
        self,
        model: DiffusionTransformer,
        autoencoder: torch.nn.Module,
        text_encoder: torch.nn.Module,
        tokenizer,
        latent_mean: Optional[tuple[float]] = None,
        latent_std: Optional[tuple[float]] = None,
        patch_size: int = 2,
        downsample_factor: int = 8,
        latent_channels: int = 4,
        timestep_mean: float = 0.0,
        timestep_std: float = 1.0,
        timestep_shift: float = 1.0,
        image_key: str = 'image',
        caption_key: str = 'caption',
        pooled_embedding_features: int = 768,
    ):
        super().__init__()
        self.model = model
        self.autoencoder = autoencoder
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        if latent_mean is None:
            self.latent_mean = 4 * (0.0)
        if latent_std is None:
            self.latent_std = 4 * (1 / 0.18215,)
        self.latent_mean = torch.tensor(latent_mean).view(1, -1, 1, 1)
        self.latent_std = torch.tensor(latent_std).view(1, -1, 1, 1)
        self.patch_size = patch_size
        self.downsample_factor = downsample_factor
        self.latent_channels = latent_channels
        self.timestep_mean = timestep_mean
        self.timestep_std = timestep_std
        self.timestep_shift = timestep_shift
        self.image_key = image_key
        self.caption_key = caption_key
        self.pooled_embedding_features = pooled_embedding_features

        # Embedding MLP for the pooled text embeddings
        self.pooled_embedding_mlp = VectorEmbedding(pooled_embedding_features, model.num_features)

        # freeze text_encoder during diffusion training and use half precision
        self.autoencoder.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.autoencoder = self.autoencoder.half()
        self.text_encoder = self.text_encoder.half()

        # Only FSDP wrap models we are training
        self.model._fsdp_wrap = False
        self.autoencoder._fsdp_wrap = False
        self.text_encoder._fsdp_wrap = True

        # Param counts relevant for MFU computation
        # First calc the AdaLN params separately
        self.adaLN_params = sum(p.numel() for n, p in self.model.named_parameters() if 'adaLN_mlp_linear' in n)
        # For MFU calc we must be careful to prevent double counting of MMDiT flops.
        # Here, count the number of params applied to each sequence element.
        # Last block must be handled differently since post attn layers don't run on conditioning sequence
        self.n_seq_params_per_block = self.model.num_features**2 * (4 + 2 * self.model.expansion_factor)
        self.n_seq_params = self.n_seq_params_per_block * (self.model.num_layers - 1)
        self.n_seq_params += 3 * (self.model.num_features**2)
        self.n_last_layer_params = self.model.num_features**2 * (1 + 2 * self.model.expansion_factor)
        # Params only on the input sequence
        self.n_input_params = self.model.input_features * self.model.num_features
        # Params only on the conditioning sequence
        self.n_cond_params = self.model.conditioning_features * self.model.num_features

        # Set up metrics
        self.train_metrics = [MeanSquaredError()]
        self.val_metrics = [MeanSquaredError()]

        # Optional rng generator
        self.rng_generator: Optional[torch.Generator] = None

    def _apply(self, fn):
        super(ComposerTextToImageMMDiT, self)._apply(fn)
        self.latent_mean = fn(self.latent_mean)
        self.latent_std = fn(self.latent_std)
        return self

    def set_rng_generator(self, rng_generator: torch.Generator):
        """Sets the rng generator for the model."""
        self.rng_generator = rng_generator

    def flops_per_batch(self, batch) -> int:
        batch_size = batch[self.image_key].shape[0]
        height, width = batch[self.image_key].shape[2:]
        input_seq_len = height * width / (self.patch_size**2 * self.downsample_factor**2)
        cond_seq_len = batch[self.caption_key].shape[1]
        seq_len = input_seq_len + cond_seq_len + self.model.num_register_tokens
        # Calulate forward flops on full sequence excluding attention
        param_flops = 2 * self.n_seq_params * batch_size * seq_len
        # Last block contributes a bit less than other blocks
        param_flops += 2 * self.n_last_layer_params * batch_size * input_seq_len
        # Include input sequence params (comparatively small)
        param_flops += 2 * self.n_input_params * batch_size * input_seq_len
        # Include conditioning sequence params (comparatively small)
        param_flops += 2 * self.n_cond_params * batch_size * cond_seq_len
        # Include flops from adaln
        param_flops += 2 * self.adaLN_params * batch_size
        # Calculate flops for attention layers
        attention_flops = 4 * self.model.num_layers * seq_len**2 * self.model.num_features * batch_size
        return 3 * param_flops + 3 * attention_flops

    def encode_image(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode an image tensor with the autoencoder and patchify the latents."""
        with torch.amp.autocast('cuda', enabled=False):
            latents = self.autoencoder.encode(image.half())['latent_dist'].sample().data
        # Scale and patchify the latents
        latents = (latents - self.latent_mean) / self.latent_std
        latent_patches, latent_coords = patchify(latents, self.patch_size)
        return latent_patches, latent_coords

    @torch.no_grad()
    def decode_image(self, latent_patches: torch.Tensor, latent_coords: torch.Tensor) -> torch.Tensor:
        """Decode image latent patches and unpatchify the image."""
        # Unpatchify the latents
        latents = [
            unpatchify(latent_patches[i], latent_coords[i], self.patch_size) for i in range(latent_patches.shape[0])
        ]
        latents = torch.stack(latents)
        # Scale the latents back to the original scale
        latents = latents * self.latent_std + self.latent_mean
        # Decode the latents
        with torch.amp.autocast('cuda', enabled=False):
            image = self.autoencoder.decode(latents.half()).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        return image

    def tokenize_prompts(self, prompts: Union[str, List[str]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Tokenize the prompts using the model's tokenizer."""
        tokenized_out = self.tokenizer(prompts,
                                       padding='max_length',
                                       max_length=self.tokenizer.model_max_length,
                                       truncation=True,
                                       return_tensors='pt')
        return tokenized_out['input_ids'], tokenized_out['attention_mask']

    def combine_attention_masks(self, attention_masks: torch.Tensor) -> torch.Tensor:
        """Combine attention masks for the encoder if there are multiple text encoders."""
        if len(attention_masks.shape) == 2:
            return attention_masks
        elif len(attention_masks.shape) == 3:
            encoder_attention_masks = attention_masks[:, 0]
            for i in range(1, attention_masks.shape[1]):
                encoder_attention_masks |= attention_masks[:, i]
            return encoder_attention_masks
        else:
            raise ValueError(f'attention_mask should have either 2 or 3 dimensions: {attention_masks.shape}')

    def make_text_embeddings_coords(self, text_embeddings: torch.Tensor) -> torch.Tensor:
        """Make text embeddings coordinates for the transformer."""
        text_embeddings_coords = torch.arange(text_embeddings.shape[1], device=text_embeddings.device)
        text_embeddings_coords = text_embeddings_coords.unsqueeze(0).expand(text_embeddings.shape[0], -1)
        text_embeddings_coords = text_embeddings_coords.unsqueeze(-1)
        return text_embeddings_coords

    def embed_tokenized_prompts(self, tokenized_prompts: torch.Tensor,
                                attention_masks: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Use the model's text encoder to embed tokenized prompts and create pooled text embeddings."""
        with torch.amp.autocast('cuda', enabled=False):
            # Ensure text embeddings are not longer than the model can handle
            if tokenized_prompts.shape[1] > self.model.conditioning_max_sequence_length:
                tokenized_prompts = tokenized_prompts[:, :self.model.conditioning_max_sequence_length]
            text_encoder_out = self.text_encoder(tokenized_prompts, attention_mask=attention_masks)
            text_embeddings, pooled_text_embeddings = text_encoder_out[0], text_encoder_out[1]
            text_embeddings_coords = self.make_text_embeddings_coords(text_embeddings)
        # Ensure the embeddings are the same dtype as the model
        text_embeddings = text_embeddings.to(next(self.model.parameters()).dtype)
        pooled_text_embeddings = pooled_text_embeddings.to(next(self.pooled_embedding_mlp.parameters()).dtype)
        # Encode the pooled embeddings
        pooled_text_embeddings = self.pooled_embedding_mlp(pooled_text_embeddings)
        return text_embeddings, text_embeddings_coords, pooled_text_embeddings

    def diffusion_forward_process(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Diffusion forward process using a rectified flow."""
        if not self.model.training:
            # Sample equally spaced timesteps across all devices
            global_batch_size = inputs.shape[0] * dist.get_world_size()
            global_timesteps = torch.linspace(0, 1, global_batch_size)
            # Get this device's subset of all the timesteps
            idx_offset = dist.get_global_rank() * inputs.shape[0]
            timesteps = global_timesteps[idx_offset:idx_offset + inputs.shape[0]].to(inputs.device)
            timesteps = timesteps.view(-1, 1, 1)
        else:
            # Sample timesteps according to a logit-normal distribution
            u = torch.randn(inputs.shape[0], device=inputs.device, generator=self.rng_generator, dtype=inputs.dtype)
            u = self.timestep_mean + self.timestep_std * u
            timesteps = torch.sigmoid(u).view(-1, 1, 1)
            timesteps = self.timestep_shift * timesteps / (1 + (self.timestep_shift - 1) * timesteps)
        # Then, add the noise to the latents according to the recitified flow
        noise = torch.randn(*inputs.shape, device=inputs.device, generator=self.rng_generator)
        noised_inputs = (1 - timesteps) * inputs + timesteps * noise
        # Compute the targets, which are the velocities
        targets = noise - inputs
        return noised_inputs, targets, timesteps[:, 0, 0]

    def forward(self, batch):
        # Get the inputs
        image, caption, caption_mask = batch[self.image_key], batch[self.caption_key], batch[self.caption_mask_key]
        # Get the image latents
        latent_patches, latent_coords = self.encode_image(image)
        # Get the text embeddings and their coords
        text_embeddings, text_embeddings_coords, pooled_text_embeddings = self.embed_tokenized_prompts(
            caption, caption_mask)
        # Diffusion forward process
        noised_inputs, targets, timesteps = self.diffusion_forward_process(latent_patches)
        # Forward through the model
        model_out = self.model(noised_inputs,
                               latent_coords,
                               timesteps,
                               conditioning=text_embeddings,
                               conditioning_coords=text_embeddings_coords,
                               constant_conditioning=pooled_text_embeddings)
        return {'predictions': model_out, 'targets': targets, 'timesteps': timesteps}

    def loss(self, outputs, batch):
        """MSE loss between outputs and targets."""
        loss = F.mse_loss(outputs['predictions'], outputs['targets'])
        return loss

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

    def make_sampling_timesteps(self, N: int):
        timesteps = torch.linspace(1, 0, N + 1)
        timesteps = self.timestep_shift * timesteps / (1 + (self.timestep_shift - 1) * timesteps)
        # Make timestep differences
        delta_t = timesteps[:-1] - timesteps[1:]
        return timesteps[:-1], delta_t

    @torch.no_grad()
    def generate(self,
                 prompt: Union[str, list],
                 negative_prompt: Optional[Union[str, list]] = None,
                 height: int = 256,
                 width: int = 256,
                 guidance_scale: float = 7.0,
                 rescaled_guidance: Optional[float] = None,
                 num_inference_steps: int = 50,
                 num_images_per_prompt: int = 1,
                 progress_bar: bool = True,
                 seed: Optional[int] = None):
        """Run generation for the model.

        Args:
            prompt (str, list): Prompt or prompts for the generation.
            negative_prompt (Optional[str, list]): Negative prompt or prompts for the generation. Default: `None`.
            height (int): Height of the generated images. Default: `256`.
            width (int): Width of the generated images. Default: `256`.
            guidance_scale (float): Scale for the guidance. Default: `7.0`.
            rescaled_guidance (Optional[float]): Rescale the guidance. Default: `None`.
            num_inference_steps (int): Number of inference steps. Default: `50`.
            num_images_per_prompt (int): Number of images per prompt. Default: `1`.
            progress_bar (bool): Whether to show a progress bar. Default: `True`.
            seed (Optional[int]): Seed for the generation. Default: `None`.

        Returns:
            torch.Tensor: Generated images. Shape [batch*num_images_per_prompt, channel, h, w].
        """
        device = next(self.model.parameters()).device
        # Create rng for the generation
        rng_generator = torch.Generator(device=device)
        if seed:
            rng_generator = rng_generator.manual_seed(seed)

        # Set default negative prompts to empty string if not provided
        if isinstance(prompt, str):
            prompt = [prompt]
        if isinstance(negative_prompt, str):
            negative_prompt = [negative_prompt] * len(prompt)
        elif isinstance(negative_prompt, list):
            assert len(negative_prompt) == len(prompt), 'Prompt and negative prompt must have the same length.'
        elif negative_prompt is None:
            negative_prompt = ['' for _ in prompt]
        # Duplicate the images in the prompt and negative prompt if needed.
        prompt = [item for item in prompt for _ in range(num_images_per_prompt)]
        negative_prompt = [item for item in negative_prompt for _ in range(num_images_per_prompt)]
        # Tokenize both prompt and negative prompts
        prompt_tokens, prompt_mask = self.tokenize_prompts(prompt)
        prompt_tokens, prompt_mask = prompt_tokens.to(device), prompt_mask.to(device)
        negative_prompt_tokens, negative_prompt_mask = self.tokenize_prompts(negative_prompt)
        negative_prompt_tokens, negative_prompt_mask = negative_prompt_tokens.to(device), negative_prompt_mask.to(
            device)
        # Embed the tokenized prompts and negative prompts
        text_embeddings, text_embeddings_coords, pooled_embedding = self.embed_tokenized_prompts(
            prompt_tokens, prompt_mask)
        neg_text_embeddings, neg_text_embeddings_coords, pooled_neg_embedding = self.embed_tokenized_prompts(
            negative_prompt_tokens, negative_prompt_mask)

        # Generate initial noise
        latent_height = height // self.downsample_factor
        latent_width = width // self.downsample_factor
        latents = torch.randn(text_embeddings.shape[0],
                              self.latent_channels,
                              latent_height,
                              latent_width,
                              device=device)
        latent_patches, latent_coords = patchify(latents, self.patch_size)

        # Set up for CFG
        text_embeddings = torch.cat([text_embeddings, neg_text_embeddings], dim=0)
        text_embeddings_coords = torch.cat([text_embeddings_coords, neg_text_embeddings_coords], dim=0)
        pooled_embedding = torch.cat([pooled_embedding, pooled_neg_embedding], dim=0)
        latent_coords_input = torch.cat([latent_coords, latent_coords], dim=0)

        # backward diffusion process
        timesteps, delta_t = self.make_sampling_timesteps(num_inference_steps)
        timesteps, delta_t = timesteps.to(device), delta_t.to(device)
        for i, t in tqdm(enumerate(timesteps), disable=not progress_bar):
            latent_patches_input = torch.cat([latent_patches, latent_patches], dim=0)
            # Get the model prediction
            model_out = self.model(latent_patches_input,
                                   latent_coords_input,
                                   t.unsqueeze(0),
                                   conditioning=text_embeddings,
                                   conditioning_coords=text_embeddings_coords,
                                   constant_conditioning=pooled_embedding)
            # Do CFG
            pred_cond, pred_uncond = model_out.chunk(2, dim=0)
            pred = pred_uncond + guidance_scale * (pred_cond - pred_uncond)
            # Optionally rescale the classifer free guidance
            if rescaled_guidance is not None:
                std_pos = torch.std(pred_cond, dim=(1, 2), keepdim=True)
                std_cfg = torch.std(pred, dim=(1, 2), keepdim=True)
                pred_rescaled = pred * (std_pos / std_cfg)
                pred = pred_rescaled * rescaled_guidance + pred * (1 - rescaled_guidance)
            # Update the latents
            latent_patches = latent_patches - pred * delta_t[i]
        # Decode the latents
        image = self.decode_image(latent_patches, latent_coords)
        return image.detach()  # (batch*num_images_per_prompt, channel, h, w)


class ComposerPrecomputedTextLatentsToImageMMDiT(ComposerModel):
    """ComposerModel for text to image with a diffusion transformer and precomputed text latents.

    Args:
        model (DiffusionTransformer): Core diffusion model.
        autoencoder (torch.nn.Module): HuggingFace or compatible vae.
            must support `.encode()` and `decode()` functions.
        t5_tokenizer (Optional): Tokenizer for T5. Should only be specified during inference. Default: `None`.
        t5_encoder (Optional): T5 text encoder. Should only be specified during inference. Default: `None`.
        clip_tokenizer (Optional): Tokenizer for CLIP. Should only be specified during inference. Default: `None`.
        clip_encoder (Optional): CLIP text encoder. Should only be specified during inference. Default: `None`.
        latent_mean (Optional[tuple[float]]): The means of the latent space. If not specified, defaults to
            4 * (0.0,). Default: `None`.
        latent_std (Optional[tuple[float]]): The standard deviations of the latent space. If not specified,
            defaults to 4 * (1/0.13025,). Default: `None`.
        patch_size (int): The size of the patches in the image latents. Default: `2`.
        downsample_factor (int): The factor by which the image is downsampled by the autoencoder. Default `8`.
        max_seq_len (int): The maximum sequence length for the text encoders. Default: `512`.
        latent_channels (int): The number of channels in the autoencoder latent space. Default: `4`.
        timestep_mean (float): The mean of the logit-normal distribution for sampling timesteps. Default: `0.0`.
        timestep_std (float): The standard deviation of the logit-normal distribution for sampling timesteps.
            Default: `1.0`.
        timestep_shift (float): The shift parameter for the logit-normal distribution for sampling timesteps.
            A value of `1.0` is no shift. Default: `1.0`.
        image_key (str): The name of the images in the dataloader batch. Default: `image`.
        t5_latent_key (str): The key in the batch dict that contains the T5 latents. Default: `'T5_LATENTS'`.
        clip_latent_key (str): The key in the batch dict that contains the CLIP latents. Default: `'CLIP_LATENTS'`.
        clip_pooled_key (str): The key in the batch dict that contains the CLIP pooled embeddings. Default: `'CLIP_POOLED'`.
        pooled_embedding_features (int): The number of features in the pooled text embeddings. Default: `768`.
    """

    def __init__(
        self,
        model: DiffusionTransformer,
        autoencoder: torch.nn.Module,
        t5_tokenizer: Optional[PreTrainedTokenizer] = None,
        t5_encoder: Optional[torch.nn.Module] = None,
        clip_tokenizer: Optional[PreTrainedTokenizer] = None,
        clip_encoder: Optional[torch.nn.Module] = None,
        latent_mean: Optional[tuple[float]] = None,
        latent_std: Optional[tuple[float]] = None,
        patch_size: int = 2,
        downsample_factor: int = 8,
        max_seq_len: int = 512,
        latent_channels: int = 4,
        timestep_mean: float = 0.0,
        timestep_std: float = 1.0,
        timestep_shift: float = 1.0,
        image_key: str = 'image',
        t5_latent_key: str = 'T5_LATENTS',
        clip_latent_key: str = 'CLIP_LATENTS',
        clip_pooled_key: str = 'CLIP_POOLED',
        pooled_embedding_features: int = 768,
    ):
        super().__init__()
        self.model = model
        self.autoencoder = autoencoder
        self.t5_tokenizer = t5_tokenizer
        self.t5_encoder = t5_encoder
        self.clip_tokenizer = clip_tokenizer
        self.clip_encoder = clip_encoder
        if latent_mean is None:
            self.latent_mean = 4 * (0.0)
        if latent_std is None:
            self.latent_std = 4 * (1 / 0.18215,)
        self.latent_mean = torch.tensor(latent_mean).view(1, -1, 1, 1)
        self.latent_std = torch.tensor(latent_std).view(1, -1, 1, 1)
        self.patch_size = patch_size
        self.downsample_factor = downsample_factor
        self.max_seq_len = max_seq_len
        self.latent_channels = latent_channels
        self.timestep_mean = timestep_mean
        self.timestep_std = timestep_std
        self.timestep_shift = timestep_shift
        self.image_key = image_key
        self.t5_latent_key = t5_latent_key
        self.clip_latent_key = clip_latent_key
        self.clip_pooled_key = clip_pooled_key
        self.pooled_embedding_features = pooled_embedding_features

        # Embedding MLPs and norms for the pooled text embeddings
        self.t5_proj = torch.nn.Linear(4096, model.num_features)
        self.t5_ln = torch.nn.LayerNorm(model.num_features)
        self.clip_proj = torch.nn.Linear(768, model.num_features)
        self.clip_ln = torch.nn.LayerNorm(model.num_features)
        self.pooled_embedding_mlp = VectorEmbedding(pooled_embedding_features, model.num_features)
        # freeze text_encoder during diffusion training and use half precision
        self.autoencoder.requires_grad_(False)
        self.autoencoder = self.autoencoder.half()

        # Only FSDP wrap models we are training
        self.model._fsdp_wrap = True
        self.autoencoder._fsdp_wrap = False

        # Param counts relevant for MFU computation
        # First calc the AdaLN params separately
        self.adaLN_params = sum(p.numel() for n, p in self.model.named_parameters() if 'adaLN_mlp_linear' in n)
        # For MFU calc we must be careful to prevent double counting of MMDiT flops.
        # Here, count the number of params applied to each sequence element.
        # Last block must be handled differently since post attn layers don't run on conditioning sequence
        self.n_seq_params_per_block = self.model.num_features**2 * (4 + 2 * self.model.expansion_factor)
        self.n_seq_params = self.n_seq_params_per_block * (self.model.num_layers - 1)
        self.n_seq_params += 3 * (self.model.num_features**2)
        self.n_last_layer_params = self.model.num_features**2 * (1 + 2 * self.model.expansion_factor)
        # Params only on the input sequence
        self.n_input_params = self.model.input_features * self.model.num_features
        # Params only on the conditioning sequence
        self.n_cond_params = self.model.conditioning_features * self.model.num_features

        # Set up metrics
        self.train_metrics = [MeanSquaredError()]
        self.val_metrics = [MeanSquaredError()]

        # Optional rng generator
        self.rng_generator: Optional[torch.Generator] = None

    def _apply(self, fn):
        super(ComposerPrecomputedTextLatentsToImageMMDiT, self)._apply(fn)
        self.latent_mean = fn(self.latent_mean)
        self.latent_std = fn(self.latent_std)
        return self

    def set_rng_generator(self, rng_generator: torch.Generator):
        """Sets the rng generator for the model."""
        self.rng_generator = rng_generator

    def flops_per_batch(self, batch) -> int:
        batch_size = batch[self.image_key].shape[0]
        height, width = batch[self.image_key].shape[2:]
        input_seq_len = height * width / (self.patch_size**2 * self.downsample_factor**2)
        cond_seq_len = batch[self.t5_latent_key].shape[1] + batch[self.clip_latent_key].shape[1]
        seq_len = input_seq_len + cond_seq_len + self.model.num_register_tokens
        # Calulate forward flops on full sequence excluding attention
        param_flops = 2 * self.n_seq_params * batch_size * seq_len
        # Last block contributes a bit less than other blocks
        param_flops += 2 * self.n_last_layer_params * batch_size * input_seq_len
        # Include input sequence params (comparatively small)
        param_flops += 2 * self.n_input_params * batch_size * input_seq_len
        # Include conditioning sequence params (comparatively small)
        param_flops += 2 * self.n_cond_params * batch_size * cond_seq_len
        # Include flops from adaln
        param_flops += 2 * self.adaLN_params * batch_size
        # Calculate flops for attention layers
        attention_flops = 4 * self.model.num_layers * seq_len**2 * self.model.num_features * batch_size
        return 3 * param_flops + 3 * attention_flops

    def encode_image(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode an image tensor with the autoencoder and patchify the latents."""
        with torch.amp.autocast('cuda', enabled=False):
            latents = self.autoencoder.encode(image.half())['latent_dist'].sample().data
        # Scale and patchify the latents
        latents = (latents - self.latent_mean) / self.latent_std
        latent_patches, latent_coords = patchify(latents, self.patch_size)
        return latent_patches, latent_coords

    @torch.no_grad()
    def decode_image(self, latent_patches: torch.Tensor, latent_coords: torch.Tensor) -> torch.Tensor:
        """Decode image latent patches and unpatchify the image."""
        # Unpatchify the latents
        latents = [
            unpatchify(latent_patches[i], latent_coords[i], self.patch_size) for i in range(latent_patches.shape[0])
        ]
        latents = torch.stack(latents)
        # Scale the latents back to the original scale
        latents = latents * self.latent_std + self.latent_mean
        # Decode the latents
        with torch.amp.autocast('cuda', enabled=False):
            image = self.autoencoder.decode(latents.half()).sample
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
        t5_tokenized_captions = t5_tokenizer_out['input_ids'].to(device)
        t5_attn_mask = t5_tokenizer_out['attention_mask'].to(torch.bool).to(device)
        t5_embed = self.t5_encoder(input_ids=t5_tokenized_captions, attention_mask=t5_attn_mask)[0]
        # Encode with CLIP
        clip_tokenizer_out = self.clip_tokenizer(text,
                                                 padding='max_length',
                                                 max_length=self.clip_tokenizer.model_max_length,
                                                 truncation=True,
                                                 return_tensors='pt')
        clip_tokenized_captions = clip_tokenizer_out['input_ids'].to(device)
        clip_attn_mask = clip_tokenizer_out['attention_mask'].to(torch.bool).to(device)
        clip_out = self.clip_encoder(input_ids=clip_tokenized_captions,
                                     attention_mask=clip_attn_mask,
                                     output_hidden_states=True)
        clip_embed = clip_out.hidden_states[-2]
        pooled_embeddings = clip_out[1]
        return t5_embed, clip_embed, pooled_embeddings

    def prepare_text_embeddings(self, t5_embed: torch.Tensor, clip_embed: torch.Tensor) -> torch.Tensor:
        if t5_embed.shape[1] > self.max_seq_len:
            t5_embed = t5_embed[:, :self.max_seq_len]
        if clip_embed.shape[1] > self.max_seq_len:
            clip_embed = clip_embed[:, :self.max_seq_len]
        t5_embed = self.t5_proj(t5_embed)
        clip_embed = self.clip_proj(clip_embed)
        # Apply layernorms
        t5_embed = self.t5_ln(t5_embed)
        clip_embed = self.clip_ln(clip_embed)
        # Concatenate the text embeddings
        text_embeds = torch.cat([t5_embed, clip_embed], dim=1)
        return text_embeds

    def make_text_embeddings_coords(self, text_embeddings: torch.Tensor) -> torch.Tensor:
        """Make text embeddings coordinates for the transformer."""
        text_embeddings_coords = torch.arange(text_embeddings.shape[1], device=text_embeddings.device)
        text_embeddings_coords = text_embeddings_coords.unsqueeze(0).expand(text_embeddings.shape[0], -1)
        text_embeddings_coords = text_embeddings_coords.unsqueeze(-1)
        return text_embeddings_coords

    def diffusion_forward_process(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Diffusion forward process using a rectified flow."""
        if not self.model.training:
            # Sample equally spaced timesteps across all devices
            global_batch_size = inputs.shape[0] * dist.get_world_size()
            global_timesteps = torch.linspace(0, 1, global_batch_size)
            # Get this device's subset of all the timesteps
            idx_offset = dist.get_global_rank() * inputs.shape[0]
            timesteps = global_timesteps[idx_offset:idx_offset + inputs.shape[0]].to(inputs.device)
            timesteps = timesteps.view(-1, 1, 1)
        else:
            # Sample timesteps according to a logit-normal distribution
            u = torch.randn(inputs.shape[0], device=inputs.device, generator=self.rng_generator, dtype=inputs.dtype)
            u = self.timestep_mean + self.timestep_std * u
            timesteps = torch.sigmoid(u).view(-1, 1, 1)
            timesteps = self.timestep_shift * timesteps / (1 + (self.timestep_shift - 1) * timesteps)
        # Then, add the noise to the latents according to the recitified flow
        noise = torch.randn(*inputs.shape, device=inputs.device, generator=self.rng_generator)
        noised_inputs = (1 - timesteps) * inputs + timesteps * noise
        # Compute the targets, which are the velocities
        targets = noise - inputs
        return noised_inputs, targets, timesteps[:, 0, 0]

    def forward(self, batch):
        # Get the inputs
        image = batch[self.image_key]
        # Get the image latents
        latent_patches, latent_coords = self.encode_image(image)

        # Text embeddings are shape (B, seq_len, emb_dim), optionally truncate to a max length
        t5_embed = batch[self.t5_latent_key]
        clip_embed = batch[self.clip_latent_key]
        pooled_text_embeddings = batch[self.clip_pooled_key]
        pooled_text_embeddings = self.pooled_embedding_mlp(pooled_text_embeddings)
        text_embeddings = self.prepare_text_embeddings(t5_embed, clip_embed)
        text_embeddings_coords = self.make_text_embeddings_coords(text_embeddings)

        # Diffusion forward process
        noised_inputs, targets, timesteps = self.diffusion_forward_process(latent_patches)
        # Forward through the model
        model_out = self.model(noised_inputs,
                               latent_coords,
                               timesteps,
                               conditioning=text_embeddings,
                               conditioning_coords=text_embeddings_coords,
                               constant_conditioning=pooled_text_embeddings)
        return {'predictions': model_out, 'targets': targets, 'timesteps': timesteps}

    def loss(self, outputs, batch):
        """MSE loss between outputs and targets."""
        loss = F.mse_loss(outputs['predictions'], outputs['targets'])
        return loss

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

    def make_sampling_timesteps(self, N: int):
        timesteps = torch.linspace(1, 0, N + 1)
        timesteps = self.timestep_shift * timesteps / (1 + (self.timestep_shift - 1) * timesteps)
        # Make timestep differences
        delta_t = timesteps[:-1] - timesteps[1:]
        return timesteps[:-1], delta_t

    @torch.no_grad()
    def generate(self,
                 prompt: Optional[list] = None,
                 negative_prompt: Optional[list] = None,
                 prompt_embeds: Optional[torch.Tensor] = None,
                 pooled_prompt: Optional[torch.Tensor] = None,
                 neg_prompt_embeds: Optional[torch.Tensor] = None,
                 pooled_neg_prompt: Optional[torch.Tensor] = None,
                 height: int = 256,
                 width: int = 256,
                 guidance_scale: float = 7.0,
                 rescaled_guidance: Optional[float] = None,
                 num_inference_steps: int = 50,
                 num_images_per_prompt: int = 1,
                 progress_bar: bool = True,
                 seed: Optional[int] = None):
        """Run generation for the model.

        Args:
            prompt (List[str], optional): Prompt or prompts for the generation.
            negative_prompt (List[str], optional): The prompt or prompts to guide the
                image generation away from. Ignored when not using guidance
                (i.e., ignored if guidance_scale is less than 1). Must be the same length
                as list of prompts. Only use if not using negative embeddings. Default: `None`.
            prompt_embeds (torch.Tensor): Optionally pass pre-embedded prompts instead
                of string prompts. Default: `None`.
            pooled_prompt (torch.Tensor): Optionally pass a precomputed pooled prompt embedding
                if using embeddings. Default: `None`.
            neg_prompt_embeds (torch.Tensor): Optionally pass pre-embedded negative
                prompts instead of string negative prompts.  Default: `None`.
            pooled_neg_prompt (torch.Tensor): Optionally pass a precomputed pooled negative
                prompt embedding if using embeddings. Default: `None`.
            height (int): Height of the generated images. Default: `256`.
            width (int): Width of the generated images. Default: `256`.
            guidance_scale (float): Scale for the guidance. Default: `7.0`.
            rescaled_guidance (Optional[float]): Rescale the guidance. Default: `None`.
            num_inference_steps (int): Number of inference steps. Default: `50`.
            num_images_per_prompt (int): Number of images per prompt. Default: `1`.
            progress_bar (bool): Whether to show a progress bar. Default: `True`.
            seed (Optional[int]): Seed for the generation. Default: `None`.

        Returns:
            torch.Tensor: Generated images. Shape [batch*num_images_per_prompt, channel, h, w].
        """
        device = next(self.model.parameters()).device
        # Create rng for the generation
        rng_generator = torch.Generator(device=device)
        if seed:
            rng_generator = rng_generator.manual_seed(seed)

        # Check that inputs are consistent with all embeddings or text inputs. All embeddings should be provided if using
        # embeddings, and none if using text.
        if (prompt_embeds is None) == (prompt is None):
            raise ValueError('One and only one of prompt or prompt_embeds should be provided.')
        if (pooled_prompt is None) != (prompt_embeds is None):
            raise ValueError('pooled_prompt should be provided if and only if using embeddings')
        if (pooled_neg_prompt is None) != (neg_prompt_embeds is None):
            raise ValueError('pooled_neg_prompt should be provided if and only if using embeddings')

        # If the prompt is specified as text, encode it.
        if prompt is not None:
            t5_embed, clip_embed, pooled_prompt = self.encode_text(prompt, self.vae.device)
            prompt_embeds = self.prepare_text_embeddings(t5_embed, clip_embed)
        # If negative prompt is specified as text, encode it.
        if negative_prompt is not None:
            t5_embed, clip_embed, pooled_neg_prompt = self.encode_text(negative_prompt, self.vae.device)
            neg_prompt_embeds = self.prepare_text_embeddings(t5_embed, clip_embed)

        text_embeddings = _duplicate_tensor(prompt_embeds, num_images_per_prompt)
        pooled_embeddings = _duplicate_tensor(pooled_prompt, num_images_per_prompt)

        # classifier free guidance + negative prompts
        # negative prompt is given in place of the unconditional input in classifier free guidance
        if neg_prompt_embeds is None:
            # Negative prompt is empty and we want to zero it out
            neg_prompt_embeds = torch.zeros_like(text_embeddings)
            pooled_neg_prompt = torch.zeros_like(pooled_embeddings)
        else:
            neg_prompt_embeds = _duplicate_tensor(neg_prompt_embeds, num_images_per_prompt)
            pooled_neg_prompt = _duplicate_tensor(pooled_neg_prompt, num_images_per_prompt)

        # Generate initial noise
        latent_height = height // self.downsample_factor
        latent_width = width // self.downsample_factor
        latents = torch.randn(text_embeddings.shape[0] * num_images_per_prompt,
                              self.latent_channels,
                              latent_height,
                              latent_width,
                              device=device)
        latent_patches, latent_coords = patchify(latents, self.patch_size)
        latent_coords_input = torch.cat([latent_coords, latent_coords], dim=0)
        # concat uncond + prompt
        text_embeddings = torch.cat([neg_prompt_embeds, text_embeddings])
        pooled_embeddings = torch.cat([pooled_neg_prompt, pooled_embeddings])
        # Encode the pooled embeddings
        pooled_embeddings = self.pooled_embedding_mlp(pooled_embeddings)
        # Make the text embeddings coords
        text_embeddings_coords = self.make_text_embeddings_coords(text_embeddings)

        # backward diffusion process
        timesteps, delta_t = self.make_sampling_timesteps(num_inference_steps)
        timesteps, delta_t = timesteps.to(device), delta_t.to(device)
        for i, t in tqdm(enumerate(timesteps), disable=not progress_bar):
            latent_patches_input = torch.cat([latent_patches, latent_patches], dim=0)
            # Get the model prediction
            model_out = self.model(latent_patches_input,
                                   latent_coords_input,
                                   t.unsqueeze(0),
                                   conditioning=text_embeddings,
                                   conditioning_coords=text_embeddings_coords,
                                   constant_conditioning=pooled_embeddings)
            # Do CFG
            pred_uncond, pred_cond = model_out.chunk(2, dim=0)
            pred = pred_uncond + guidance_scale * (pred_cond - pred_uncond)
            # Optionally rescale the classifer free guidance
            if rescaled_guidance is not None:
                std_pos = torch.std(pred_cond, dim=(1, 2), keepdim=True)
                std_cfg = torch.std(pred, dim=(1, 2), keepdim=True)
                pred_rescaled = pred * (std_pos / std_cfg)
                pred = pred_rescaled * rescaled_guidance + pred * (1 - rescaled_guidance)
            # Update the latents
            latent_patches = latent_patches - pred * delta_t[i]
        # Decode the latents
        image = self.decode_image(latent_patches, latent_coords)
        return image.detach()  # (batch*num_images_per_prompt, channel, h, w)
