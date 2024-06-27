# Copyright 2022 MosaicML Diffusion authors
# SPDX-License-Identifier: Apache-2.0

"""Composer model for text to image generation with a multimodal transformer."""

from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from composer.models import ComposerModel
from torchmetrics import MeanSquaredError
from tqdm.auto import tqdm

from diffusion.models.transformer import DiffusionTransformer, VectorEmbedding


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
    coords = torch.tensor([(i, j) for i in range(num_H_patches) for j in range(num_W_patches)])
    coords = coords.unsqueeze(0).expand(B, -1, -1).reshape(B, -1, 2)
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
        caption_mask_key (str): The name of the caption mask in the dataloader batch. Default: `caption_mask`.
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
        caption_mask_key: str = 'caption_mask',
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
        self.caption_mask_key = caption_mask_key
        self.pooled_embedding_features = pooled_embedding_features

        # Embedding MLP for the pooled text embeddings
        self.pooled_embedding_mlp = VectorEmbedding(pooled_embedding_features, model.num_features)

        # freeze text_encoder during diffusion training and use half precision
        self.autoencoder.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.autoencoder = self.autoencoder.half()
        self.text_encoder = self.text_encoder.half()

        # Only FSDP wrap models we are training
        self.model._fsdp_wrap = True
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
        seq_len = input_seq_len + cond_seq_len
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
        with torch.cuda.amp.autocast(enabled=False):
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
        with torch.cuda.amp.autocast(enabled=False):
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

    def embed_tokenized_prompts(
            self, tokenized_prompts: torch.Tensor,
            attention_masks: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Use the model's text encoder to embed tokenized prompts and create pooled text embeddings."""
        with torch.cuda.amp.autocast(enabled=False):
            # Ensure text embeddings are not longer than the model can handle
            if tokenized_prompts.shape[1] > self.model.conditioning_max_sequence_length:
                tokenized_prompts = tokenized_prompts[:, :self.model.conditioning_max_sequence_length]
            text_encoder_out = self.text_encoder(tokenized_prompts, attention_mask=attention_masks)
            text_embeddings, pooled_text_embeddings = text_encoder_out[0], text_encoder_out[1]
            text_mask = self.combine_attention_masks(attention_masks)
            text_embeddings_coords = self.make_text_embeddings_coords(text_embeddings)
        # Encode the pooled embeddings
        pooled_text_embeddings = self.pooled_embedding_mlp(pooled_text_embeddings)
        return text_embeddings, text_embeddings_coords, text_mask, pooled_text_embeddings

    def diffusion_forward_process(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Diffusion forward process using a rectified flow."""
        # First, sample timesteps according to a logit-normal distribution
        u = torch.randn(inputs.shape[0], device=inputs.device, generator=self.rng_generator)
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
        text_embeddings, text_embeddings_coords, caption_mask, pooled_text_embeddings = self.embed_tokenized_prompts(
            caption, caption_mask)
        # Diffusion forward process
        noised_inputs, targets, timesteps = self.diffusion_forward_process(latent_patches)
        # Forward through the model
        model_out = self.model(noised_inputs,
                               latent_coords,
                               timesteps,
                               conditioning=text_embeddings,
                               conditioning_coords=text_embeddings_coords,
                               input_mask=None,
                               conditioning_mask=caption_mask,
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
                 prompt: list,
                 negative_prompt: Optional[list] = None,
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
            prompt (list): List of prompts for the generation.
            negative_prompt (Optional[list]): List of negative prompts for the generation. Default: `None`.
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
        if negative_prompt is None:
            negative_prompt = ['' for _ in prompt]
        # Duplicate the images in the prompt and negative prompt if needed.
        prompt = [item for item in prompt for _ in range(num_images_per_prompt)]
        negative_prompt = [item for item in negative_prompt for _ in range(num_images_per_prompt)]
        # Tokenize both prompt and negative prompts
        prompt_tokens, prompt_mask = self.tokenize_prompts(prompt)
        negative_prompt_tokens, negative_prompt_mask = self.tokenize_prompts(negative_prompt)
        # Embed the tokenized prompts and negative prompts
        text_embeddings, text_embeddings_coords, prompt_mask, pooled_embedding = self.embed_tokenized_prompts(
            prompt_tokens, prompt_mask)
        neg_text_embeddings, neg_text_embeddings_coords, neg_prompt_mask, pooled_neg_embedding = self.embed_tokenized_prompts(
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
        text_embeddings_mask = torch.cat([prompt_mask, neg_prompt_mask], dim=0)
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
                                   input_mask=None,
                                   conditioning_mask=text_embeddings_mask,
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
