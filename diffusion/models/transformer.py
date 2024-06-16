# Copyright 2022 MosaicML Diffusion authors
# SPDX-License-Identifier: Apache-2.0

"""Diffusion Transformer model."""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from composer.models import ComposerModel
from torchmetrics import MeanSquaredError
from tqdm.auto import tqdm


def modulate(x, shift, scale):
    """Modulate the input with the shift and scale."""
    return x * (1.0 + scale) + shift


def timestep_embedding(timesteps, dim, max_period=10000):
    """Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) /
                      half).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class SelfAttention(nn.Module):
    """Standard self attention layer that supports masking."""

    def __init__(self, num_features, num_heads):
        super().__init__()
        self.num_features = num_features
        self.num_heads = num_heads
        # Linear layer to get q, k, and v
        self.qkv = nn.Linear(self.num_features, 3 * self.num_features)
        # QK layernorms
        self.q_norm = nn.LayerNorm(self.num_features, elementwise_affine=False, eps=1e-6)
        self.k_norm = nn.LayerNorm(self.num_features, elementwise_affine=False, eps=1e-6)
        # Linear layer to get the output
        self.output_layer = nn.Linear(self.num_features, self.num_features)
        # Initialize all biases to zero
        nn.init.zeros_(self.qkv.bias)
        nn.init.zeros_(self.output_layer.bias)
        # Init the standard deviation of the weights to 0.02
        nn.init.normal_(self.qkv.weight, std=0.02)
        nn.init.normal_(self.output_layer.weight, std=0.02)

    def forward(self, x, mask=None):
        # Get the shape of the input
        B, T, C = x.size()
        # Calculate the query, key, and values all in one go
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = self.q_norm(q)
        k = self.k_norm(k)
        # After this, q, k, and v will have shape (B, T, C)
        # Reshape the query, key, and values for multi-head attention
        # Also want to swap the sequence length and the head dimension for later matmuls
        q = q.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)
        k = k.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)
        v = v.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)
        # Native torch attention
        attention_out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)  # (B, H, T, C/H)
        # Swap the sequence length and the head dimension back and get rid of num_heads.
        attention_out = attention_out.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, C)
        # Final linear layer to get the output
        out = self.output_layer(attention_out)
        return out


class DiTBlock(nn.Module):
    """Transformer block that supports masking."""

    def __init__(self, num_features, num_heads, expansion_factor=4):
        super().__init__()
        self.num_features = num_features
        self.num_heads = num_heads
        self.expansion_factor = expansion_factor
        # Layer norm before the self attention
        self.layer_norm_1 = nn.LayerNorm(self.num_features, elementwise_affine=False, eps=1e-6)
        self.attention = SelfAttention(self.num_features, self.num_heads)
        # Layer norm before the MLP
        self.layer_norm_2 = nn.LayerNorm(self.num_features, elementwise_affine=False, eps=1e-6)
        # MLP layers. The MLP expands and then contracts the features.
        self.linear_1 = nn.Linear(self.num_features, self.expansion_factor * self.num_features)
        self.nonlinearity = nn.GELU(approximate='tanh')
        self.linear_2 = nn.Linear(self.expansion_factor * self.num_features, self.num_features)
        # Initialize all biases to zero
        nn.init.zeros_(self.linear_1.bias)
        nn.init.zeros_(self.linear_2.bias)
        # Initialize the linear layer weights to have a standard deviation of 0.02
        nn.init.normal_(self.linear_1.weight, std=0.02)
        nn.init.normal_(self.linear_2.weight, std=0.02)
        # AdaLN MLP
        self.adaLN_mlp_linear = nn.Linear(self.num_features, 6 * self.num_features, bias=True)
        # Initialize the modulations to zero. This will ensure the block acts as identity at initialization
        nn.init.zeros_(self.adaLN_mlp_linear.weight)
        nn.init.zeros_(self.adaLN_mlp_linear.bias)
        self.adaLN_mlp = nn.Sequential(nn.SiLU(), self.adaLN_mlp_linear)

    def forward(self, x, c, mask=None):
        # Calculate the modulations. Each is shape (B, num_features).
        mods = self.adaLN_mlp(c).unsqueeze(1).chunk(6, dim=2)
        # Forward, with modulations
        y = modulate(self.layer_norm_1(x), mods[0], mods[1])
        y = mods[2] * self.attention(y, mask=mask)
        x = x + y
        y = modulate(self.layer_norm_2(x), mods[3], mods[4])
        y = self.linear_1(y)
        y = self.nonlinearity(y)
        y = mods[5] * self.linear_2(y)
        x = x + y
        return x


class DiffusionTransformer(nn.Module):
    """Transformer model for diffusion."""

    def __init__(self,
                 num_features: int,
                 num_heads: int,
                 num_layers: int,
                 input_features: int = 192,
                 input_max_sequence_length: int = 1024,
                 input_dimension: int = 2,
                 conditioning_features: int = 1024,
                 conditioning_max_sequence_length: int = 77,
                 conditioning_dimension: int = 1,
                 expansion_factor: int = 4):
        super().__init__()
        # Params for the network architecture
        self.num_features = num_features
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.expansion_factor = expansion_factor
        # Params for input embeddings
        self.input_features = input_features
        self.input_dimension = input_dimension
        self.input_max_sequence_length = input_max_sequence_length
        # Params for conditioning embeddings
        self.conditioning_features = conditioning_features
        self.conditioning_dimension = conditioning_dimension
        self.conditioning_max_sequence_length = conditioning_max_sequence_length

        # Projection layer for the input sequence
        self.input_embedding = nn.Linear(self.input_features, self.num_features)
        # Embedding layer for the input sequence
        input_position_embedding = torch.randn(self.input_dimension, self.input_max_sequence_length, self.num_features)
        input_position_embedding /= math.sqrt(self.num_features)
        self.input_position_embedding = torch.nn.Parameter(input_position_embedding, requires_grad=True)
        # Projection layer for the conditioning sequence
        self.conditioning_embedding = nn.Linear(self.conditioning_features, self.num_features)
        # Embedding layer for the conditioning sequence
        conditioning_position_embedding = torch.randn(self.conditioning_dimension,
                                                      self.conditioning_max_sequence_length, self.num_features)
        conditioning_position_embedding /= math.sqrt(self.num_features)
        self.conditioning_position_embedding = torch.nn.Parameter(conditioning_position_embedding, requires_grad=True)
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            DiTBlock(self.num_features, self.num_heads, expansion_factor=self.expansion_factor)
            for _ in range(self.num_layers)
        ])
        # Output projection layer
        self.final_norm = nn.LayerNorm(self.num_features, elementwise_affine=False, eps=1e-6)
        self.final_linear = nn.Linear(self.num_features, self.input_features)
        # Init the output layer to zero
        nn.init.zeros_(self.final_linear.weight)
        nn.init.zeros_(self.final_linear.bias)
        # AdaLN MLP for the output layer
        self.adaLN_mlp_linear = nn.Linear(self.num_features, 2 * self.num_features)
        # Init the modulations to zero. This will ensure the block acts as identity at initialization
        nn.init.zeros_(self.adaLN_mlp_linear.weight)
        nn.init.zeros_(self.adaLN_mlp_linear.bias)
        self.adaLN_mlp = nn.Sequential(nn.SiLU(), self.adaLN_mlp_linear)

    def forward(self,
                x,
                input_coords,
                t,
                conditioning=None,
                conditioning_coords=None,
                input_mask=None,
                conditioning_mask=None):
        # TODO: Fix embeddings, fix embedding norms
        # Embed the timestep
        t = timestep_embedding(t, self.num_features)

        # Embed the input
        y = self.input_embedding(x)  # (B, T1, C)
        # Get the input position embeddings and add them to the input
        input_grid = torch.arange(self.input_dimension).view(1, 1, self.input_dimension).expand(
            y.shape[0], y.shape[1], self.input_dimension)
        y_position_embeddings = self.input_position_embedding[input_grid,
                                                              input_coords, :]  # (B, T1, input_dimension, C)
        y_position_embeddings = y_position_embeddings.sum(dim=2)  # (B, T1, C)
        y = y + y_position_embeddings  # (B, T1, C)
        if input_mask is None:
            mask = torch.ones(x.shape[0], x.shape[1], device=x.device)
        else:
            mask = input_mask

        if conditioning is not None:
            assert conditioning_coords is not None
            # Embed the conditioning
            c = self.conditioning_embedding(conditioning)  # (B, T2, C)
            # Get the conditioning position embeddings and add them to the conditioning
            c_grid = torch.arange(self.conditioning_dimension).view(1, 1, self.conditioning_dimension).expand(
                c.shape[0], c.shape[1], self.conditioning_dimension)
            c_position_embeddings = self.conditioning_position_embedding[
                c_grid, conditioning_coords, :]  # (B, T2, conditioning_dimension, C)
            c_position_embeddings = c_position_embeddings.sum(dim=2)  # (B, T2, C)
            c = c + c_position_embeddings  # (B, T2, C)
            # Concatenate the input and conditioning sequences
            y = torch.cat([y, c], dim=1)  # (B, T1 + T2, C)
            # Concatenate the masks
            if conditioning_mask is None:
                conditioning_mask = torch.ones(conditioning.shape[0], conditioning.shape[1], device=conditioning.device)
            mask = torch.cat([mask, conditioning_mask], dim=1)  # (B, T1 + T2)

        # Expand the mask to the right shape
        mask = mask.bool()
        mask = mask.unsqueeze(-1) & mask.unsqueeze(1)  # (B, T1 + T2, T1 + T2)
        identity = torch.eye(mask.shape[1], device=mask.device,
                             dtype=mask.dtype).unsqueeze(0).expand(mask.shape[0], -1, -1)
        mask = mask | identity
        mask = mask.unsqueeze(1)  # (B, 1, T1 + T2, T1 + T2)

        # Pass through the transformer blocks
        for block in self.transformer_blocks:
            y = block(y, t, mask=mask)
        # Throw away the conditioning tokens
        y = y[:, 0:x.shape[1], :]
        # Pass through the output layers to get the right number of elements
        mods = self.adaLN_mlp(t).unsqueeze(1).chunk(2, dim=2)
        y = modulate(self.final_norm(y), mods[0], mods[1])
        y = self.final_linear(y)
        return y


class ComposerTextToImageDiT(ComposerModel):
    """ComposerModel for text to image with a diffusion transformer.

    Args:
        model (DiffusionTransformer): Core diffusion model.
        autoencoder (torch.nn.Module): HuggingFace or compatible vae.
            must support `.encode()` and `decode()` functions.
        text_encoder (torch.nn.Module): HuggingFace CLIP or LLM text enoder.
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer used for
            text_encoder. For a `CLIPTextModel` this will be the
            `CLIPTokenizer` from HuggingFace transformers.
        noise_scheduler (diffusers.SchedulerMixin): HuggingFace diffusers
            noise scheduler. Used during the forward diffusion process (training).
        inference_noise_scheduler (diffusers.SchedulerMixin): HuggingFace diffusers
            noise scheduler. Used during the backward diffusion process (inference).
        prediction_type (str): The type of prediction to use. Currently `epsilon`, `v_prediction` are supported.
        latent_mean (Optional[tuple[float]]): The means of the latent space. If not specified, defaults to
            4 * (0.0,). Default: `None`.
        latent_std (Optional[tuple[float]]): The standard deviations of the latent space. If not specified,
            defaults to 4 * (1/0.13025,). Default: `None`.
        patch_size (int): The size of the patches in the image latents. Default: `2`.
        downsample_factor (int): The factor by which the image is downsampled by the autoencoder. Default `8`.
        latent_channels (int): The number of channels in the autoencoder latent space. Default: `4`.
        image_key (str): The name of the images in the dataloader batch. Default: `image`.
        caption_key (str): The name of the caption in the dataloader batch. Default: `caption`.
        caption_mask_key (str): The name of the caption mask in the dataloader batch. Default: `caption_mask`.
    """

    def __init__(
        self,
        model: DiffusionTransformer,
        autoencoder: torch.nn.Module,
        text_encoder: torch.nn.Module,
        tokenizer,
        noise_scheduler,
        inference_noise_scheduler,
        prediction_type: str = 'epsilon',
        latent_mean: Optional[tuple[float]] = None,
        latent_std: Optional[tuple[float]] = None,
        patch_size: int = 2,
        downsample_factor: int = 8,
        latent_channels: int = 4,
        image_key: str = 'image',
        caption_key: str = 'caption',
        caption_mask_key: str = 'caption_mask',
    ):
        super().__init__()
        self.model = model
        self.autoencoder = autoencoder
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.noise_scheduler = noise_scheduler
        self.inference_scheduler = inference_noise_scheduler
        self.prediction_type = prediction_type.lower()
        if self.prediction_type not in ['epsilon', 'v_prediction']:
            raise ValueError(f'Unrecognized prediction type {self.prediction_type}')
        if latent_mean is None:
            self.latent_mean = 4 * (0.0)
        if latent_std is None:
            self.latent_std = 4 * (1 / 0.18215,)
        self.latent_mean = torch.tensor(latent_mean).view(1, -1, 1, 1)
        self.latent_std = torch.tensor(latent_std).view(1, -1, 1, 1)
        self.patch_size = patch_size
        self.downsample_factor = downsample_factor
        self.latent_channels = latent_channels
        self.image_key = image_key
        self.caption_key = caption_key
        self.caption_mask_key = caption_mask_key

        # freeze text_encoder during diffusion training and use half precision
        self.autoencoder.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.autoencoder = self.autoencoder.half()
        self.text_encoder = self.text_encoder.half()

        # Only FSDP wrap models we are training
        self.model._fsdp_wrap = True
        self.autoencoder._fsdp_wrap = False
        self.text_encoder._fsdp_wrap = False

        # Params for MFU computation, subtract off the embedding params
        self.n_params = sum(p.numel() for p in self.model.parameters())
        self.n_params -= self.model.input_position_embedding.numel()
        self.n_params -= self.model.conditioning_position_embedding.numel()

        # Set up metrics
        self.train_metrics = [MeanSquaredError()]
        self.val_metrics = [MeanSquaredError()]

        # Optional rng generator
        self.rng_generator: Optional[torch.Generator] = None

    def _apply(self, fn):
        super(ComposerTextToImageDiT, self)._apply(fn)
        self.latent_mean = fn(self.latent_mean)
        self.latent_std = fn(self.latent_std)
        return self

    def set_rng_generator(self, rng_generator: torch.Generator):
        """Sets the rng generator for the model."""
        self.rng_generator = rng_generator

    def flops_per_batch(self, batch):
        batch_size = batch[self.image_key].shape[0]
        height, width = batch[self.image_key].shape[2:]
        input_seq_len = height * width / self.patch_size**2
        cond_seq_len = batch[self.caption_key].shape[1]
        seq_len = input_seq_len + cond_seq_len
        # Calulate forward flops excluding attention
        param_flops = 2 * self.n_params * batch_size * seq_len
        # Calculate flops for attention layers
        attention_flops = 4 * self.model.num_layers * seq_len**2 * self.model.num_features * batch_size
        return 3 * param_flops + 3 * attention_flops

    def patchify(self, latents):
        # Assume img is a tensor of shape [B, C, H, W]
        B, C, H, W = latents.shape
        assert H % self.patch_size == 0 and W % self.patch_size == 0, 'Image dimensions must be divisible by patch_size'
        # Reshape and permute to get non-overlapping patches
        num_H_patches = H // self.patch_size
        num_W_patches = W // self.patch_size
        patches = latents.reshape(B, C, num_H_patches, self.patch_size, num_W_patches, self.patch_size)
        patches = patches.permute(0, 2, 4, 1, 3, 5).reshape(B, -1, C * self.patch_size * self.patch_size)
        # Generate coordinates for each patch
        coords = torch.tensor([(i, j) for i in range(num_H_patches) for j in range(num_W_patches)])
        coords = coords.unsqueeze(0).expand(B, -1, -1).reshape(B, -1, 2)
        return patches, coords

    def unpatchify(self, patches, coords):
        # Assume patches is a tensor of shape [num_patches, C * patch_size * patch_size]
        C = patches.shape[1] // (self.patch_size * self.patch_size)
        # Calculate the height and width of the original image from the coordinates
        H = coords[:, 0].max() * self.patch_size + self.patch_size
        W = coords[:, 1].max() * self.patch_size + self.patch_size
        # Initialize an empty tensor for the reconstructed image
        img = torch.zeros((C, H, W), device=patches.device, dtype=patches.dtype)
        # Iterate over the patches and their coordinates
        for patch, (y, x) in zip(patches, self.patch_size * coords):
            # Reshape the patch to [C, patch_size, patch_size]
            patch = patch.view(C, self.patch_size, self.patch_size)
            # Place the patch in the corresponding location in the image
            img[:, y:y + self.patch_size, x:x + self.patch_size] = patch
        return img

    def diffusion_forward_process(self, inputs: torch.Tensor):
        """Diffusion forward process."""
        # Sample a timestep for every element in the batch
        timesteps = torch.randint(0,
                                  len(self.noise_scheduler), (inputs.shape[0],),
                                  device=inputs.device,
                                  generator=self.rng_generator)
        # Generate the noise, applied to the whole input sequence
        noise = torch.randn(*inputs.shape, device=inputs.device, generator=self.rng_generator)
        # Add the noise to the latents according to the schedule
        noised_inputs = self.noise_scheduler.add_noise(inputs, noise, timesteps)
        # Generate the targets
        if self.prediction_type == 'epsilon':
            targets = noise
        elif self.prediction_type == 'sample':
            targets = inputs
        elif self.prediction_type == 'v_prediction':
            targets = self.noise_scheduler.get_velocity(inputs, noise, timesteps)
        else:
            raise ValueError(
                f'prediction type must be one of sample, epsilon, or v_prediction. Got {self.prediction_type}')
        return noised_inputs, targets, timesteps

    def forward(self, batch):
        # Get the inputs
        image, caption, caption_mask = batch[self.image_key], batch[self.caption_key], batch[self.caption_mask_key]
        # Get the text embeddings and image latents
        with torch.cuda.amp.autocast(enabled=False):
            latents = self.autoencoder.encode(image.half())['latent_dist'].sample().data
            text_encoder_out = self.text_encoder(caption, attention_mask=caption_mask)
            text_embeddings = text_encoder_out[0]
        # Make the text embedding coords
        text_embeddings_coords = torch.arange(text_embeddings.shape[1], device=text_embeddings.device)
        text_embeddings_coords = text_embeddings_coords.unsqueeze(0).expand(text_embeddings.shape[0], -1).unsqueeze(-1)
        # Zero dropped captions if needed
        if 'drop_caption_mask' in batch.keys():
            text_embeddings *= batch['drop_caption_mask'].view(-1, 1, 1)
        # Scale and patchify the latents
        latents = (latents - self.latent_mean) / self.latent_std
        latent_patches, latent_coords = self.patchify(latents)
        # Diffusion forward process
        noised_inputs, targets, timesteps = self.diffusion_forward_process(latent_patches)
        # Forward through the model
        model_out = self.model(noised_inputs,
                               latent_coords,
                               timesteps,
                               conditioning=text_embeddings,
                               conditioning_coords=text_embeddings_coords,
                               input_mask=None,
                               conditioning_mask=caption_mask)
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

    def combine_attention_masks(self, attention_mask):
        if len(attention_mask.shape) == 2:
            return attention_mask
        elif len(attention_mask.shape) == 3:
            encoder_attention_mask = attention_mask[:, 0]
            for i in range(1, attention_mask.shape[1]):
                encoder_attention_mask |= attention_mask[:, i]
            return encoder_attention_mask
        else:
            raise ValueError(f'attention_mask should have either 2 or 3 dimensions: {attention_mask.shape}')

    def embed_prompt(self, prompt):
        with torch.cuda.amp.autocast(enabled=False):
            tokenized_out = self.tokenizer(prompt,
                                           padding='max_length',
                                           max_length=self.tokenizer.model_max_length,
                                           truncation=True,
                                           return_tensors='pt')
            tokenized_prompts = tokenized_out['input_ids'].to(self.text_encoder.device)
            prompt_mask = tokenized_out['attention_mask'].to(self.text_encoder.device)
            text_embeddings = self.text_encoder(tokenized_prompts, attention_mask=prompt_mask)[0]
            prompt_mask = self.combine_attention_masks(prompt_mask)
        return text_embeddings, prompt_mask

    def generate(self,
                 prompt: Optional[list] = None,
                 negative_prompt: Optional[list] = None,
                 height: Optional[int] = None,
                 width: Optional[int] = None,
                 guidance_scale: float = 7.0,
                 rescaled_guidance: Optional[float] = None,
                 num_inference_steps: int = 50,
                 progress_bar: bool = True,
                 seed: Optional[int] = None):
        """Generate from the model."""
        device = next(self.model.parameters()).device
        # Create rng for the generation
        rng_generator = torch.Generator(device=device)
        if seed:
            rng_generator = rng_generator.manual_seed(seed)

        # Get the text embeddings
        if prompt is not None:
            text_embeddings, prompt_mask = self.embed_prompt(prompt)
            text_embeddings_coords = torch.arange(text_embeddings.shape[1], device=text_embeddings.device)
            text_embeddings_coords = text_embeddings_coords.unsqueeze(0).expand(text_embeddings.shape[0], -1)
            text_embeddings_coords = text_embeddings_coords.unsqueeze(-1)
        else:
            raise ValueError('Prompt must be specified')
        if negative_prompt is not None:
            negative_text_embeddings, negative_prompt_mask = self.embed_prompt(negative_prompt)
        else:
            negative_text_embeddings = torch.zeros_like(text_embeddings)
            negative_prompt_mask = torch.zeros_like(prompt_mask)
        negative_text_embeddings_coords = torch.arange(negative_text_embeddings.shape[1],
                                                       device=negative_text_embeddings.device)
        negative_text_embeddings_coords = negative_text_embeddings_coords.unsqueeze(0).expand(
            negative_text_embeddings.shape[0], -1)
        negative_text_embeddings_coords = negative_text_embeddings_coords.unsqueeze(-1)

        # Generate initial noise
        latent_height = height // self.downsample_factor
        latent_width = width // self.downsample_factor
        latents = torch.randn(text_embeddings.shape[0],
                              self.latent_channels,
                              latent_height,
                              latent_width,
                              device=device)
        latent_patches, latent_coords = self.patchify(latents)

        # Set up for CFG
        text_embeddings = torch.cat([text_embeddings, negative_text_embeddings], dim=0)
        text_embeddings_coords = torch.cat([text_embeddings_coords, negative_text_embeddings_coords], dim=0)
        text_embeddings_mask = torch.cat([prompt_mask, negative_prompt_mask], dim=0)
        latent_coords_input = torch.cat([latent_coords] * 2)

        # Prep for reverse process
        self.inference_scheduler.set_timesteps(num_inference_steps)
        # scale the initial noise by the standard deviation required by the scheduler
        latent_patches = latent_patches * self.inference_scheduler.init_noise_sigma

        # backward diffusion process
        for t in tqdm(self.inference_scheduler.timesteps, disable=not progress_bar):
            latent_patches_input = torch.cat([latent_patches] * 2)
            latent_patches_input = self.inference_scheduler.scale_model_input(latent_patches_input, t)
            # Get the model prediction
            model_out = self.model(latent_patches_input,
                                   latent_coords_input,
                                   t.unsqueeze(0),
                                   conditioning=text_embeddings,
                                   conditioning_coords=text_embeddings_coords,
                                   input_mask=None,
                                   conditioning_mask=text_embeddings_mask)
            # Do CFG
            pred_uncond, pred_cond = model_out.chunk(2, dim=0)
            pred = pred_uncond + guidance_scale * (pred_cond - pred_uncond)
            # Update the inputs
            latent_patches = self.inference_scheduler.step(pred, t, latent_patches, generator=rng_generator).prev_sample
        # Unpatchify the latents
        latents = [self.unpatchify(latent_patches[i], latent_coords[i]) for i in range(latent_patches.shape[0])]
        latents = torch.stack(latents)
        # Scale the latents back to the original scale
        latents = latents * self.latent_std + self.latent_mean
        # Decode the latents
        image = self.autoencoder.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        return image.detach()  # (batch*num_images_per_prompt, channel, h, w)
