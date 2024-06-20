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


def get_multidimensional_position_embeddings(position_embeddings, coords):
    """Position embeddings are shape (D, T, F). Coords are shape (B, S, D)."""
    B, S, D = coords.shape
    F = position_embeddings.shape[2]
    coords = coords.reshape(B * S, D)
    sequenced_embeddings = [position_embeddings[d, coords[:, d]] for d in range(D)]
    sequenced_embeddings = torch.stack(sequenced_embeddings, dim=-1)
    sequenced_embeddings = sequenced_embeddings.view(B, S, F, D)
    return sequenced_embeddings  # (B, S, F, D)


def patchify(latents, patch_size):
    """Converts a tensor of shape [B, C, H, W] to patches of shape [B, num_patches, C * patch_size * patch_size]."""
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


def unpatchify(patches, coords, patch_size):
    """Converts a tensor of shape [num_patches, C * patch_size * patch_size] to an image of shape [C, H, W]."""
    # Assume patches is a tensor of shape [num_patches, C * patch_size * patch_size]
    C = patches.shape[1] // (patch_size * patch_size)
    # Calculate the height and width of the original image from the coordinates
    H = coords[:, 0].max() * patch_size + patch_size
    W = coords[:, 1].max() * patch_size + patch_size
    # Initialize an empty tensor for the reconstructed image
    img = torch.zeros((C, H, W), device=patches.device, dtype=patches.dtype)
    # Iterate over the patches and their coordinates
    for patch, (y, x) in zip(patches, patch_size * coords):
        # Reshape the patch to [C, patch_size, patch_size]
        patch = patch.view(C, patch_size, patch_size)
        # Place the patch in the corresponding location in the image
        img[:, y:y + patch_size, x:x + patch_size] = patch
    return img


class PreAttentionBlock(nn.Module):
    """Block to compute QKV before attention."""

    def __init__(self, num_features):
        super().__init__()
        self.num_features = num_features

        # AdaLN MLP for pre-attention. Initialized to zero so modulation acts as identity at initialization.
        self.adaLN_mlp_linear = nn.Linear(self.num_features, 2 * self.num_features, bias=True)
        nn.init.zeros_(self.adaLN_mlp_linear.weight)
        nn.init.zeros_(self.adaLN_mlp_linear.bias)
        self.adaLN_mlp = nn.Sequential(nn.SiLU(), self.adaLN_mlp_linear)
        # Input layernorm
        self.input_norm = nn.LayerNorm(self.num_features, elementwise_affine=True, eps=1e-6)
        # Linear layer to get q, k, and v
        self.qkv = nn.Linear(self.num_features, 3 * self.num_features)
        # QK layernorms. Original MMDiT used RMSNorm here.
        self.q_norm = nn.LayerNorm(self.num_features, elementwise_affine=True, eps=1e-6)
        self.k_norm = nn.LayerNorm(self.num_features, elementwise_affine=True, eps=1e-6)
        # Initialize all biases to zero
        nn.init.zeros_(self.qkv.bias)
        # Init the standard deviation of the weights to 0.02 as is tradition
        nn.init.normal_(self.qkv.weight, std=0.02)

    def forward(self, x, t):
        # Calculate the modulations
        mods = self.adaLN_mlp(t).unsqueeze(1).chunk(2, dim=2)
        # Forward, with modulations
        x = modulate(self.input_norm(x), mods[0], mods[1])
        # Calculate the query, key, and values all in one go
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = self.q_norm(q)
        k = self.k_norm(k)
        return q, k, v


class SelfAttention(nn.Module):
    """Standard self attention layer that supports masking."""

    def __init__(self, num_features, num_heads):
        super().__init__()
        self.num_features = num_features
        self.num_heads = num_heads

    def forward(self, q, k, v, mask=None):
        # Get the shape of the inputs
        B, T, C = v.size()
        # Reshape the query, key, and values for multi-head attention
        q = q.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)
        k = k.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)
        v = v.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)
        # Native torch attention
        attention_out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)  # (B, H, T, C/H)
        # Swap the sequence length and the head dimension back and get rid of num_heads.
        attention_out = attention_out.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, C)
        return attention_out


class PostAttentionBlock(nn.Module):
    """Block to postprocess V after attention."""

    def __init__(self, num_features, expansion_factor=4):
        super().__init__()
        self.num_features = num_features
        self.expansion_factor = expansion_factor
        # AdaLN MLP for post-attention. Initialized to zero so modulation acts as identity at initialization.
        self.adaLN_mlp_linear = nn.Linear(self.num_features, 4 * self.num_features, bias=True)
        nn.init.zeros_(self.adaLN_mlp_linear.weight)
        nn.init.zeros_(self.adaLN_mlp_linear.bias)
        self.adaLN_mlp = nn.Sequential(nn.SiLU(), self.adaLN_mlp_linear)
        # Linear layer to process v
        self.linear_v = nn.Linear(self.num_features, self.num_features)
        # Layernorm for the output
        self.output_norm = nn.LayerNorm(self.num_features, elementwise_affine=True, eps=1e-6)
        # Transformer style MLP layers
        self.linear_1 = nn.Linear(self.num_features, self.expansion_factor * self.num_features)
        self.nonlinearity = nn.GELU(approximate='tanh')
        self.linear_2 = nn.Linear(self.expansion_factor * self.num_features, self.num_features)
        # Initialize all biases to zero
        nn.init.zeros_(self.linear_1.bias)
        nn.init.zeros_(self.linear_2.bias)
        # Output MLP
        self.output_mlp = nn.Sequential(self.linear_1, self.nonlinearity, self.linear_2)

    def forward(self, v, x, t):
        """Forward takes v from self attention and the original sequence x with scalar conditioning t."""
        # Calculate the modulations
        mods = self.adaLN_mlp(t).unsqueeze(1).chunk(4, dim=2)
        # Postprocess v with linear + gating modulation
        y = mods[0] * self.linear_v(v)
        y = x + y
        # Adaptive layernorm
        y = modulate(self.output_norm(y), mods[1], mods[2])
        # Output MLP
        y = self.output_mlp(y)
        # Gating modulation for the output
        y = mods[3] * y
        y = x + y
        return y


class MMDiTBlock(nn.Module):
    """Transformer block that supports masking, multimodal attention, and adaptive norms."""

    def __init__(self, num_features, num_heads, expansion_factor=4, is_last=False):
        super().__init__()
        self.num_features = num_features
        self.num_heads = num_heads
        self.expansion_factor = expansion_factor
        self.is_last = is_last
        # Pre-attention blocks for two modalities
        self.pre_attention_block_1 = PreAttentionBlock(self.num_features)
        self.pre_attention_block_2 = PreAttentionBlock(self.num_features)
        # Self-attention
        self.attention = SelfAttention(self.num_features, self.num_heads)
        # Post-attention blocks for two modalities
        self.post_attention_block_1 = PostAttentionBlock(self.num_features, self.expansion_factor)
        if not self.is_last:
            self.post_attention_block_2 = PostAttentionBlock(self.num_features, self.expansion_factor)

    def forward(self, x1, x2, t, mask=None):
        # Pre-attention for the two modalities
        q1, k1, v1 = self.pre_attention_block_1(x1, t)
        q2, k2, v2 = self.pre_attention_block_2(x2, t)
        # Concat q, k, v along the sequence dimension
        q = torch.cat([q1, q2], dim=1)
        k = torch.cat([k1, k2], dim=1)
        v = torch.cat([v1, v2], dim=1)
        # Self-attention
        v = self.attention(q, k, v, mask=mask)
        # Split the attention output back into the two modalities
        seq_len_1, seq_len_2 = x1.size(1), x2.size(1)
        y1, y2 = v.split([seq_len_1, seq_len_2], dim=1)
        # Post-attention for the two modalities
        y1 = self.post_attention_block_1(y1, x1, t)
        if not self.is_last:
            y2 = self.post_attention_block_2(y2, x2, t)
        return y1, y2


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
            MMDiTBlock(self.num_features, self.num_heads, expansion_factor=self.expansion_factor)
            for _ in range(self.num_layers - 1)
        ])
        # Turn off post attn layers for conditioning sequence in final block
        self.transformer_blocks.append(
            MMDiTBlock(self.num_features, self.num_heads, expansion_factor=self.expansion_factor, is_last=True))
        # Output projection layer
        self.final_norm = nn.LayerNorm(self.num_features, elementwise_affine=True, eps=1e-6)
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

    def fsdp_wrap_fn(self, module):
        if isinstance(module, MMDiTBlock):
            return True
        return False

    def activation_checkpointing_fn(self, module):
        if isinstance(module, MMDiTBlock):
            return True
        return False

    def forward(self,
                x,
                input_coords,
                t,
                conditioning,
                conditioning_coords,
                input_mask=None,
                conditioning_mask=None,
                constant_conditioning=None):
        # Embed the timestep
        t = timestep_embedding(t, self.num_features)
        # Optionally add constant conditioning
        if constant_conditioning is not None:
            t = t + constant_conditioning
        # Embed the input
        y = self.input_embedding(x)  # (B, T1, C)
        # Get the input position embeddings and add them to the input
        y_position_embeddings = get_multidimensional_position_embeddings(self.input_position_embedding, input_coords)
        y_position_embeddings = y_position_embeddings.sum(dim=-1)  # (B, T1, C)
        y = y + y_position_embeddings  # (B, T1, C)
        if input_mask is None:
            mask = torch.ones(x.shape[0], x.shape[1], device=x.device)
        else:
            mask = input_mask

        # Embed the conditioning
        c = self.conditioning_embedding(conditioning)  # (B, T2, C)
        # Get the conditioning position embeddings and add them to the conditioning
        c_position_embeddings = get_multidimensional_position_embeddings(self.conditioning_position_embedding,
                                                                         conditioning_coords)
        c_position_embeddings = c_position_embeddings.sum(dim=-1)  # (B, T2, C)
        c = c + c_position_embeddings  # (B, T2, C)
        # Concatenate the masks
        if conditioning_mask is None:
            conditioning_mask = torch.ones(conditioning.shape[0], conditioning.shape[1], device=conditioning.device)
        mask = torch.cat([mask, conditioning_mask], dim=1)  # (B, T1 + T2)

        # Expand the mask to the right shape
        mask = mask.bool()
        mask = mask.unsqueeze(-1) & mask.unsqueeze(1)  # (B, T1 + T2, T1 + T2)
        identity = torch.eye(mask.shape[1], device=mask.device, dtype=mask.dtype).unsqueeze(0)
        mask = mask | identity
        mask = mask.unsqueeze(1)  # (B, 1, T1 + T2, T1 + T2)

        # Pass through the transformer blocks
        for block in self.transformer_blocks:
            y, c = block(y, c, t, mask=mask)
        # Pass through the output layers to get the right number of elements
        mods = self.adaLN_mlp(t).unsqueeze(1).chunk(2, dim=2)
        y = modulate(self.final_norm(y), mods[0], mods[1])
        y = self.final_linear(y)
        return y


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
        use_pooled_embedding: bool = False,
    ):
        super().__init__()
        self.model = model
        self.autoencoder = autoencoder
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.noise_scheduler = noise_scheduler
        self.inference_scheduler = inference_noise_scheduler
        self.prediction_type = prediction_type.lower()
        if self.prediction_type not in ['epsilon', 'sample', 'v_prediction']:
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
        self.use_pooled_embedding = use_pooled_embedding

        # Projection layer for the pooled text embeddings
        if self.use_pooled_embedding:
            self.pooled_projection_layer = nn.Linear(self.model.conditioning_features, self.model.num_features)

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

    def flops_per_batch(self, batch):
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
            # Ensure text embeddings are not longer than the model can handle
            if text_embeddings.shape[1] > self.model.conditioning_max_sequence_length:
                text_embeddings = text_embeddings[:, :self.model.conditioning_max_sequence_length]
                caption_mask = caption_mask[:, :self.model.conditioning_max_sequence_length]

        # Optionally use pooled embeddings
        if self.use_pooled_embedding:
            pooled_text_embeddings = text_encoder_out[1]
            pooled_text_embeddings = self.pooled_projection_layer(pooled_text_embeddings)
        else:
            pooled_text_embeddings = None

        # Make the text embedding coords
        text_embeddings_coords = torch.arange(text_embeddings.shape[1], device=text_embeddings.device)
        text_embeddings_coords = text_embeddings_coords.unsqueeze(0).expand(text_embeddings.shape[0], -1).unsqueeze(-1)
        # Project the text embeddings
        # Zero dropped captions if needed
        if 'drop_caption_mask' in batch.keys():
            text_embeddings *= batch['drop_caption_mask'].view(-1, 1, 1)
            if self.use_pooled_embedding:
                pooled_text_embeddings *= batch['drop_caption_mask'].view(-1, 1)
        # Scale and patchify the latents
        latents = (latents - self.latent_mean) / self.latent_std
        latent_patches, latent_coords = patchify(latents, self.patch_size)
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

    @torch.no_grad()
    def generate(self,
                 prompt: Optional[list] = None,
                 negative_prompt: Optional[list] = None,
                 height: int = 256,
                 width: int = 256,
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
        latent_patches, latent_coords = patchify(latents, self.patch_size)

        # Set up for CFG
        text_embeddings = torch.cat([text_embeddings, negative_text_embeddings], dim=0)
        text_embeddings_coords = torch.cat([text_embeddings_coords, negative_text_embeddings_coords], dim=0)
        text_embeddings_mask = torch.cat([prompt_mask, negative_prompt_mask], dim=0)
        latent_coords_input = torch.cat([latent_coords, latent_coords], dim=0)

        # Prep for reverse process
        self.inference_scheduler.set_timesteps(num_inference_steps)
        # scale the initial noise by the standard deviation required by the scheduler
        latent_patches = latent_patches * self.inference_scheduler.init_noise_sigma

        # Ensure text embeddings, mask, and coords are not longer than the model can handle
        if text_embeddings.shape[1] > self.model.conditioning_max_sequence_length:
            text_embeddings = text_embeddings[:, :self.model.conditioning_max_sequence_length]
            text_embeddings_coords = text_embeddings_coords[:, :self.model.conditioning_max_sequence_length]
            text_embeddings_mask = text_embeddings_mask[:, :self.model.conditioning_max_sequence_length]

        # backward diffusion process
        for t in tqdm(self.inference_scheduler.timesteps, disable=not progress_bar):
            latent_patches_input = torch.cat([latent_patches, latent_patches], dim=0)
            latent_patches_input = self.inference_scheduler.scale_model_input(latent_patches_input, t)
            # Get the model prediction
            model_out = self.model(latent_patches_input,
                                   latent_coords_input,
                                   t.unsqueeze(0).to(device),
                                   conditioning=text_embeddings,
                                   conditioning_coords=text_embeddings_coords,
                                   input_mask=None,
                                   conditioning_mask=text_embeddings_mask)
            # Do CFG
            pred_cond, pred_uncond = model_out.chunk(2, dim=0)
            pred = pred_uncond + guidance_scale * (pred_cond - pred_uncond)
            # Update the inputs
            latent_patches = self.inference_scheduler.step(pred, t, latent_patches, generator=rng_generator).prev_sample
        # Unpatchify the latents
        latents = [
            unpatchify(latent_patches[i], latent_coords[i], self.patch_size) for i in range(latent_patches.shape[0])
        ]
        latents = torch.stack(latents)
        # Scale the latents back to the original scale
        latents = latents * self.latent_std + self.latent_mean
        # Decode the latents
        image = self.autoencoder.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        return image.detach()  # (batch*num_images_per_prompt, channel, h, w)
