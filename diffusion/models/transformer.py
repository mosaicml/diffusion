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
                 conditioning_dimension: int = 2,
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


class ComposerDiffusionTransformer(ComposerModel):
    """Diffusion transformer ComposerModel.

    Args:
        model (DiffusionTransformer): Core diffusion model.
        prediction_type (str): The type of prediction to use. Currently `epsilon`, `v_prediction` are supported.
        T_max (int): The maximum number of timesteps. Default: 1000.
        input_key (str): The name of the inputs in the dataloader batch. Default: `input`.
        input_coords_key (str): The name of the input coordinates in the dataloader batch. Default: `input_coords`.
        input_mask_key (str): The name of the input mask in the dataloader batch. Default: `input_mask`.
        conditioning_key (str): The name of the conditioning info in the dataloader batch. Default: `conditioning`.
        conditioning_coords_key (str): The name of the conditioning coordinates in the dataloader batch. Default: `conditioning_coords`.
        conditioning_mask_key (str): The name of the conditioning mask in the dataloader batch. Default: `conditioning_mask`.
    """

    def __init__(
        self,
        model: DiffusionTransformer,
        prediction_type: str = 'epsilon',
        T_max: int = 1000,
        input_key: str = 'input',
        input_coords_key: str = 'input_coords',
        input_mask_key: str = 'input_mask',
        conditioning_key: str = 'conditioning',
        conditioning_coords_key: str = 'conditioning_coords',
        conditioning_mask_key: str = 'conditioning_mask',
    ):
        super().__init__()
        self.model = model
        self.model._fsdp_wrap = True

        # Diffusion parameters
        self.prediction_type = prediction_type.lower()
        if self.prediction_type not in ['epsilon', 'v_prediction']:
            raise ValueError(f'Unrecognized prediction type {self.prediction_type}')
        self.T_max = T_max

        # Set up input keys
        self.input_key = input_key
        self.input_coords_key = input_coords_key
        self.input_mask_key = input_mask_key
        # Set up conditioning keys
        self.conditioning_key = conditioning_key
        self.conditioning_coords_key = conditioning_coords_key
        self.conditioning_mask_key = conditioning_mask_key

        # Params for MFU computation, subtract off the embedding params
        self.n_params = sum(p.numel() for p in self.model.parameters())
        self.n_params -= self.model.input_position_embedding.numel()
        self.n_params -= self.model.conditioning_position_embedding.numel()

        # Set up metrics
        self.train_metrics = [MeanSquaredError()]
        self.val_metrics = [MeanSquaredError()]

        # Optional rng generator
        self.rng_generator: Optional[torch.Generator] = None

    def set_rng_generator(self, rng_generator: torch.Generator):
        """Sets the rng generator for the model."""
        self.rng_generator = rng_generator

    def flops_per_batch(self, batch):
        batch_size, input_seq_len = batch[self.input_key].shape[0:2]
        cond_seq_len = batch[self.conditioning_key].shape[1]
        seq_len = input_seq_len + cond_seq_len
        # Calulate forward flops excluding attention
        param_flops = 2 * self.n_params * batch_size * seq_len
        # Calculate flops for attention layers
        attention_flops = 4 * self.model.num_layers * seq_len**2 * self.model.num_features * batch_size
        return 3 * param_flops + 3 * attention_flops

    def diffusion_forward_process(self, inputs: torch.Tensor):
        """Diffusion forward process."""
        # Sample a timestep for every element in the batch
        timesteps = self.T_max * torch.rand(inputs.shape[0], device=inputs.device, generator=self.rng_generator)
        # Generate the noise, applied to the whole input sequence
        noise = torch.randn(*inputs.shape, device=inputs.device, generator=self.rng_generator)
        # Add the noise to the latents according to the natural schedule
        cos_t = torch.cos(timesteps * torch.pi / (2 * self.T_max)).view(-1, 1, 1)
        sin_t = torch.sin(timesteps * torch.pi / (2 * self.T_max)).view(-1, 1, 1)
        noised_inputs = cos_t * inputs + sin_t * noise
        if self.prediction_type == 'epsilon':
            # Get the (epsilon) targets
            targets = noise
        elif self.prediction_type == 'v_prediction':
            # Get the (velocity) targets
            targets = -sin_t * inputs + cos_t * noise
        else:
            raise ValueError(f'Unrecognized prediction type {self.prediction_type}')
        # TODO: Implement other prediction types
        return noised_inputs, targets, timesteps

    def forward(self, batch):
        # Get the inputs
        inputs = batch[self.input_key]
        inputs_coords = batch[self.input_coords_key]
        inputs_mask = batch[self.input_mask_key]
        # Get the conditioning
        conditioning = batch[self.conditioning_key]
        conditioning_coords = batch[self.conditioning_coords_key]
        conditioning_mask = batch[self.conditioning_mask_key]
        # Diffusion forward process
        noised_inputs, targets, timesteps = self.diffusion_forward_process(inputs)
        # Forward through the model
        model_out = self.model(noised_inputs,
                               inputs_coords,
                               timesteps,
                               conditioning=conditioning,
                               conditioning_coords=conditioning_coords,
                               input_mask=inputs_mask,
                               conditioning_mask=conditioning_mask)
        return {'predictions': model_out, 'targets': targets, 'timesteps': timesteps}

    def loss(self, outputs, batch):
        """MSE loss between outputs and targets."""
        losses = {}
        # Need to mask out elements in the loss that are not present in the input
        mask = batch[self.input_mask_key]  # (B, T1), 1 if included, 0 otherwise.
        loss = (outputs['predictions'] - outputs['targets'])**2  # (B, T1, C)
        loss = loss.mean(dim=2)  # (B, T1)
        losses['total'] = (loss * mask).sum() / mask.sum()
        return losses

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

    def update_inputs(self, inputs, predictions, t, delta_t):
        """Gets the input update."""
        angle = t * torch.pi / (2 * self.T_max)
        cos_t = torch.cos(angle).view(-1, 1, 1)
        sin_t = torch.sin(angle).view(-1, 1, 1)
        if self.prediction_type == 'epsilon':
            if angle == torch.pi / 2:
                # Optimal update here is to do nothing.
                pass
            elif torch.abs(torch.pi / 2 - angle) < 1e-4:
                # Need to avoid instability near t = T_max
                inputs = inputs - (predictions - sin_t * inputs)
            else:
                inputs = inputs - (predictions - sin_t * inputs) * delta_t / cos_t
        elif self.prediction_type == 'v_prediction':
            inputs = inputs - delta_t * predictions
        return inputs

    def generate(self,
                 input_coords: torch.Tensor,
                 input_mask: torch.Tensor,
                 conditioning: torch.Tensor,
                 conditioning_coords: torch.Tensor,
                 conditioning_mask: torch.Tensor,
                 guidance_scale: float = 7.0,
                 num_timesteps: int = 50,
                 progress_bar: bool = True,
                 seed: Optional[int] = None):
        """Generate from the model."""
        device = next(self.model.parameters()).device
        # Create rng for the generation
        rng_generator = torch.Generator(device=device)
        if seed:
            rng_generator = rng_generator.manual_seed(seed)
        # From the input coordinates, generate a noisy input sequence
        inputs = torch.randn(*input_coords.shape[:-1],
                             self.model.input_features,
                             device=device,
                             generator=rng_generator)
        # Set up for CFG
        input_coords = torch.cat([input_coords, input_coords], dim=0)
        input_mask = torch.cat([input_mask, input_mask], dim=0).to(device)
        conditioning = torch.cat([torch.zeros_like(conditioning), conditioning], dim=0).to(device)
        conditioning_coords = torch.cat([conditioning_coords, conditioning_coords], dim=0)
        conditioning_mask = torch.cat([torch.zeros_like(conditioning_mask), conditioning_mask], dim=0).to(device)
        # Make the timesteps
        timesteps = torch.linspace(self.T_max, 0, num_timesteps + 1, device=device)
        time_deltas = -torch.diff(timesteps) * (torch.pi / (2 * self.T_max))
        timesteps = timesteps[:-1]
        # backward diffusion process
        for i, t in enumerate(tqdm(timesteps, disable=not progress_bar)):
            # Expand t to the batch size
            t = t * torch.ones(inputs.shape[0], device=device)
            # Duplicate the inputs for CFG
            doubled_inputs = torch.cat([inputs, inputs], dim=0)
            # Get the model prediction
            model_out = self.model(doubled_inputs,
                                   input_coords,
                                   t,
                                   conditioning=conditioning,
                                   conditioning_coords=conditioning_coords,
                                   input_mask=input_mask,
                                   conditioning_mask=conditioning_mask)
            # Do CFG
            pred_uncond, pred_cond = model_out.chunk(2, dim=0)
            pred = pred_uncond + guidance_scale * (pred_cond - pred_uncond)
            # Update the inputs
            inputs = self.update_inputs(inputs, pred, t, time_deltas[i])
        return inputs
