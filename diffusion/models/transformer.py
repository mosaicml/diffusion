# Copyright 2022 MosaicML Diffusion authors
# SPDX-License-Identifier: Apache-2.0

"""Diffusion Transformer model."""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Modulate the input with the shift and scale."""
    return x * (1.0 + scale) + shift


def get_multidimensional_position_embeddings(position_embeddings: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
    """Extracts position embeddings for a multidimensional sequence given by coordinates.

    Position embeddings are shape (D, T, F). Coords are shape (B, S, D). Position embeddings should be
    interpreted as D dimensional embeddings with F features each for a maximum of T timesteps.
    Coordinates or `coords` is a batch of size B of sequences of length S with D dimensional integer
    coordinates. For example, if D=2, then each of the B, S elements of the sequence would have a 2D
    X,Y coordinate.

    Args:
        position_embeddings (torch.Tensor): Position embeddings of shape (D, T, F).
        coords (torch.Tensor): Coordinates of shape (B, S, D).

    Returns:
        torch.Tensor: Sequenced embeddings of shape (B, S, F, D)
    """
    B = coords.shape[0]
    F = position_embeddings.shape[-1]  # Position embedding dimensions
    coords = coords.permute(2, 0, 1)  # (D, B, S)
    position_embeddings = position_embeddings.unsqueeze(1).expand(-1, B, -1, -1)  # (D, B, T, F)
    # Prepare indices for torch.gather
    coords = coords.unsqueeze(-1).expand(-1, -1, -1, F)  # (D, B, S, F)
    # Use torch.gather to collect embeddings
    embeddings = torch.gather(position_embeddings, 2, coords)  # (D, B, S, F)
    # Rearrange embeddings to the desired output shape
    embeddings = embeddings.permute(1, 2, 3, 0)  # (B, S, F, D)
    return embeddings


class MuInputLinear(nn.Module):
    """Linear input layer with the mu parameterization.

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        bias (bool): Whether or not to use a bias. Default: `True`.
    """

    def __init__(self, in_features, out_features, bias=True):
        super(MuInputLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.mu_input_linear = nn.Linear(in_features, out_features, bias)
        self.mu_init()

    def mu_init(self):
        """Initializes a linear layer according to mu-parameterizaion."""
        scale = 1 / math.sqrt(self.in_features)
        if self.mu_input_linear.bias is not None:
            nn.init.zeros_(self.mu_input_linear.bias)
        nn.init.normal_(self.mu_input_linear.weight, std=scale)

    def forward(self, x):
        return self.mu_input_linear(x)


class MuLinear(nn.Module):
    """Linear layer with the mu parameterization.

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        bias (bool): Whether or not to use a bias. Default: `True`.
    """

    def __init__(self, in_features, out_features, bias=True):
        super(MuLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.mu_linear = nn.Linear(in_features, out_features, bias)
        self.mu_init()

    def mu_init(self):
        """Initializes a linear layer according to mu-parameterizaion."""
        scale = 1 / math.sqrt(self.in_features)
        if self.mu_linear.bias is not None:
            nn.init.zeros_(self.mu_linear.bias)
        nn.init.normal_(self.mu_linear.weight, std=scale)

    def forward(self, x):
        return self.mu_linear(x)


class MuOutputLinear(nn.Module):
    """Linear outpus layer with the mu parameterization.

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        bias (bool): Whether or not to use a bias. Default: `True`.
    """

    def __init__(self, in_features, out_features, bias=True):
        super(MuOutputLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.mu_output_linear = nn.Linear(in_features, out_features, bias)
        self.mu_init()

    def mu_init(self):
        """Initializes a linear layer according to mu-parameterizaion."""
        scale = 1 / self.in_features
        if self.mu_output_linear.bias is not None:
            nn.init.zeros_(self.mu_output_linear.bias)
        nn.init.normal_(self.mu_output_linear.weight, std=scale)

    def rescale_init(self, scale):
        rescale = math.sqrt(1 / (self.in_features * scale))
        nn.init.normal_(self.mu_output_linear.weight, std=rescale)

    def forward(self, x):
        return self.mu_output_linear(x)


class FP32LayerNorm(nn.Module):
    """LayerNorm in FP32.

    Args:
        normalized_shape (int): input shape from an expected input of size (..., normalized_shape)
        eps (float): a value added to the denominator for numerical stability. Default: `1e-5`
        elementwise_affine (bool): a boolean value that when set to True, this module has learnable
                            per-element affine parameters initialized to ones (for weights)
                            and zeros (for biases). Default: `True`.
    """

    def __init__(self, normalized_shape: int, eps: float = 1e-5, elementwise_affine: bool = True):
        super().__init__()
        self.layer_norm = nn.LayerNorm(normalized_shape, eps, elementwise_affine)

    def forward(self, x):
        original_dtype = x.dtype
        x = x.to(dtype=torch.float32)
        x = self.layer_norm(x)
        return x.to(dtype=original_dtype)


class AdaptiveLayerNorm(nn.Module):
    """Adaptive LayerNorm.

    Scales and shifts the output of a LayerNorm using an MLP conditioned on a scalar.

    Args:
        num_features (int): Number of input features.
    """

    def __init__(self, num_features: int):
        super().__init__()
        self.num_features = num_features
        # MLP for computing modulations.
        # Initialized to zero so modulation acts as identity at initialization.
        self.adaLN_mlp_linear_shift = MuLinear(self.num_features, self.num_features, bias=True)
        self.adaLN_mlp_linear_scale = MuLinear(self.num_features, self.num_features, bias=True)
        nn.init.zeros_(self.adaLN_mlp_linear_shift.mu_linear.weight)
        nn.init.zeros_(self.adaLN_mlp_linear_scale.mu_linear.weight)
        nn.init.zeros_(self.adaLN_mlp_linear_shift.mu_linear.bias)
        nn.init.zeros_(self.adaLN_mlp_linear_scale.mu_linear.bias)
        self.adaLN_mlp_shift = nn.Sequential(nn.SiLU(), self.adaLN_mlp_linear_shift)
        self.adaLN_mlp_scale = nn.Sequential(nn.SiLU(), self.adaLN_mlp_linear_scale)
        # LayerNorm
        self.layernorm = FP32LayerNorm(self.num_features, elementwise_affine=False, eps=1e-6)

    @torch.compile()
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # Calculate the modulations
        shift = self.adaLN_mlp_linear_shift(t)
        scale = self.adaLN_mlp_linear_scale(t)
        # Apply the modulations
        return modulate(self.layernorm(x), shift, scale)


class ModulationLayer(nn.Module):
    """Modulation layer.

    Scales the input by a factor determined by a scalar input.

    Args:
        num_features (int): Number of input features.
    """

    def __init__(self, num_features: int):
        super().__init__()
        self.num_features = num_features
        # MLP for computing modulation.
        # Initialized to zero so modulation starts off at zero.
        self.adaLN_mlp_linear = MuLinear(self.num_features, self.num_features, bias=True)
        nn.init.zeros_(self.adaLN_mlp_linear.mu_linear.weight)
        nn.init.zeros_(self.adaLN_mlp_linear.mu_linear.bias)
        self.adaLN_mlp = nn.Sequential(nn.SiLU(), self.adaLN_mlp_linear)

    @torch.compile()
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # Calculate the modulations
        mods = self.adaLN_mlp(t)
        return x * mods


class ScalarEmbedding(nn.Module):
    """Embedding block for scalars.

    Embeds a scalar into a vector of size `num_features` using a sinusoidal embedding followed by an MLP.

    Args:
        num_features (int): The size of the output vector.
        sinusoidal_embedding_dim (int): The size of the intermediate sinusoidal embedding. Default: `256`.
        max_period (float): The maximum period of the sinusoidal embedding. Default: `10000.0`.

    Returns:
        torch.Tensor: The embedded scalar
    """

    def __init__(self, num_features: int, sinusoidal_embedding_dim: int = 256, max_period: float = 10000.0):
        super().__init__()
        self.num_features = num_features
        self.sinusoidal_embedding_dim = sinusoidal_embedding_dim
        self.max_period = max_period
        self.linear_1 = MuInputLinear(self.sinusoidal_embedding_dim, self.num_features)
        self.linear_2 = MuLinear(self.num_features, self.num_features)
        self.mlp = nn.Sequential(self.linear_1, nn.SiLU(), self.linear_2)
        # Make the freqs
        half_dim = self.sinusoidal_embedding_dim // 2
        self.freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half_dim, dtype=torch.float32) /
                               half_dim)

    def _apply(self, fn):
        super(ScalarEmbedding, self)._apply(fn)
        self.freqs = fn(self.freqs)
        return self

    def timestep_embedding(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Create sinusoidal timestep embeddings.

        Args:
            timesteps (torch.Tensor): The timesteps to embed.
        """
        args = timesteps[:, None].float() * self.freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return embedding

    @torch.compile()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sinusoidal_embedding = self.timestep_embedding(x)
        # Ensure embedding is the correct dtype
        sinusoidal_embedding = sinusoidal_embedding.to(next(self.parameters()).dtype)
        return self.mlp(sinusoidal_embedding)


class VectorEmbedding(nn.Module):
    """Embedding block for vectors.

    Embeds vectors via an MLP into a vector of size `num_features`.

    Args:
        input_features (int): The size of the input vector.
        num_features (int): The size of the output vector.
    """

    def __init__(self, input_features: int, num_features: int):
        super().__init__()
        self.input_features = input_features
        self.num_features = num_features
        self.linear_1 = MuInputLinear(self.input_features, self.num_features)
        self.linear_2 = MuLinear(self.num_features, self.num_features)
        self.mlp = nn.Sequential(self.linear_1, nn.SiLU(), self.linear_2)

    @torch.compile()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class PreAttentionBlock(nn.Module):
    """Block to compute QKV before attention.

    Includes QK layernorms and an adaptive layernorms.

    Args:
        num_features (int): Number of input features.
    """

    def __init__(self, num_features: int):
        super().__init__()
        self.num_features = num_features
        # Adaptive layernorm
        self.adaptive_layernorm = AdaptiveLayerNorm(self.num_features)
        # Linear layer to get q, k, and v
        self.q_proj = MuLinear(self.num_features, self.num_features, bias=False)
        self.k_proj = MuLinear(self.num_features, self.num_features, bias=False)
        self.v_proj = MuLinear(self.num_features, self.num_features, bias=False)
        # QK layernorms. Original MMDiT used RMSNorm here.
        self.q_norm = FP32LayerNorm(self.num_features, elementwise_affine=False, eps=1e-6)
        self.k_norm = FP32LayerNorm(self.num_features, elementwise_affine=False, eps=1e-6)

    @torch.compile()
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.adaptive_layernorm(x, t)
        # Calculate the query, key, and values all in one go
        q, k, v = self.q_norm(self.q_proj(x)), self.k_norm(self.k_proj(x)), self.v_proj(x)
        return q, k, v


class SelfAttention(nn.Module):
    """Multi-head self-attention layer with selectable attention implementations.

    Args:
        num_features (int): Number of input features.
        num_heads (int): Number of attention heads.
        attention_implementation (str): Attention implementation ('flash', 'mem_efficient', 'math'). If not specified, will let
            SDPA decide. Default: 'None'.
    """

    def __init__(self, num_features: int, num_heads: int, attention_implementation: Optional[str] = None):
        super().__init__()
        self.num_features = num_features
        self.num_heads = num_heads
        self.head_dim = num_features // num_heads
        self.attn_scale = 1 / self.head_dim
        assert self.num_features % self.num_heads == 0, 'num_features must be divisible by num_heads'
        if attention_implementation is not None:
            assert attention_implementation in ('flash', 'mem_efficient', 'math'), (
                "attention_implementation must be 'flash', 'mem_efficient', or 'math', or None")
        self.attention_implementation = attention_implementation
        self.sdp_backends = self._get_sdp_backends()

    def _get_sdp_backends(self):
        if self.attention_implementation == 'flash':
            backends = [SDPBackend.FLASH_ATTENTION]
        elif self.attention_implementation == 'mem_efficient':
            backends = [SDPBackend.EFFICIENT_ATTENTION]
        elif self.attention_implementation == 'math':
            backends = [SDPBackend.MATH]
        else:
            # Let SDPA take the wheel
            backends = [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]
        return backends

    @torch.compile()
    def forward(self,
                q: torch.Tensor,
                k: torch.Tensor,
                v: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, C = q.size()
        H = self.num_heads
        D = self.head_dim

        # Reshape q, k, v for multi-head attention
        q = q.view(B, T, H, D).transpose(1, 2)  # (B, H, T, D)
        k = k.view(B, T, H, D).transpose(1, 2)  # (B, H, T, D)
        v = v.view(B, T, H, D).transpose(1, 2)  # (B, H, T, D)

        # Attention with selectable implementation
        if self.attention_implementation is None:
            attention_out = F.scaled_dot_product_attention(q.float(),
                                                           k.float(),
                                                           v.float(),
                                                           attn_mask=mask,
                                                           scale=self.attn_scale)
        else:
            with sdpa_kernel(self.sdp_backends):
                attention_out = F.scaled_dot_product_attention(q.float(),
                                                               k.float(),
                                                               v.float(),
                                                               attn_mask=mask,
                                                               scale=self.attn_scale)
        attention_out = attention_out.to(dtype=v.dtype)

        # Reshape back to (B, T, C)
        attention_out = attention_out.transpose(1, 2).reshape(B, T, C)
        return attention_out


class PostAttentionBlock(nn.Module):
    """Block to postprocess v after attention.

    Includes adaptive layernorms.

    Args:
        num_features (int): Number of input features.
        expansion_factor (int): Expansion factor for the MLP. Default: `4`.
    """

    def __init__(self, num_features: int, expansion_factor: int = 4):
        super().__init__()
        self.num_features = num_features
        self.expansion_factor = expansion_factor
        # Input modulation
        self.modulate_v = ModulationLayer(self.num_features)
        # Linear layer to process v
        self.linear_v = MuLinear(self.num_features, self.num_features, bias=False)
        # Layernorm for the output
        self.output_norm = AdaptiveLayerNorm(self.num_features)
        # Transformer style MLP layers
        self.linear_1 = MuLinear(self.num_features, self.expansion_factor * self.num_features)
        self.nonlinearity = nn.GELU(approximate='tanh')
        self.linear_2 = MuLinear(self.expansion_factor * self.num_features, self.num_features)
        # Output MLP
        self.output_mlp = nn.Sequential(self.linear_1, self.nonlinearity, self.linear_2)
        # Output modulation
        self.modulate_output = ModulationLayer(self.num_features)

    @torch.compile()
    def forward(self, v: torch.Tensor, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Forward takes v from self attention and the original sequence x with scalar conditioning t."""
        # Postprocess v with linear + gating modulation
        y = self.modulate_v(self.linear_v(v), t)
        y = x + y
        # Adaptive layernorm
        y = self.output_norm(y, t)
        # Output MLP
        y = self.output_mlp(y)
        # Gating modulation for the output
        y = self.modulate_output(y, t)
        y = x + y
        return y


class DiTBlock(nn.Module):
    """Transformer block that supports masking, and adaptive norms.

    Args:
        num_features (int): Number of input features.
        num_heads (int): Number of attention heads.
        expansion_factor (int): Expansion factor for the MLP. Default: `4`.
        attention_implementation (str): Attention implementation ('flash', 'mem_efficient', 'math'). If not specified, will let
            SDPA decide. Default: 'None'.
    """

    def __init__(self,
                 num_features: int,
                 num_heads: int,
                 expansion_factor: int = 4,
                 attention_implementation: Optional[str] = None):
        super().__init__()
        self.num_features = num_features
        self.num_heads = num_heads
        self.expansion_factor = expansion_factor
        if attention_implementation is not None:
            assert attention_implementation in ('flash', 'mem_efficient', 'math'), (
                "attention_implementation must be 'flash', 'mem_efficient', or 'math', or None")
        self.attention_implementation = attention_implementation
        # Pre-attention block
        self.pre_attention_block = PreAttentionBlock(self.num_features)
        # Self-attention
        self.attention = SelfAttention(self.num_features,
                                       self.num_heads,
                                       attention_implementation=attention_implementation)
        # Post-attention block
        self.post_attention_block = PostAttentionBlock(self.num_features, self.expansion_factor)

    @torch.compile()
    def forward(self, x: torch.Tensor, t: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Pre-attention
        q, k, v = self.pre_attention_block(x, t)
        # Self-attention
        v = self.attention(q, k, v, mask=mask)
        # Post-attention
        v = self.post_attention_block(v, x, t)
        return v


class MMDiTBlock(nn.Module):
    """Transformer block that supports masking, multimodal attention, and adaptive norms.

    Can optionally be the last block in the network, in which case it does not apply post-attention layers to the
    conditioning sequence, as those params may not be used.

    Args:
        num_features (int): Number of input features.
        num_heads (int): Number of attention heads.
        expansion_factor (int): Expansion factor for the MLP. Default: `4`.
        is_last (bool): Whether this is the last block in the network. Default: `False`.
        attention_implementation (str): Attention implementation ('flash', 'mem_efficient', 'math'). If not specified, will let
            SDPA decide. Default: 'None'.
    """

    def __init__(self,
                 num_features: int,
                 num_heads: int,
                 expansion_factor: int = 4,
                 is_last: bool = False,
                 attention_implementation: Optional[str] = None):
        super().__init__()
        self.num_features = num_features
        self.num_heads = num_heads
        self.expansion_factor = expansion_factor
        self.is_last = is_last
        if attention_implementation is not None:
            assert attention_implementation in ('flash', 'mem_efficient', 'math'), (
                "attention_implementation must be 'flash', 'mem_efficient', or 'math', or None")
        self.attention_implementation = attention_implementation
        # Pre-attention blocks for two modalities
        self.pre_attention_block_1 = PreAttentionBlock(self.num_features)
        self.pre_attention_block_2 = PreAttentionBlock(self.num_features)
        # Self-attention
        self.attention = SelfAttention(self.num_features,
                                       self.num_heads,
                                       attention_implementation=self.attention_implementation)
        # Post-attention blocks for two modalities
        self.post_attention_block_1 = PostAttentionBlock(self.num_features, self.expansion_factor)
        if not self.is_last:
            self.post_attention_block_2 = PostAttentionBlock(self.num_features, self.expansion_factor)

    @torch.compile()
    def forward(self,
                x1: torch.Tensor,
                x2: torch.Tensor,
                t: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
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


class DiTGroup(nn.Module):
    """A group of DiT blocks, for convenience.

    Args:
        num_blocks (int): Number of blocks in the group
        num_features (int): Number of input features.
        num_heads (int): Number of attention heads.
        expansion_factor (int): Expansion factor for the MLP. Default: `4`.
        attention_implementation (str): Attention implementation ('flash', 'mem_efficient', 'math'). If not specified, will let
            SDPA decide. Default: 'None'.
    """

    def __init__(self,
                 num_blocks,
                 num_features,
                 num_heads,
                 expansion_factor,
                 attention_implementation: Optional[str] = None):
        super().__init__()
        self.blocks = nn.ModuleList([
            DiTBlock(num_features, num_heads, expansion_factor, attention_implementation=attention_implementation)
            for _ in range(num_blocks)
        ])

    @torch.compile()
    def forward(self, x: torch.Tensor, t: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        for block in self.blocks:
            x = block(x, t, mask=mask)
        return x


class MMDiTGroup(nn.Module):
    """A group of MMDiT blocks, for convenience.

    Args:
        num_blocks (int): Number of blocks in the group
        num_features (int): Number of input features.
        num_heads (int): Number of attention heads.
        expansion_factor (int): Expansion factor for the MLP. Default: `4`.
        is_last (bool): Whether this is the last block in the network. Default: `False`.
        attention_implementation (str): Attention implementation ('flash', 'mem_efficient', 'math'). If not specified, will let
            SDPA decide. Default: 'None'.
    """

    def __init__(self,
                 num_blocks,
                 num_features,
                 num_heads,
                 expansion_factor,
                 is_last=False,
                 attention_implementation: Optional[str] = None):
        super().__init__()
        self.blocks = nn.ModuleList([
            MMDiTBlock(num_features,
                       num_heads,
                       expansion_factor,
                       is_last=(is_last and i == num_blocks - 1),
                       attention_implementation=attention_implementation) for i in range(num_blocks)
        ])

    @torch.compile()
    def forward(self,
                x1: torch.Tensor,
                x2: torch.Tensor,
                t: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        for block in self.blocks:
            x1, x2 = block(x1, x2, t, mask=mask)
        return x1, x2


class DiffusionTransformer(nn.Module):
    """Transformer model for generic diffusion.

    Supports input and conditioning sequences with different lengths and dimensions.

    Args:
        num_features (int): Number of hidden features.
        num_heads (int): Number of attention heads.
        num_mmdit_layers (int): Number of MMDiT layers.
        num_dit_layers (int): Number of DiT layers. Default: `0`.
        attention_implementation (Optional[str]): Attention implementation ('flash', 'mem_efficient', 'math'). If not specified, will let
            SDPA decide. Default: 'None'.
        input_features (int): Number of features in the input sequence. Default: `192`.
        input_max_sequence_length (int): Maximum sequence length for the input sequence. Default: `1024`.
        input_dimension (int): Dimension of the input sequence. Default: `2`.
        conditioning_features (int): Number of features in the conditioning sequence. Default: `1024`.
        conditioning_max_sequence_length (int): Maximum sequence length for the conditioning sequence. Default: `77`.
        conditioning_dimension (int): Dimension of the conditioning sequence. Default: `1`.
        expansion_factor (int): Expansion factor for the MLPs. Default: `4`.
        num_register_tokens (int): Number of register tokens to use. Default: `0`.
        mmdit_block_group_size (int): Size of MMDiT block groups. Must be a divisor of num_mmdit_layers. Default: `1`.
        dit_block_group_size (int): Size of DiT block groups. Must be a divisor of num_dit_layers. Default: `1`.

    """

    def __init__(self,
                 num_features: int,
                 num_heads: int,
                 num_mmdit_layers: int,
                 num_dit_layers: int = 0,
                 attention_implementation: Optional[str] = None,
                 input_features: int = 192,
                 input_max_sequence_length: int = 1024,
                 input_dimension: int = 2,
                 conditioning_features: int = 1024,
                 conditioning_max_sequence_length: int = 77,
                 conditioning_dimension: int = 1,
                 expansion_factor: int = 4,
                 num_register_tokens: int = 0,
                 mmdit_block_group_size: int = 1,
                 dit_block_group_size: int = 1):
        super().__init__()
        # Params for the network architecture
        self.num_features = num_features
        self.num_heads = num_heads
        self.num_mmdit_layers = num_mmdit_layers
        self.num_dit_layers = num_dit_layers
        self.num_layers = self.num_mmdit_layers + self.num_dit_layers  # For convenience.
        self.expansion_factor = expansion_factor
        self.attention_implementation = attention_implementation
        self.mmdit_block_group_size = mmdit_block_group_size
        self.dit_block_group_size = dit_block_group_size
        # Params for input embeddings
        self.input_features = input_features
        self.input_dimension = input_dimension
        self.input_max_sequence_length = input_max_sequence_length
        # Params for conditioning embeddings
        self.conditioning_features = conditioning_features
        self.conditioning_dimension = conditioning_dimension
        self.conditioning_max_sequence_length = conditioning_max_sequence_length
        self.num_register_tokens = num_register_tokens
        # Embedding block for the timestep
        self.timestep_embedding = ScalarEmbedding(self.num_features, max_period=1 / (2 * math.pi))
        # Projection layer for the input sequence
        self.input_embedding = MuLinear(self.input_features, self.num_features)
        # Embedding layer for the input sequence
        input_position_embedding = torch.randn(self.input_dimension, self.input_max_sequence_length, self.num_features)
        input_position_embedding /= math.sqrt(self.num_features)
        self.input_position_embedding = torch.nn.Parameter(input_position_embedding, requires_grad=True)
        # Projection layer for the conditioning sequence
        self.conditioning_embedding = MuLinear(self.conditioning_features, self.num_features)
        # Embedding layer for the conditioning sequence
        conditioning_position_embedding = torch.randn(self.conditioning_dimension,
                                                      self.conditioning_max_sequence_length, self.num_features)
        conditioning_position_embedding /= math.sqrt(self.num_features)
        self.conditioning_position_embedding = torch.nn.Parameter(conditioning_position_embedding, requires_grad=True)
        # Optional register tokens
        if self.num_register_tokens > 0:
            register_tokens = torch.randn(1, self.num_register_tokens, self.num_features) / math.sqrt(self.num_features)
            self.register_tokens = torch.nn.Parameter(register_tokens, requires_grad=True)

        # MMDiT blocks:
        self.mmdit_groups = nn.ModuleList()
        assert self.num_mmdit_layers % self.mmdit_block_group_size == 0, 'num_mmdit_layers must be divisible by mmdit_block_group_size.'
        num_mmdit_groups = self.num_mmdit_layers // self.mmdit_block_group_size
        for _ in range(num_mmdit_groups - 1):
            self.mmdit_groups.append(
                MMDiTGroup(self.mmdit_block_group_size,
                           self.num_features,
                           self.num_heads,
                           expansion_factor=self.expansion_factor,
                           attention_implementation=self.attention_implementation))
        if num_mmdit_groups > 0:
            self.mmdit_groups.append(
                MMDiTGroup(self.mmdit_block_group_size,
                           self.num_features,
                           self.num_heads,
                           expansion_factor=self.expansion_factor,
                           is_last=True,
                           attention_implementation=self.attention_implementation))
        # DiT blocks
        self.dit_groups = nn.ModuleList()
        assert self.num_dit_layers % self.dit_block_group_size == 0, 'num_mmdit_layers must be divisible by mmdit_block_group_size.'
        num_dit_groups = self.num_dit_layers // self.dit_block_group_size
        for _ in range(num_dit_groups):
            self.dit_groups.append(
                DiTGroup(self.mmdit_block_group_size,
                         self.num_features,
                         self.num_heads,
                         expansion_factor=self.expansion_factor,
                         attention_implementation=self.attention_implementation))
        # Output projection layer
        self.final_norm = AdaptiveLayerNorm(self.num_features)
        self.final_linear = MuOutputLinear(self.num_features, self.input_features)

    def fsdp_wrap_fn(self, module: nn.Module) -> bool:
        if isinstance(module, (MMDiTGroup, DiTGroup)):
            return True
        return False

    def activation_checkpointing_fn(self, module: nn.Module) -> bool:
        if isinstance(module, (MMDiTGroup, DiTGroup)):
            return True
        return False

    def forward(self,
                x: torch.Tensor,
                input_coords: torch.Tensor,
                t: torch.Tensor,
                conditioning: torch.Tensor,
                conditioning_coords: torch.Tensor,
                constant_conditioning: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through the diffusion transformer.

        Args:
            x (torch.Tensor): The input sequence of shape (B, T1, C1).
            input_coords (torch.Tensor): The coordinates of the D dimensional input sequence of shape (B, T1, D).
            t (torch.Tensor): The scalar timesteps of shape (B, 1).
            conditioning (torch.Tensor): The conditioning sequence of shape (B, T2, C2).
            conditioning_coords (torch.Tensor): The coordinates of the D dimensional conditioning sequence of shape (B, T2, D).
            constant_conditioning (Optional[torch.Tensor]): Optional additional constant conditioning (B, num_features).

        Returns:
            torch.Tensor: The output sequence of shape (B, T1, C1).
        """
        # Embed the timestep
        t = self.timestep_embedding(t)
        # Optionally add constant conditioning. This assumes it has been embedded already.
        if constant_conditioning is not None:
            t = t + constant_conditioning
        # Unsqueeze for use in adaptive norm layers
        t = t.unsqueeze(1)
        # Embed the input
        y = self.input_embedding(x)  # (B, T1, C)
        # Get the input position embeddings and add them to the input
        y_position_embeddings = get_multidimensional_position_embeddings(self.input_position_embedding, input_coords)
        y_position_embeddings = y_position_embeddings.sum(dim=-1)  # (B, T1, C)
        y = y + y_position_embeddings  # (B, T1, C)
        # Embed the conditioning
        c = self.conditioning_embedding(conditioning)  # (B, T2, C)
        # Get the conditioning position embeddings and add them to the conditioning
        c_position_embeddings = get_multidimensional_position_embeddings(self.conditioning_position_embedding,
                                                                         conditioning_coords)
        c_position_embeddings = c_position_embeddings.sum(dim=-1)  # (B, T2, C)
        c = c + c_position_embeddings  # (B, T2, C)
        # Optionally add the register tokens
        if self.num_register_tokens > 0:
            repeated_register = self.register_tokens.repeat(c.shape[0], 1, 1)
            c = torch.cat([c, repeated_register], dim=1)

        # Pass through the MMDiT blocks
        for mmdit_group in self.mmdit_groups:
            y, c = mmdit_group(y, c, t, mask=None)
        # Pass through the DiT blocks
        if self.num_dit_layers > 0:
            # Initial concat since DiT does not separate modalities
            img_len = y.shape[1]
            y = torch.cat([y, c], dim=1)
            for dit_group in self.dit_groups:
                y = dit_group(y, t, mask=None)
            # Chop off the conditioning
            y = y[:, :img_len]
        # Pass through the output layers to get the right number of elements
        y = self.final_norm(y, t)
        y = self.final_linear(y)
        return y
