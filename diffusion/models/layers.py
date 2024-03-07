# Copyright 2022 MosaicML Diffusion authors
# SPDX-License-Identifier: Apache-2.0

"""Helpful layers and functions for UNet and Autoencoder construction."""

from typing import Optional, TypeVar

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import xformers  # type: ignore
except:
    pass

_T = TypeVar('_T', bound=nn.Module)


def zero_module(module: _T) -> _T:
    """Zero out the parameters of a module and return it."""
    for p in module.parameters():
        p.detach().zero_()
    return module


class ClippedAttnProcessor2_0:
    """Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).

    Modified from https://github.com/huggingface/diffusers/blob/v0.21.0-release/src/diffusers/models/attention_processor.py#L977 to
    allow clipping QKV values.

    Args:
        clip_val (float, defaults to 6.0): Amount to clip query, key, and value by.
    """

    def __init__(self, clip_val=6.0):
        if not hasattr(F, 'scaled_dot_product_attention'):
            raise ImportError('AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.')
        self.clip_val = clip_val

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        scale: float = 1.0,
    ):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        else:
            channel, height, width = None, None, None

        batch_size, sequence_length, _ = (hidden_states.shape
                                          if encoder_hidden_states is None else encoder_hidden_states.shape)

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states, scale=scale)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states, scale=scale)
        value = attn.to_v(encoder_hidden_states, scale=scale)

        query = query.clamp(min=-self.clip_val, max=self.clip_val)
        key = key.clamp(min=-self.clip_val, max=self.clip_val)
        value = value.clamp(min=-self.clip_val, max=self.clip_val)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = F.scaled_dot_product_attention(query,
                                                       key,
                                                       value,
                                                       attn_mask=attention_mask,
                                                       dropout_p=0.0,
                                                       is_causal=False)

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states, scale=scale)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class ClippedXFormersAttnProcessor:
    """Processor for implementing memory efficient attention using xFormers.

    Modified from https://github.com/huggingface/diffusers/blob/v0.21.0-release/src/diffusers/models/attention_processor.py#L888 to
    allow clipping QKV values.

    Args:
        attention_op (`Callable`, *optional*, defaults to `None`):
            The base
            [operator](https://facebookresearch.github.io/xformers/components/ops.html#xformers.ops.AttentionOpBase) to
            use as the attention operator. It is recommended to set to `None`, and allow xFormers to choose the best
            operator.
        clip_val (float, defaults to 6.0): Amount to clip query, key, and value by.
    """

    def __init__(self, clip_val=6.0, attention_op=None):
        self.attention_op = attention_op
        self.clip_val = clip_val

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        scale: float = 1.0,
    ):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        else:
            channel, height, width = None, None, None

        batch_size, key_tokens, _ = (hidden_states.shape
                                     if encoder_hidden_states is None else encoder_hidden_states.shape)

        attention_mask = attn.prepare_attention_mask(attention_mask, key_tokens, batch_size)
        if attention_mask is not None:
            # expand our mask's singleton query_tokens dimension:
            #   [batch*heads,            1, key_tokens] ->
            #   [batch*heads, query_tokens, key_tokens]
            # so that it can be added as a bias onto the attention scores that xformers computes:
            #   [batch*heads, query_tokens, key_tokens]
            # we do this explicitly because xformers doesn't broadcast the singleton dimension for us.
            _, query_tokens, _ = hidden_states.shape
            attention_mask = attention_mask.expand(-1, query_tokens, -1)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states, scale=scale)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states, scale=scale)
        value = attn.to_v(encoder_hidden_states, scale=scale)

        query = query.clamp(min=-self.clip_val, max=self.clip_val)
        key = key.clamp(min=-self.clip_val, max=self.clip_val)
        value = value.clamp(min=-self.clip_val, max=self.clip_val)

        query = attn.head_to_batch_dim(query).contiguous()
        key = attn.head_to_batch_dim(key).contiguous()
        value = attn.head_to_batch_dim(value).contiguous()

        hidden_states = xformers.ops.memory_efficient_attention(query,
                                                                key,
                                                                value,
                                                                attn_bias=attention_mask,
                                                                op=self.attention_op,
                                                                scale=attn.scale)
        hidden_states = hidden_states.to(query.dtype)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states, scale=scale)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            assert channel
            assert height
            assert width
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class ResNetBlock(nn.Module):
    """Basic ResNet block.

    Args:
        input_channels (int): Number of input channels.
        output_channels (int): Number of output channels.
        use_conv_shortcut (bool): Whether to use a conv on the shortcut. Default: `False`.
        dropout (float): Dropout probability. Defaults to 0.0.
        zero_init_last (bool): Whether to initialize the last conv layer to zero. Default: `False`.
    """

    def __init__(
        self,
        input_channels: int,
        output_channels: Optional[int] = None,
        use_conv_shortcut: bool = False,
        dropout_probability: float = 0.0,
        zero_init_last: bool = False,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels if output_channels is not None else input_channels
        self.use_conv_shortcut = use_conv_shortcut
        self.dropout_probability = dropout_probability
        self.zero_init_last = zero_init_last

        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=self.input_channels, eps=1e-6, affine=True)
        self.conv1 = nn.Conv2d(self.input_channels, self.output_channels, kernel_size=3, stride=1, padding=1)
        nn.init.kaiming_normal_(self.conv1.weight, nonlinearity='linear')
        # Output layer is immediately after a silu. Need to account for that in init.
        self.conv1.weight.data *= 1.6761
        self.norm2 = nn.GroupNorm(num_groups=32, num_channels=self.output_channels, eps=1e-6, affine=True)
        self.dropout = nn.Dropout2d(p=self.dropout_probability)
        self.conv2 = nn.Conv2d(self.output_channels, self.output_channels, kernel_size=3, stride=1, padding=1)

        # Optionally use a conv on the shortcut, but only if the input and output channels are different
        if self.input_channels != self.output_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = nn.Conv2d(self.input_channels,
                                               self.output_channels,
                                               kernel_size=3,
                                               stride=1,
                                               padding=1)
            else:
                self.conv_shortcut = nn.Conv2d(self.input_channels,
                                               self.output_channels,
                                               kernel_size=1,
                                               stride=1,
                                               padding=0)
            nn.init.kaiming_normal_(self.conv_shortcut.weight, nonlinearity='linear')
        else:
            self.conv_shortcut = nn.Identity()

        # Init the final conv layer parameters to zero if desired. Otherwise, kaiming uniform
        if self.zero_init_last:
            self.residual_scale = 1.0
            self.conv2 = zero_module(self.conv2)
        else:
            self.residual_scale = 0.70711
            nn.init.kaiming_normal_(self.conv2.weight, nonlinearity='linear')
            # Output layer is immediately after a silu. Need to account for that in init.
            self.conv2.weight.data *= 1.6761 * self.residual_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward through the residual block."""
        shortcut = self.residual_scale * self.conv_shortcut(x)
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)
        return h + shortcut


class AttentionLayer(nn.Module):
    """Basic single headed attention layer for use on tensors with HW dimensions.

    Args:
        input_channels (int): Number of input channels.
        dropout (float): Dropout probability. Defaults to 0.0.
    """

    def __init__(self, input_channels: int, dropout_probability: float = 0.0):
        super().__init__()
        self.input_channels = input_channels
        self.dropout_probability = dropout_probability
        # Normalization layer. Here we're using groupnorm to be consistent with the original implementation.
        self.norm = nn.GroupNorm(num_groups=32, num_channels=self.input_channels, eps=1e-6, affine=True)
        # Conv layer to transform the input into q, k, and v
        self.qkv_conv = nn.Conv2d(self.input_channels, 3 * self.input_channels, kernel_size=1, stride=1, padding=0)
        # Init the qkv conv weights
        nn.init.kaiming_normal_(self.qkv_conv.weight, nonlinearity='linear')
        # Conv layer to project to the output.
        self.proj_conv = nn.Conv2d(self.input_channels, self.input_channels, kernel_size=1, stride=1, padding=0)
        nn.init.kaiming_normal_(self.proj_conv.weight, nonlinearity='linear')

    def _reshape_for_attention(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape the input tensor for attention."""
        # x is (B, C, H, W), need it to be (B, H*W, C) for attention
        x = x.permute(0, 2, 3, 1).reshape(x.shape[0], -1, x.shape[1]).contiguous()
        return x

    def _reshape_from_attention(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """Reshape the input tensor from attention."""
        # x is (B, H*W, C), need it to be (B, C, H, W) for conv
        x = x.reshape(x.shape[0], H, W, -1).permute(0, 3, 1, 2).contiguous()
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward through the attention layer."""
        # Need to remember H, W to get back to it
        H, W = x.shape[2:]
        h = self.norm(x)
        # Get q, k, and v
        qkv = self.qkv_conv(h)
        qkv = self._reshape_for_attention(qkv)
        q, k, v = torch.split(qkv, self.input_channels, dim=2)
        # Use torch's built in attention function
        h = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout_probability)
        # Reshape back into an image style tensor
        h = self._reshape_from_attention(h, H, W)
        # Project to the output
        h = self.proj_conv(h)
        return x + h


class Downsample(nn.Module):
    """Downsampling layer that downsamples by a factor of 2.

    Args:
        input_channels (int): Number of input channels.
        resample_with_conv (bool): Whether to use a conv for downsampling.
    """

    def __init__(self, input_channels: int, resample_with_conv: bool):
        super().__init__()
        self.input_channels = input_channels
        self.resample_with_conv = resample_with_conv
        if self.resample_with_conv:
            self.conv = nn.Conv2d(self.input_channels, self.input_channels, kernel_size=3, stride=2, padding=0)
            nn.init.kaiming_normal_(self.conv.weight, nonlinearity='linear')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.resample_with_conv:
            # Need to do asymmetric padding to ensure the correct pixels are used in the downsampling conv
            # and ensure the correct output size is generated for even sizes.
            x = F.pad(x, (0, 1, 0, 1), mode='constant', value=0)
            x = self.conv(x)
        else:
            x = F.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class Upsample(nn.Module):
    """Upsampling layer that upsamples by a factor of 2.

    Args:
        input_channels (int): Number of input channels.
        resample_with_conv (bool): Whether to use a conv for upsampling.
    """

    def __init__(self, input_channels: int, resample_with_conv: bool):
        super().__init__()
        self.input_channels = input_channels
        self.resample_with_conv = resample_with_conv
        if self.resample_with_conv:
            self.conv = nn.Conv2d(self.input_channels, self.input_channels, kernel_size=3, stride=1, padding=1)
            nn.init.kaiming_normal_(self.conv.weight, nonlinearity='linear')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode='nearest', antialias=False)
        if self.resample_with_conv:
            x = self.conv(x)
        return x


class GradientScalingLayer(nn.Module):
    """Layer that scales the gradient by a multiplicative constant.

    By default, this constant is 1.0, so this layer acts as an identity function.

    To use, one must also register the backward hook:
    scaling_layer = GradientScalingLayer()
    scaling_layer.register_full_backward_hook(scaling_layer.backward_hook)

    And then set the scale via
    scaling_layer.set_scale(scale)
    """

    def __init__(self):
        super().__init__()
        self.scale: float = 1.0

    def set_scale(self, scale: float):
        self.scale = scale

    def forward(self, x):
        return x

    def backward_hook(self, module, grad_input, grad_output):
        return (self.scale * grad_input[0],)
