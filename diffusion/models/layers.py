# Copyright 2022 MosaicML Diffusion authors
# SPDX-License-Identifier: Apache-2.0

"""Helpful layers and functions for UNet construction."""

from typing import Optional

import torch
import torch.nn.functional as F
import xformers  # type: ignore


def zero_module(module):
    """Zero out the parameters of a module and return it."""
    for p in module.parameters():
        p.detach().zero_()
    return module


class ClampedAttnProcessor2_0:
    """Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).

    Modified from https://github.com/huggingface/diffusers/blob/v0.21.0-release/src/diffusers/models/attention_processor.py.
    """

    def __init__(self, clip_val=6.0):
        if not hasattr(F, 'scaled_dot_product_attention'):
            raise ImportError('AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.')
        self.clip_val = clip_val
        print('initializing ClampedAttnProcessor2_0 with value %f!' % clip_val)

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


class ClampedXFormersAttnProcessor:
    """Processor for implementing memory efficient attention using xFormers.

    Modified from https://github.com/huggingface/diffusers/blob/v0.21.0-release/src/diffusers/models/attention_processor.py.

    Args:
        attention_op (`Callable`, *optional*, defaults to `None`):
            The base
            [operator](https://facebookresearch.github.io/xformers/components/ops.html#xformers.ops.AttentionOpBase) to
            use as the attention operator. It is recommended to set to `None`, and allow xFormers to choose the best
            operator.
    """

    def __init__(self, clip_val=6.0, attention_op=None):
        self.attention_op = attention_op
        self.clip_val = clip_val

        print('initializing ClampedXFormersAttnProcessor with value %f - intentionally buggy version!' % clip_val)

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
        key = query.clamp(min=-self.clip_val, max=self.clip_val)  # key.clamp(min=-self.clip_val, max=self.clip_val)
        value = query.clamp(min=-self.clip_val, max=self.clip_val)  # value.clamp(min=-self.clip_val, max=self.clip_val)

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
