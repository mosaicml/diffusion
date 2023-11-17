# Copyright 2022 MosaicML Diffusion authors
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from diffusers import AutoencoderKL
from transformers import PretrainedConfig

from diffusion.models.autoencoder import AutoEncoder, Decoder, Encoder, NlayerDiscriminator
from diffusion.models.layers import AttentionLayer, Downsample, GradientScalingLayer, ResNetBlock, Upsample


@pytest.mark.parametrize('input_channels', [32])
@pytest.mark.parametrize('output_channels', [32, 64])
@pytest.mark.parametrize('use_conv_shortcut', [True, False])
@pytest.mark.parametrize('dropout_probability', [0.0, 0.1])
def test_resnet_block(input_channels, output_channels, use_conv_shortcut, dropout_probability):
    block = ResNetBlock(input_channels=input_channels,
                        output_channels=output_channels,
                        use_conv_shortcut=use_conv_shortcut,
                        dropout_probability=dropout_probability,
                        zero_init_last=True)
    x = torch.randn(1, input_channels, 5, 5)
    y = block(x)
    assert y.shape == (1, output_channels, 5, 5), f'{y.shape}'
    if input_channels == output_channels:
        # Model should be exactly identity here
        torch.testing.assert_close(x, y)


@pytest.mark.parametrize('input_channels', [32, 64])
@pytest.mark.parametrize('size', [6, 7])
def test_attention(input_channels, size):
    attention = AttentionLayer(input_channels=input_channels)
    x = torch.randn(1, input_channels, size, size)
    y = attention(x)
    assert y.shape == x.shape


@pytest.mark.parametrize('input_channels', [3, 4])
@pytest.mark.parametrize('size', [6, 7])
@pytest.mark.parametrize('resample_with_conv', [True, False])
def test_downsample(input_channels, resample_with_conv, size):
    downsample = Downsample(input_channels=input_channels, resample_with_conv=resample_with_conv)
    x = torch.randn(1, input_channels, size, size)
    y = downsample(x)
    assert y.shape == (1, input_channels, size // 2, size // 2)


@pytest.mark.parametrize('input_channels', [3, 4])
@pytest.mark.parametrize('size', [6, 7])
@pytest.mark.parametrize('resample_with_conv', [True, False])
def test_upsample(input_channels, resample_with_conv, size):
    upsample = Upsample(input_channels=input_channels, resample_with_conv=resample_with_conv)
    x = torch.randn(1, input_channels, size, size)
    y = upsample(x)
    assert y.shape == (1, input_channels, size * 2, size * 2)


def test_encoder():
    encoder = Encoder(input_channels=3,
                      hidden_channels=32,
                      latent_channels=4,
                      double_latent_channels=True,
                      channel_multipliers=(1, 2, 4, 8),
                      num_residual_blocks=4,
                      use_conv_shortcut=False,
                      dropout_probability=0.0,
                      resample_with_conv=True)
    x = torch.randn(1, 3, 16, 16)
    y = encoder(x)
    assert y.shape == (1, 8, 2, 2)


def test_decoder():
    decoder = Decoder(output_channels=3,
                      hidden_channels=32,
                      latent_channels=4,
                      channel_multipliers=(1, 2, 4, 8),
                      num_residual_blocks=4,
                      use_conv_shortcut=False,
                      dropout_probability=0.0,
                      resample_with_conv=True)
    x = torch.randn(1, 4, 2, 2)
    y = decoder(x)
    assert y.shape == (1, 3, 16, 16)


@pytest.mark.parametrize('height', [32])
@pytest.mark.parametrize('width', [32])
@pytest.mark.parametrize('num_layers', [3])
def test_discriminator(height, width, num_layers):
    discriminator = NlayerDiscriminator(input_channels=3, num_filters=16, num_layers=num_layers)
    x = torch.randn(1, 3, height, width)
    y = discriminator(x)
    downsample_factor = 2**(num_layers + 1)
    assert y.shape == (1, 1, height // downsample_factor,
                       width // downsample_factor), f'{y.shape}, {height}, {width}, {downsample_factor}'


@pytest.mark.parametrize('scale', [1, -1, -0.5, 0.5])
def test_attention_gradient_scaling(scale):
    # Make the layer
    layer = GradientScalingLayer()
    # Need to register the backward hook to scale the gradient
    layer.register_full_backward_hook(layer.backward_hook)
    # Input vector
    x = torch.randn(1, 3, requires_grad=True)
    # Forward pass
    y = layer(x)
    # Check that the output is the same as the input
    torch.testing.assert_close(x, y)
    # Set the scale
    layer.set_scale(scale)
    # Take the gradient
    grad = torch.autograd.grad(y.sum(), x, retain_graph=True)[0]
    # Check that the gradient is 1 times the scale
    torch.testing.assert_close(grad, scale * torch.ones_like(grad))


def test_autoencoder():
    # Get the HF autoencoder
    model_name = 'stabilityai/stable-diffusion-2-base'
    config = PretrainedConfig.get_config_dict(model_name, subfolder='vae')
    hf_autoencoder = AutoencoderKL(**config[0])
    # Make the corresponding autoencoder from this codebase
    autoencoder = AutoEncoder(input_channels=3,
                              output_channels=3,
                              hidden_channels=128,
                              latent_channels=4,
                              double_latent_channels=True,
                              channel_multipliers=(1, 2, 4, 4),
                              num_residual_blocks=2,
                              use_conv_shortcut=False,
                              dropout_probability=0.0,
                              resample_with_conv=True)
    # Check that the number of parameters is the same
    hf_params = sum(p.numel() for p in hf_autoencoder.parameters() if p.requires_grad)
    params = sum(p.numel() for p in autoencoder.parameters() if p.requires_grad)
    assert hf_params == params, f'{hf_params} == {params}'
