# Copyright 2022 MosaicML Diffusion authors
# SPDX-License-Identifier: Apache-2.0

"""Autoencoder parts for training latent diffusion models."""

from typing import Optional, Tuple

import lpips
import torch
import torch.nn as nn
import torch.nn.functional as F
from composer.models import ComposerModel
from diffusers import AutoencoderKL
from torch.autograd import Function
from torchmetrics import MeanMetric, MeanSquaredError, Metric
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity


def zero_module(module: nn.Module) -> nn.Module:
    """Zero out the parameters of a module and return it."""
    for p in module.parameters():
        nn.init.zeros_(p)
    return module


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
        else:
            self.conv_shortcut = nn.Identity()

        # Init the final conv layer parameters to zero if desired. Otherwise, kaiming uniform
        if self.zero_init_last:
            self.conv2 = zero_module(self.conv2)
        else:
            nn.init.kaiming_normal_(self.conv2.weight, nonlinearity='linear')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward through the residual block."""
        shortcut = self.conv_shortcut(x)
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
        q, k, v = torch.split(qkv, self.input_channels, dim=1)
        # Use torch's built in attention function
        q, k, v = self._reshape_for_attention(q), self._reshape_for_attention(k), self._reshape_for_attention(v)
        h = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout_probability)
        # Reshape back into an image style tensor
        h = self._reshape_from_attention(h, H, W)
        # Project to the output
        h = self.proj_conv(h)
        return h


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
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        if self.resample_with_conv:
            x = self.conv(x)
        return x


class Encoder(nn.Module):
    """Encoder module for an autoencoder.

    Args:
        input_channels (int): Number of input channels. Default: `3`.
        hidden_channels (int): Number of hidden channels. Default: `128`.
        latent_channels (int): Number of latent channels. Default: `4`.
        double_latent_channels (bool): Whether to double the latent channels. Default: `True`.
        channel_multipliers (Tuple[int, ...]): Multipliers for the number of channels in each block. Default: `(1, 2, 4, 8)`.
        num_residual_blocks (int): Number of residual blocks in each block. Default: `4`.
        use_conv_shortcut (bool): Whether to use a conv for the shortcut. Default: `False`.
        dropout (float): Dropout probability. Default: `0.0`.
        resample_with_conv (bool): Whether to use a conv for downsampling. Default: `True`.
        zero_init_last (bool): Whether to initialize the last conv layer to zero. Default: `False`.
    """

    def __init__(self,
                 input_channels: int = 3,
                 hidden_channels: int = 128,
                 latent_channels: int = 4,
                 double_latent_channels: bool = True,
                 channel_multipliers: Tuple[int, ...] = (1, 2, 4, 8),
                 num_residual_blocks: int = 4,
                 use_conv_shortcut=False,
                 dropout_probability: float = 0.0,
                 resample_with_conv: bool = True,
                 zero_init_last: bool = False):
        super().__init__()
        self.input_channels = input_channels
        self.latent_channels = latent_channels
        self.double_latent_channels = double_latent_channels

        self.hidden_channels = hidden_channels
        self.channel_multipliers = channel_multipliers
        self.num_residual_blocks = num_residual_blocks
        self.use_conv_shortcut = use_conv_shortcut
        self.dropout_probability = dropout_probability
        self.resample_with_conv = resample_with_conv
        self.zero_init_last = zero_init_last

        # Inital conv layer to get to the hidden dimensionality
        self.conv_in = nn.Conv2d(input_channels, hidden_channels, kernel_size=3, padding=1)
        nn.init.kaiming_normal_(self.conv_in.weight, nonlinearity='linear')

        # construct the residual blocks
        self.blocks = nn.ModuleList()
        block_input_channels = self.hidden_channels
        block_output_channels = self.hidden_channels
        for i, cm in enumerate(self.channel_multipliers):
            block_output_channels = cm * self.hidden_channels
            # Create the residual blocks
            for _ in range(self.num_residual_blocks):
                block = ResNetBlock(input_channels=block_input_channels,
                                    output_channels=block_output_channels,
                                    use_conv_shortcut=use_conv_shortcut,
                                    dropout_probability=dropout_probability,
                                    zero_init_last=zero_init_last)
                self.blocks.append(block)
                block_input_channels = block_output_channels
            # Add the downsampling block at the end, but not the very end.
            if i < len(self.channel_multipliers) - 1:
                down_block = Downsample(input_channels=block_output_channels,
                                        resample_with_conv=self.resample_with_conv)
                self.blocks.append(down_block)
        # Make the middle blocks
        middle_block_1 = ResNetBlock(input_channels=block_output_channels,
                                     output_channels=block_output_channels,
                                     use_conv_shortcut=use_conv_shortcut,
                                     dropout_probability=dropout_probability,
                                     zero_init_last=zero_init_last)
        self.blocks.append(middle_block_1)

        attention = AttentionLayer(input_channels=block_output_channels)
        self.blocks.append(attention)

        middle_block_2 = ResNetBlock(input_channels=block_output_channels,
                                     output_channels=block_output_channels,
                                     use_conv_shortcut=use_conv_shortcut,
                                     dropout_probability=dropout_probability,
                                     zero_init_last=zero_init_last)
        self.blocks.append(middle_block_2)

        # Make the final layers for the output
        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=block_output_channels, eps=1e-6, affine=True)
        output_channels = 2 * self.latent_channels if self.double_latent_channels else self.latent_channels
        self.conv_out = nn.Conv2d(block_output_channels, output_channels, kernel_size=3, stride=1, padding=1)
        nn.init.kaiming_normal_(self.conv_out.weight, nonlinearity='linear')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward through the encoder."""
        h = self.conv_in(x)
        for block in self.blocks:
            h = block(h)
        h = self.norm_out(h)
        h = F.silu(h)
        h = self.conv_out(h)
        return h


class Decoder(nn.Module):
    """Decoder module for an autoencoder.

    Args:
        latent_channels (int): Number of latent channels. Default: `4`.
        output_channels (int): Number of output channels. Default: `3`.
        hidden_channels (int): Number of hidden channels. Default: `128`.
        channel_multipliers (Tuple[int, ...]): Multipliers for the number of channels in each block. Default: `(1, 2, 4, 8)`.
        num_residual_blocks (int): Number of residual blocks in each block. Default: `4`.
        use_conv_shortcut (bool): Whether to use a conv for the shortcut. Default: `False`.
        dropout (float): Dropout probability. Default: `0.0`.
        resample_with_conv (bool): Whether to use a conv for upsampling. Default: `True`.
        zero_init_last (bool): Whether to initialize the last conv layer to zero. Default: `False`.
    """

    def __init__(self,
                 latent_channels: int = 4,
                 output_channels: int = 3,
                 hidden_channels: int = 128,
                 channel_multipliers: Tuple[int, ...] = (1, 2, 4, 8),
                 num_residual_blocks: int = 4,
                 use_conv_shortcut=False,
                 dropout_probability: float = 0.0,
                 resample_with_conv: bool = True,
                 zero_init_last: bool = False):
        super().__init__()
        self.latent_channels = latent_channels
        self.output_channels = output_channels
        self.hidden_channels = hidden_channels
        self.channel_multipliers = channel_multipliers
        self.num_residual_blocks = num_residual_blocks
        self.use_conv_shortcut = use_conv_shortcut
        self.dropout_probability = dropout_probability
        self.resample_with_conv = resample_with_conv
        self.zero_init_last = zero_init_last

        # Input conv layer to get to the hidden dimensionality
        channels = self.hidden_channels * self.channel_multipliers[-1]
        self.conv_in = nn.Conv2d(self.latent_channels, channels, kernel_size=3, stride=1, padding=1)
        nn.init.kaiming_normal_(self.conv_in.weight, nonlinearity='linear')

        # Make the middle blocks
        self.blocks = nn.ModuleList()
        middle_block_1 = ResNetBlock(input_channels=channels,
                                     output_channels=channels,
                                     use_conv_shortcut=use_conv_shortcut,
                                     dropout_probability=dropout_probability,
                                     zero_init_last=zero_init_last)
        self.blocks.append(middle_block_1)

        attention = AttentionLayer(input_channels=channels)
        self.blocks.append(attention)

        middle_block_2 = ResNetBlock(input_channels=channels,
                                     output_channels=channels,
                                     use_conv_shortcut=use_conv_shortcut,
                                     dropout_probability=dropout_probability,
                                     zero_init_last=zero_init_last)
        self.blocks.append(middle_block_2)

        # construct the residual blocks
        block_channels = channels
        for i, cm in enumerate(self.channel_multipliers[::-1]):
            block_channels = self.hidden_channels * cm
            for _ in range(self.num_residual_blocks + 1):  # Why the +1?
                block = ResNetBlock(input_channels=channels,
                                    output_channels=block_channels,
                                    use_conv_shortcut=use_conv_shortcut,
                                    dropout_probability=dropout_probability,
                                    zero_init_last=zero_init_last)
                self.blocks.append(block)
                channels = block_channels
            # Add the upsampling block at the end, but not the very end.
            if i < len(self.channel_multipliers) - 1:
                upsample = Upsample(input_channels=block_channels, resample_with_conv=self.resample_with_conv)
                self.blocks.append(upsample)
        # Make the final layers for the output
        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=block_channels, eps=1e-6, affine=True)
        self.conv_out = nn.Conv2d(block_channels, self.output_channels, kernel_size=3, stride=1, padding=1)
        nn.init.kaiming_normal_(self.conv_out.weight, nonlinearity='linear')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward through the decoder."""
        h = self.conv_in(x)
        for block in self.blocks:
            h = block(h)
        h = self.norm_out(h)
        h = F.silu(h)
        h = self.conv_out(h)
        return h


class AutoEncoder(nn.Module):
    """Autoencoder module for training a latent diffusion model.

    Args:
        input_channels (int): Number of input channels. Default: `3`.
        output_channels (int): Number of output channels. Default: `3`.
        hidden_channels (int): Number of hidden channels. Default: `128`.
        latent_channels (int): Number of latent channels. Default: `4`.
        double_latent_channels (bool): Whether to double the latent channels. Default: `True`.
        channel_multipliers (Tuple[int, ...]): Multipliers for the number of channels in each block. Default: `(1, 2, 4, 8)`.
        num_residual_blocks (int): Number of residual blocks in each block. Default: `4`.
        use_conv_shortcut (bool): Whether to use a conv for the shortcut. Default: `False`.
        dropout (float): Dropout probability. Default: `0.0`.
        resample_with_conv (bool): Whether to use a conv for down/up sampling. Default: `True`.
        zero_init_last (bool): Whether to initialize the last conv layer to zero. Default: `False`.
    """

    def __init__(self,
                 input_channels: int = 3,
                 output_channels: int = 3,
                 hidden_channels: int = 128,
                 latent_channels: int = 4,
                 double_latent_channels: bool = True,
                 channel_multipliers: Tuple[int, ...] = (1, 2, 4, 8),
                 num_residual_blocks: int = 4,
                 use_conv_shortcut=False,
                 dropout_probability: float = 0.0,
                 resample_with_conv: bool = True,
                 zero_init_last: bool = False):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.hidden_channels = hidden_channels
        self.latent_channels = latent_channels
        self.double_latent_channels = double_latent_channels
        self.channel_multipliers = channel_multipliers
        self.num_residual_blocks = num_residual_blocks
        self.use_conv_shortcut = use_conv_shortcut
        self.dropout_probability = dropout_probability
        self.resample_with_conv = resample_with_conv
        self.zero_init_last = zero_init_last

        self.encoder = Encoder(input_channels=self.input_channels,
                               hidden_channels=self.hidden_channels,
                               latent_channels=self.latent_channels,
                               double_latent_channels=self.double_latent_channels,
                               channel_multipliers=self.channel_multipliers,
                               num_residual_blocks=self.num_residual_blocks,
                               use_conv_shortcut=self.use_conv_shortcut,
                               dropout_probability=self.dropout_probability,
                               resample_with_conv=self.resample_with_conv,
                               zero_init_last=self.zero_init_last)

        channels = 2 * self.latent_channels if self.double_latent_channels else self.latent_channels
        self.quant_conv = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0)
        nn.init.kaiming_normal_(self.quant_conv.weight, nonlinearity='linear')

        self.decoder = Decoder(latent_channels=self.latent_channels,
                               output_channels=self.output_channels,
                               hidden_channels=self.hidden_channels,
                               channel_multipliers=self.channel_multipliers,
                               num_residual_blocks=self.num_residual_blocks,
                               use_conv_shortcut=self.use_conv_shortcut,
                               dropout_probability=self.dropout_probability,
                               resample_with_conv=self.resample_with_conv,
                               zero_init_last=self.zero_init_last)

        self.post_quant_conv = nn.Conv2d(self.latent_channels, self.latent_channels, kernel_size=1, stride=1, padding=0)
        nn.init.kaiming_normal_(self.post_quant_conv.weight, nonlinearity='linear')

    def get_last_layer_weight(self) -> torch.Tensor:
        """Get the weight of the last layer of the decoder."""
        return self.decoder.conv_out.weight

    def encode(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Encode an input tensor into a latent tensor."""
        h = self.encoder(x)
        moments = self.quant_conv(h)
        # Split the moments into mean and log variance
        mean, log_var = moments[:, :self.latent_channels], moments[:, self.latent_channels:]
        return {'mean': mean, 'log_var': log_var}

    def decode(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        """Decode a latent tensor into an output tensor."""
        z = self.post_quant_conv(z)
        x_recon = self.decoder(z)
        return {'x_recon': x_recon}

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward through the autoencoder."""
        encoded = self.encode(x)
        mean, log_var = encoded['mean'], encoded['log_var']
        # Reparameteriztion trick
        z = mean + torch.exp(0.5 * log_var) * torch.randn_like(mean)
        x_recon = self.decode(z)['x_recon']
        return {'x_recon': x_recon, 'latents': z, 'mean': mean, 'log_var': log_var}


class NlayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator.

    Based on code from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py

    Args:
        input_channels (int): Number of input channels. Default: `3`.
        num_filters (int): Number of filters in the first layer. Default: `64`.
        num_layers (int): Number of layers in the discriminator. Default: `3`.
    """

    def __init__(self, input_channels: int = 3, num_filters: int = 64, num_layers: int = 3):
        super().__init__()
        self.input_channels = input_channels
        self.num_filters = num_filters
        self.num_layers = num_layers

        self.blocks = nn.ModuleList()
        self.blocks.append(nn.Conv2d(self.input_channels, self.num_filters, kernel_size=4, stride=2, padding=1))
        out_filters = self.num_filters
        for n in range(1, self.num_layers):
            in_filters = self.num_filters * 2**(n - 1)
            out_filters = self.num_filters * min(2**n, 8)
            conv = nn.Conv2d(in_filters, out_filters, kernel_size=4, stride=2, padding=1, bias=False)
            norm = nn.BatchNorm2d(out_filters)
            nonlinearity = nn.LeakyReLU(0.2, True)
            self.blocks.extend([conv, norm, nonlinearity])
        # Make the output layers
        final_out_filters = self.num_filters * min(2**self.num_layers, 8)
        conv = nn.Conv2d(out_filters, final_out_filters, kernel_size=4, stride=1, padding=1, bias=False)
        norm = nn.BatchNorm2d(final_out_filters)
        nonlinearity = nn.LeakyReLU(0.2, True)
        self.blocks.extend([conv, norm, nonlinearity])
        # Output layer
        output_conv = nn.Conv2d(final_out_filters, 1, kernel_size=4, stride=1, padding=1, bias=False)
        self.blocks.append(output_conv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward through the discriminator."""
        for block in self.blocks:
            x = block(x)
        return x


class GradientReversalLayer(Function):
    """Layer that reverses the direction of the gradient."""

    @staticmethod
    def forward(ctx, x):
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return (-grad_output)


class GradientScalingLayer(nn.Module):
    """Layer that scales the gradient."""

    def __init__(self):
        super().__init__()
        self.scale = 1.0

    def set_scale(self, scale: float):
        self.scale = scale

    def forward(self, x):
        return x

    def backward_hook(self, module, grad_input, grad_output):
        return (self.scale * grad_input[0],)


class AutoEncoderLoss(nn.Module):
    """Loss function for training an autoencoder. Includes LPIPs and a discriminator.

    Args:
        input_key (str): Key for the input to the model. Default: `image`.
        output_channels (int): Number of output channels. Default: `3`.
        learn_log_var (bool): Whether to learn the output log variance. Default: `True`.
        log_var_init (float): Initial value for the log variance. Default: `0.0`.
        kl_divergence_weight (float): Weight for the KL divergence loss. Default: `1.0`.
        lpips_weight (float): Weight for the LPIPs loss. Default: `0.25`.
        discriminator_weight (float): Weight for the discriminator loss. Default: `0.5`.
        discriminator_num_filters (int): Number of filters in the first layer of the discriminator. Default: `64`.
        discriminator_num_layers (int): Number of layers in the discriminator. Default: `3`.
    """

    def __init__(self,
                 input_key: str = 'image',
                 output_channels: int = 3,
                 learn_log_var: bool = True,
                 log_var_init: float = 0.0,
                 kl_divergence_weight: float = 1.0,
                 lpips_weight: float = 0.25,
                 discriminator_weight: float = 0.5,
                 discriminator_num_filters: int = 64,
                 discriminator_num_layers: int = 3):
        super().__init__()
        self.input_key = input_key
        self.output_channels = output_channels
        self.learn_log_var = learn_log_var
        self.log_var_init = log_var_init
        self.kl_divergence_weight = kl_divergence_weight
        self.lpips_weight = lpips_weight
        self.discriminator_weight = discriminator_weight
        self.discriminator_num_filters = discriminator_num_filters
        self.discriminator_num_layers = discriminator_num_layers

        # Set up log variance
        if self.learn_log_var:
            self.log_var = nn.Parameter(torch.zeros(size=()))
        else:
            self.log_var = torch.zeros(size=())
        self.log_var.data.fill_(self.log_var_init)

        # Set up LPIPs loss
        self.lpips = lpips.LPIPS(net='vgg').eval()
        # Ensure that lpips does not get trained
        for param in self.lpips.parameters():
            param.requires_grad_(False)
        for param in self.lpips.net.parameters():
            param.requires_grad_(False)

        # Set up the discriminator
        self.discriminator_num_filters = discriminator_num_filters
        self.discriminator_num_layers = discriminator_num_layers
        self.discriminator_weight = discriminator_weight
        self.discriminator = NlayerDiscriminator(input_channels=self.output_channels,
                                                 num_filters=self.discriminator_num_filters,
                                                 num_layers=self.discriminator_num_layers)
        self.reverse_gradients = GradientReversalLayer()
        self.scale_gradients = GradientScalingLayer()
        self.scale_gradients.register_full_backward_hook(self.scale_gradients.backward_hook)

    def set_discriminator_weight(self, weight: float):
        self.discriminator_weight = weight

    def calc_discriminator_adaptive_weight(self, nll_loss, fake_loss, last_layer):
        # Need to ensure the grad scale from the discriminator back to 1.0 to get the right norm
        self.scale_gradients.set_scale(1.0)
        # Get the grad norm from the nll loss
        nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
        # Get the grad norm for the discriminator loss
        disc_grads = torch.autograd.grad(fake_loss, last_layer, retain_graph=True)[0]
        # Calculate the updated discriminator weight based on the grad norms
        nll_grads_norm = torch.norm(nll_grads)
        disc_grads_norm = torch.norm(disc_grads)
        disc_weight = nll_grads_norm / (disc_grads_norm + 1e-4)
        disc_weight = torch.clamp(disc_weight, 0.0, 1e4).detach()
        disc_weight *= self.discriminator_weight
        # Set the discriminator weight
        self.scale_gradients.set_scale(disc_weight.item())
        return disc_weight, nll_grads_norm, disc_grads_norm

    def forward(self, outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor],
                last_layer: torch.Tensor) -> dict[str, torch.Tensor]:
        losses = {}
        # Basic L1 reconstruction loss
        ae_loss = F.l1_loss(outputs['x_recon'], batch[self.input_key], reduction='none')
        # Count the number of output elements to normalize the loss
        num_output_elements = ae_loss.numel() // ae_loss.shape[0]
        losses['ae_loss'] = ae_loss.mean()

        # LPIPs loss. Images for LPIPS must be in [-1, 1]
        recon_img = outputs['x_recon'].clamp(-1, 1)
        target_img = batch[self.input_key].clamp(-1, 1)
        lpips_loss = self.lpips(recon_img, target_img)
        losses['lpips_loss'] = lpips_loss.mean()

        # Make the nll loss
        rec_loss = ae_loss + self.lpips_weight * lpips_loss
        # Note: the +2 here comes from the nll of the laplace distribution.
        # It's only here to make you feel better by keeping the loss positive for longer.
        nll_loss = rec_loss / torch.exp(self.log_var) + self.log_var + 2
        nll_loss = nll_loss.mean()
        losses['nll_loss'] = nll_loss
        losses['output_variance'] = torch.exp(self.log_var)

        # Discriminator loss
        real = self.discriminator(batch[self.input_key])
        fake = self.discriminator(self.reverse_gradients.apply(self.scale_gradients(recon_img)))
        real_loss = F.binary_cross_entropy_with_logits(real, torch.ones_like(real))
        fake_loss = F.binary_cross_entropy_with_logits(fake, torch.zeros_like(fake))
        losses['disc_real_loss'] = real_loss
        losses['disc_fake_loss'] = fake_loss
        losses['disc_loss'] = 0.5 * (real_loss + fake_loss)

        # Update the adaptive discriminator weight
        disc_weight, nll_grads_norm, disc_grads_norm = self.calc_discriminator_adaptive_weight(
            nll_loss, fake_loss, last_layer)
        losses['disc_weight'] = disc_weight
        losses['nll_grads_norm'] = nll_grads_norm
        losses['disc_grads_norm'] = disc_grads_norm

        # Make the KL divergence loss (effectively regularize the latents)
        mean = outputs['mean']
        log_var = outputs['log_var']
        kl_div_loss = -0.5 * (1 + log_var - mean.pow(2) - log_var.exp())
        # Count the number of latent elements to normalize the loss
        num_latent_elements = mean.numel() // kl_div_loss.shape[0]
        losses['kl_div_loss'] = kl_div_loss.mean()

        # Combine the losses. Downweight the kl_div_loss to account for differing dimensionalities.
        dimensionality_weight = num_latent_elements / num_output_elements
        total_loss = losses['nll_loss'] + self.kl_divergence_weight * dimensionality_weight * losses['kl_div_loss']
        total_loss += losses['disc_loss']
        losses['total'] = total_loss
        return losses


class ComposerAutoEncoder(ComposerModel):
    """Composer wrapper for the AutoEncoder.

    Args:
        input_channels (int): Number of input channels. Default: `3`.
        output_channels (int): Number of output channels. Default: `3`.
        hidden_channels (int): Number of hidden channels. Default: `128`.
        latent_channels (int): Number of latent channels. Default: `4`.
        double_latent_channels (bool): Whether to double the latent channels. Default: `True`.
        channel_multipliers (Tuple[int, ...]): Multipliers for the number of channels in each block. Default: `(1, 2, 4, 4)`.
        num_residual_blocks (int): Number of residual blocks in each block. Default: `2`.
        use_conv_shortcut (bool): Whether to use a conv for the shortcut. Default: `False`.
        dropout (float): Dropout probability. Default: `0.0`.
        resample_with_conv (bool): Whether to use a conv for down/up sampling. Default: `True`.
        zero_init_last (bool): Whether to initialize the last conv layer to zero. Default: `False`.
        input_key (str): Key for the input to the model. Default: `image`.
        learn_log_var (bool): Whether to learn the output log variance. Default: `True`.
        log_var_init (float): Initial value for the log variance. Default: `0.0`.
        kl_divergence_weight (float): Weight for the KL divergence loss. Default: `1.0`.
        lpips_weight (float): Weight for the LPIPs loss. Default: `0.25`.
        discriminator_weight (float): Weight for the discriminator loss. Default: `0.5`.
        discriminator_num_filters (int): Number of filters in the first layer of the discriminator. Default: `64`.
        discriminator_num_layers (int): Number of layers in the discriminator. Default: `3`.
    """

    def __init__(self,
                 input_channels: int = 3,
                 output_channels: int = 3,
                 hidden_channels: int = 128,
                 latent_channels: int = 4,
                 double_latent_channels: bool = True,
                 channel_multipliers: Tuple[int, ...] = (1, 2, 4, 4),
                 num_residual_blocks: int = 2,
                 use_conv_shortcut=False,
                 dropout_probability: float = 0.0,
                 resample_with_conv: bool = True,
                 zero_init_last: bool = False,
                 input_key: str = 'image',
                 learn_log_var: bool = True,
                 log_var_init: float = 0.0,
                 kl_divergence_weight: float = 1.0,
                 lpips_weight: float = 0.25,
                 discriminator_weight: float = 0.5,
                 discriminator_num_filters: int = 64,
                 discriminator_num_layers: int = 3):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.hidden_channels = hidden_channels
        self.latent_channels = latent_channels
        self.double_latent_channels = double_latent_channels
        self.channel_multipliers = channel_multipliers
        self.num_residual_blocks = num_residual_blocks
        self.use_conv_shortcut = use_conv_shortcut
        self.dropout_probability = dropout_probability
        self.resample_with_conv = resample_with_conv
        self.zero_init_last = zero_init_last
        self.input_key = input_key

        self.model = AutoEncoder(input_channels=self.input_channels,
                                 output_channels=self.output_channels,
                                 hidden_channels=self.hidden_channels,
                                 latent_channels=self.latent_channels,
                                 double_latent_channels=self.double_latent_channels,
                                 channel_multipliers=self.channel_multipliers,
                                 num_residual_blocks=self.num_residual_blocks,
                                 use_conv_shortcut=self.use_conv_shortcut,
                                 dropout_probability=self.dropout_probability,
                                 resample_with_conv=self.resample_with_conv,
                                 zero_init_last=self.zero_init_last)

        self.learn_log_var = learn_log_var
        self.log_var_init = log_var_init
        self.kl_divergence_weight = kl_divergence_weight
        self.lpips_weight = lpips_weight

        self.autoencoder_loss = AutoEncoderLoss(input_key=self.input_key,
                                                output_channels=self.output_channels,
                                                learn_log_var=self.learn_log_var,
                                                log_var_init=self.log_var_init,
                                                kl_divergence_weight=self.kl_divergence_weight,
                                                lpips_weight=self.lpips_weight,
                                                discriminator_weight=discriminator_weight,
                                                discriminator_num_filters=discriminator_num_filters,
                                                discriminator_num_layers=discriminator_num_layers)

        # Set up train metrics
        train_metrics = [MeanSquaredError()]
        self.train_metrics = {metric.__class__.__name__: metric for metric in train_metrics}
        # Set up val metrics
        psnr_metric = PeakSignalNoiseRatio(data_range=2.0)
        ssim_metric = StructuralSimilarityIndexMeasure(data_range=2.0)
        lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type='vgg')
        val_metrics = [MeanSquaredError(), MeanMetric(), lpips_metric, psnr_metric, ssim_metric]
        self.val_metrics = {metric.__class__.__name__: metric for metric in val_metrics}

    def get_last_layer_weight(self) -> torch.Tensor:
        """Get the weight of the last layer of the decoder."""
        return self.model.get_last_layer_weight()

    def forward(self, batch):
        outputs = self.model(batch[self.input_key])
        return outputs

    def loss(self, outputs, batch):
        last_layer = self.get_last_layer_weight()
        return self.autoencoder_loss(outputs, batch, last_layer)

    def eval_forward(self, batch, outputs=None):
        if outputs is not None:
            return outputs
        outputs = self.forward(batch)
        return outputs

    def get_metrics(self, is_train: bool = False):
        if is_train:
            metrics = self.train_metrics
        else:
            metrics = self.val_metrics

        if isinstance(metrics, Metric):
            metrics_dict = {metrics.__class__.__name__: metrics}
        elif isinstance(metrics, list):
            metrics_dict = {metrics.__class__.__name__: metric for metric in metrics}
        else:
            metrics_dict = {}
            for name, metric in metrics.items():
                assert isinstance(metric, Metric)
                metrics_dict[name] = metric

        return metrics_dict

    def update_metric(self, batch, outputs, metric):
        clamped_imgs = outputs['x_recon'].clamp(-1, 1)
        if isinstance(metric, MeanMetric):
            metric.update(torch.square(outputs['latents']))
        elif isinstance(metric, LearnedPerceptualImagePatchSimilarity):
            metric.update(clamped_imgs, batch[self.input_key])
        elif isinstance(metric, PeakSignalNoiseRatio):
            metric.update(clamped_imgs, batch[self.input_key])
        elif isinstance(metric, StructuralSimilarityIndexMeasure):
            metric.update(clamped_imgs, batch[self.input_key])
        else:
            metric.update(outputs['x_recon'], batch[self.input_key])


class ComposerDiffusersAutoEncoder(ComposerModel):
    """Composer wrapper for the Huggingface Diffusers Autoencoder.

    Args:
        model (diffusers.AutoencoderKL): Diffusers autoencoder to train.
        loss_fn (AutoEncoderLoss): Auto encoder loss module.
        input_key (str): Key for the input to the model. Default: `image`.
    """

    def __init__(self, model: AutoencoderKL, autoencoder_loss: AutoEncoderLoss, input_key: str = 'image'):
        super().__init__()
        self.model = model
        self.autoencoder_loss = autoencoder_loss
        self.input_key = input_key

        # Set up train metrics
        train_metrics = [MeanSquaredError()]
        self.train_metrics = {metric.__class__.__name__: metric for metric in train_metrics}
        # Set up val metrics
        psnr_metric = PeakSignalNoiseRatio(data_range=2.0)
        ssim_metric = StructuralSimilarityIndexMeasure(data_range=2.0)
        lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type='vgg')
        val_metrics = [MeanSquaredError(), MeanMetric(), lpips_metric, psnr_metric, ssim_metric]
        self.val_metrics = {metric.__class__.__name__: metric for metric in val_metrics}

    def get_last_layer_weight(self) -> torch.Tensor:
        """Get the weight of the last layer of the decoder."""
        return self.model.decoder.conv_out.weight

    def forward(self, batch):
        latent_dist = self.model.encode(batch[self.input_key])['latent_dist']
        latents = latent_dist.sample()
        mean, log_var = latent_dist.mean, latent_dist.logvar
        recon = self.model.decode(latents).sample
        return {'x_recon': recon, 'latents': latents, 'mean': mean, 'log_var': log_var}

    def loss(self, outputs, batch):
        last_layer = self.get_last_layer_weight()
        return self.autoencoder_loss(outputs, batch, last_layer)

    def eval_forward(self, batch, outputs=None):
        # Skip this if outputs have already been computed, e.g. during training
        if outputs is not None:
            return outputs
        outputs = self.forward(batch)
        return outputs

    def get_metrics(self, is_train: bool = False):
        if is_train:
            metrics = self.train_metrics
        else:
            metrics = self.val_metrics

        if isinstance(metrics, Metric):
            metrics_dict = {metrics.__class__.__name__: metrics}
        elif isinstance(metrics, list):
            metrics_dict = {metrics.__class__.__name__: metric for metric in metrics}
        else:
            metrics_dict = {}
            for name, metric in metrics.items():
                assert isinstance(metric, Metric)
                metrics_dict[name] = metric

        return metrics_dict

    def update_metric(self, batch, outputs, metric):
        clamped_imgs = outputs['x_recon'].clamp(-1, 1)
        if isinstance(metric, MeanMetric):
            metric.update(torch.square(outputs['latents']))
        elif isinstance(metric, LearnedPerceptualImagePatchSimilarity):
            metric.update(clamped_imgs, batch[self.input_key])
        elif isinstance(metric, PeakSignalNoiseRatio):
            metric.update(clamped_imgs, batch[self.input_key])
        elif isinstance(metric, StructuralSimilarityIndexMeasure):
            metric.update(clamped_imgs, batch[self.input_key])
        else:
            metric.update(outputs['x_recon'], batch[self.input_key])
