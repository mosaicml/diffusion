# Copyright 2022 MosaicML Diffusion authors
# SPDX-License-Identifier: Apache-2.0

"""Autoencoder parts for training latent diffusion models.

Based on the implementation from https://github.com/CompVis/stable-diffusion
"""

import os
from typing import Dict, Tuple

import lpips
import torch
import torch.nn as nn
import torch.nn.functional as F
from composer.models import ComposerModel
from composer.utils.file_helpers import get_file
from diffusers import AutoencoderKL
from torchmetrics import MeanMetric, MeanSquaredError, Metric
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from diffusion.models.layers import AttentionLayer, Downsample, GradientScalingLayer, ResNetBlock, Upsample


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
        use_attention (bool): Whether to use attention layers. Default: `True`.
    """

    def __init__(self,
                 input_channels: int = 3,
                 hidden_channels: int = 128,
                 latent_channels: int = 4,
                 double_latent_channels: bool = True,
                 channel_multipliers: Tuple[int, ...] = (1, 2, 4, 8),
                 num_residual_blocks: int = 4,
                 use_conv_shortcut: bool = False,
                 dropout_probability: float = 0.0,
                 resample_with_conv: bool = True,
                 zero_init_last: bool = False,
                 use_attention: bool = True):
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
        self.use_attention = use_attention

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

        if self.use_attention:
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
        # Output layer is immediately after a silu. Need to account for that in init.
        self.conv_out.weight.data *= 1.6761

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
        use_attention (bool): Whether to use attention layers. Default: `True`.
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
                 zero_init_last: bool = False,
                 use_attention: bool = True):
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
        self.use_attention = use_attention

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

        if self.use_attention:
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
        # Output layer is immediately after a silu. Need to account for that in init.
        # Also want the output variance to mimic images with pixel values uniformly distributed in [-1, 1].
        # These two effects essentially cancel out.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward through the decoder."""
        h = self.conv_in(x)
        for block in self.blocks:
            h = block(h)
        h = self.norm_out(h)
        h = F.silu(h)
        h = self.conv_out(h)
        return h


class GaussianDistribution:
    """Gaussian distribution parameterized with mean and log variance."""

    def __init__(self, mean: torch.Tensor, log_var: torch.Tensor):
        self.mean = mean
        self.log_var = log_var
        self.var = torch.exp(log_var)
        self.std = torch.exp(0.5 * log_var)

    def __getitem__(self, key):
        if key == 'latent_dist':
            return GaussianDistribution(self.mean, self.log_var)
        elif key == 'mean':
            return self.mean[key]
        elif key == 'log_var':
            return self.log_var[key]
        else:
            raise KeyError(key)

    @property
    def latent_dist(self):
        return self

    def sample(self) -> torch.Tensor:
        """Sample from the distribution."""
        return self.mean + self.std * torch.randn_like(self.mean)


class AutoEncoderOutput:
    """Output from an autoencoder."""

    def __init__(self, x_recon: torch.Tensor):
        self.x_recon = x_recon

    def __getitem__(self, key):
        if key == 'x_recon':
            return self.x_recon
        else:
            raise KeyError(key)

    @property
    def sample(self) -> torch.Tensor:
        """Sample from the output."""
        return self.x_recon


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
        use_attention (bool): Whether to use attention layers. Default: `True`.
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
                 use_attention: bool = True):
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
        self.use_attention = use_attention
        self.config = {}
        self.set_extra_state(None)

        self.encoder = Encoder(input_channels=self.input_channels,
                               hidden_channels=self.hidden_channels,
                               latent_channels=self.latent_channels,
                               double_latent_channels=self.double_latent_channels,
                               channel_multipliers=self.channel_multipliers,
                               num_residual_blocks=self.num_residual_blocks,
                               use_conv_shortcut=self.use_conv_shortcut,
                               dropout_probability=self.dropout_probability,
                               resample_with_conv=self.resample_with_conv,
                               zero_init_last=self.zero_init_last,
                               use_attention=self.use_attention)

        channels = 2 * self.latent_channels if self.double_latent_channels else self.latent_channels
        self.quant_conv = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0)
        nn.init.kaiming_normal_(self.quant_conv.weight, nonlinearity='linear')
        # KL divergence is minimized when mean is 0.0 and log variance is 0.0
        # However, this corresponds to no information in the latent space.
        # So, init these such that latents are mean 0 and variance 1, with a rough snr of 1
        self.quant_conv.weight.data[:channels // 2] *= 0.707
        self.quant_conv.weight.data[channels // 2:] *= 0.707
        if self.quant_conv.bias is not None:
            self.quant_conv.bias.data[channels // 2:].fill_(-0.9431)

        self.decoder = Decoder(latent_channels=self.latent_channels,
                               output_channels=self.output_channels,
                               hidden_channels=self.hidden_channels,
                               channel_multipliers=self.channel_multipliers,
                               num_residual_blocks=self.num_residual_blocks,
                               use_conv_shortcut=self.use_conv_shortcut,
                               dropout_probability=self.dropout_probability,
                               resample_with_conv=self.resample_with_conv,
                               zero_init_last=self.zero_init_last,
                               use_attention=self.use_attention)

        self.post_quant_conv = nn.Conv2d(self.latent_channels, self.latent_channels, kernel_size=1, stride=1, padding=0)
        nn.init.kaiming_normal_(self.post_quant_conv.weight, nonlinearity='linear')

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def get_extra_state(self):
        return {'config': self.config}

    def set_extra_state(self, state):
        # Save the autoencoder config
        config = {}
        config['input_channels'] = self.input_channels
        config['output_channels'] = self.output_channels
        config['hidden_channels'] = self.hidden_channels
        config['latent_channels'] = self.latent_channels
        config['double_latent_channels'] = self.double_latent_channels
        config['channel_multipliers'] = self.channel_multipliers
        config['num_residual_blocks'] = self.num_residual_blocks
        config['use_conv_shortcut'] = self.use_conv_shortcut
        config['dropout_probability'] = self.dropout_probability
        config['resample_with_conv'] = self.resample_with_conv
        config['use_attention'] = self.use_attention
        config['zero_init_last'] = self.zero_init_last
        self.config = config

    def get_last_layer_weight(self) -> torch.Tensor:
        """Get the weight of the last layer of the decoder."""
        return self.decoder.conv_out.weight

    def encode(self, x: torch.Tensor) -> GaussianDistribution:
        """Encode an input tensor into a latent tensor."""
        h = self.encoder(x)
        moments = self.quant_conv(h)
        # Split the moments into mean and log variance
        mean, log_var = moments[:, :self.latent_channels], moments[:, self.latent_channels:]
        return GaussianDistribution(mean, log_var)

    def decode(self, z: torch.Tensor) -> AutoEncoderOutput:
        """Decode a latent tensor into an output tensor."""
        z = self.post_quant_conv(z)
        x_recon = self.decoder(z)
        return AutoEncoderOutput(x_recon)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward through the autoencoder."""
        encoded_dist = self.encode(x)
        z = encoded_dist.sample()
        x_recon = self.decode(z)['x_recon']
        return {'x_recon': x_recon, 'latents': z, 'mean': encoded_dist.mean, 'log_var': encoded_dist.log_var}


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

        self.blocks = nn.Sequential()
        input_conv = nn.Conv2d(self.input_channels, self.num_filters, kernel_size=4, stride=2, padding=1)
        nn.init.kaiming_normal_(input_conv.weight, nonlinearity='linear')
        nonlinearity = nn.LeakyReLU(0.2, True)
        self.blocks.extend([input_conv, nonlinearity])

        num_filters = self.num_filters
        out_filters = self.num_filters
        for n in range(1, self.num_layers):
            out_filters = self.num_filters * min(2**n, 8)
            conv = nn.Conv2d(num_filters, out_filters, kernel_size=4, stride=2, padding=1, bias=False)
            num_filters = out_filters
            # Init these as if a linear layer follows them because batchnorm happens before leaky relu.
            nn.init.kaiming_normal_(conv.weight, nonlinearity='linear')
            norm = nn.BatchNorm2d(out_filters)
            nonlinearity = nn.LeakyReLU(0.2, True)
            self.blocks.extend([conv, norm, nonlinearity])
        # Make the output layers
        final_out_filters = self.num_filters * min(2**self.num_layers, 8)
        conv = nn.Conv2d(out_filters, final_out_filters, kernel_size=4, stride=1, padding=1, bias=False)
        nn.init.kaiming_normal_(conv.weight, nonlinearity='linear')
        norm = nn.BatchNorm2d(final_out_filters)
        nonlinearity = nn.LeakyReLU(0.2, True)
        self.blocks.extend([conv, norm, nonlinearity])
        # Output layer
        output_conv = nn.Conv2d(final_out_filters, 1, kernel_size=4, stride=1, padding=1, bias=False)
        nn.init.kaiming_normal_(output_conv.weight, nonlinearity='linear')
        # Should init output layer such that outputs are generally within the linear region of a sigmoid.
        # This likely makes little difference in practice.
        output_conv.weight.data *= 0.1
        self.blocks.append(output_conv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward through the discriminator."""
        return self.blocks(x)


class AutoEncoderLoss(nn.Module):
    """Loss function for training an autoencoder. Includes LPIPs and a discriminator.

    Args:
        input_key (str): Key for the input to the model. Default: `image`.
        ae_output_channels (int): Number of output channels. Default: `3`.
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
                 ae_output_channels: int = 3,
                 learn_log_var: bool = True,
                 log_var_init: float = 0.0,
                 kl_divergence_weight: float = 1.0,
                 lpips_weight: float = 0.25,
                 discriminator_weight: float = 0.5,
                 discriminator_num_filters: int = 64,
                 discriminator_num_layers: int = 3):
        super().__init__()
        self.input_key = input_key
        self.ae_output_channels = ae_output_channels
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
        self.discriminator = NlayerDiscriminator(input_channels=self.ae_output_channels,
                                                 num_filters=self.discriminator_num_filters,
                                                 num_layers=self.discriminator_num_layers)
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
        # Set the discriminator weight. It should be negative to reverse gradients into the autoencoder.
        self.scale_gradients.set_scale(-disc_weight.item())
        return disc_weight, nll_grads_norm, disc_grads_norm

    def forward(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor],
                last_layer: torch.Tensor) -> Dict[str, torch.Tensor]:
        losses = {}
        # Basic L1 reconstruction loss
        ae_loss = F.l1_loss(outputs['x_recon'], batch[self.input_key], reduction='none')
        # Count the number of output elements to normalize the loss
        num_output_elements = ae_loss[0].numel()
        losses['ae_loss'] = ae_loss.mean()

        # LPIPs loss. Images for LPIPS must be in [-1, 1]
        recon_img = outputs['x_recon'].clamp(-1, 1)
        target_img = batch[self.input_key].clamp(-1, 1)
        lpips_loss = self.lpips(recon_img, target_img)
        losses['lpips_loss'] = lpips_loss.mean()

        # Make the nll loss
        rec_loss = ae_loss + self.lpips_weight * lpips_loss
        # Note: the + 2 here comes from the nll of the laplace distribution.
        nll_loss = rec_loss / torch.exp(self.log_var) + self.log_var + 2
        nll_loss = nll_loss.mean()
        losses['nll_loss'] = nll_loss
        losses['output_variance'] = torch.exp(self.log_var)

        # Discriminator loss
        real = self.discriminator(batch[self.input_key])
        fake = self.discriminator(self.scale_gradients(outputs['x_recon']))
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
        num_latent_elements = mean[0].numel()
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
        model (AutoEncoder): AutoEncoder model to train.
        autoencoder_loss (AutoEncoderLoss): Auto encoder loss module.
        input_key (str): Key for the input to the model. Default: `image`.
    """

    def __init__(self, model: AutoEncoder, autoencoder_loss: AutoEncoderLoss, input_key: str = 'image'):
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
            metric.update(torch.sqrt(torch.square(outputs['latents'])))
        elif isinstance(metric, LearnedPerceptualImagePatchSimilarity):
            metric.update(clamped_imgs, batch[self.input_key])
        elif isinstance(metric, PeakSignalNoiseRatio):
            metric.update(clamped_imgs, batch[self.input_key])
        elif isinstance(metric, StructuralSimilarityIndexMeasure):
            metric.update(clamped_imgs, batch[self.input_key])
        elif isinstance(metric, MeanSquaredError):
            metric.update(outputs['x_recon'], batch[self.input_key])
        else:
            metric.update(outputs['x_recon'], batch[self.input_key])


class ComposerDiffusersAutoEncoder(ComposerAutoEncoder):
    """Composer wrapper for the Huggingface Diffusers Autoencoder.

    Args:
        model (diffusers.AutoencoderKL): Diffusers autoencoder to train.
        autoencoder_loss (AutoEncoderLoss): Auto encoder loss module.
        input_key (str): Key for the input to the model. Default: `image`.
    """

    def __init__(self, model: AutoencoderKL, autoencoder_loss: AutoEncoderLoss, input_key: str = 'image'):
        super().__init__(model, autoencoder_loss, input_key)
        self.model = model
        self.autoencoder_loss = autoencoder_loss
        self.input_key = input_key

    def get_last_layer_weight(self) -> torch.Tensor:
        """Get the weight of the last layer of the decoder."""
        return self.model.decoder.conv_out.weight

    def forward(self, batch):
        latent_dist = self.model.encode(batch[self.input_key])['latent_dist']
        latents = latent_dist.sample()
        mean, log_var = latent_dist.mean, latent_dist.logvar
        recon = self.model.decode(latents).sample
        return {'x_recon': recon, 'latents': latents, 'mean': mean, 'log_var': log_var}


def load_autoencoder(load_path: str, local_path: str = '/tmp/autoencoder_weights.pt'):
    """Function to load an AutoEncoder from a composer checkpoint without the loss weights.

    Will also load the latent statistics if the statistics tracking callback was used.

    Args:
        load_path (str): Path to the composer checkpoint. Can be a local folder, URL, or composer object store.
        local_path (str): Local path to save the autoencoder weights to. Default: `/tmp/autoencoder_weights.pt`.

    Returns:
        autoencoder (AutoEncoder): AutoEncoder model with weights loaded from the checkpoint.
        latent_statistics (Dict[str, Union[list, float]]): Dictionary of latent statistics if present, else `None`.
    """
    # Download the autoencoder weights and init them
    if not os.path.exists(local_path):
        get_file(path=load_path, destination=local_path)
    # Load the autoencoder weights from the state dict
    state_dict = torch.load(local_path, map_location='cpu')
    # Get the config from the state dict and init the model using it
    autoencoder_config = state_dict['state']['model']['model._extra_state']['config']
    autoencoder = AutoEncoder(**autoencoder_config)
    # Need to clean up the state dict to remove loss and metrics.
    cleaned_state_dict = {}
    for key in list(state_dict['state']['model'].keys()):
        if key.split('.')[0] == 'model':
            cleaned_key = '.'.join(key.split('.')[1:])
            cleaned_state_dict[cleaned_key] = state_dict['state']['model'][key]
    # Load the cleaned state dict into the model
    autoencoder.load_state_dict(cleaned_state_dict, strict=True)
    # If present, extract the channel means and standard deviations from the state dict
    if 'LogLatentStatistics' in state_dict['state']['callbacks']:
        latent_statistics = {'latent_means': [], 'latent_stds': []}
        logged_latent_stats = state_dict['state']['callbacks']['LogLatentStatistics']
        # Extract the channelwise latent means and stds
        for i in range(autoencoder_config['latent_channels']):
            latent_statistics['latent_means'].append(logged_latent_stats[f'channel_mean_{i}'])
            latent_statistics['latent_stds'].append(logged_latent_stats[f'channel_std_{i}'])
        # Extract the global latent means and second moment
        latent_statistics['global_mean'] = logged_latent_stats['global_mean']
        latent_statistics['global_std'] = logged_latent_stats['global_std']
    else:
        latent_statistics = None
    return autoencoder, latent_statistics
