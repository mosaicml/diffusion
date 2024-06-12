# Copyright 2022 MosaicML Diffusion authors
# SPDX-License-Identifier: Apache-2.0

"""Utils for working with diffusion  schedulers."""

import torch


def shift_noise_schedule(noise_scheduler, base_dim: int = 64, shift_dim: int = 64):
    """Shifts the function SNR(t) for a noise scheduler to correct for resolution changes."""
    # First, we need to get the original SNR(t) function
    alpha_bar = noise_scheduler.alphas_cumprod
    SNR = alpha_bar / (1 - alpha_bar)
    # Shift the SNR acorrording to the resolution change
    SNR_shifted = (base_dim / shift_dim)**2 * SNR
    # Get the new alpha_bars
    alpha_bar_shifted = SNR_shifted / (1 + SNR_shifted)
    # Get the new alpha values
    alpha_shifted = torch.empty_like(alpha_bar_shifted)
    alpha_shifted[0] = alpha_bar_shifted[0]
    alpha_shifted[1:] = alpha_bar_shifted[1:] / alpha_bar_shifted[:-1]
    # Get the new beta values
    beta_shifted = 1 - alpha_shifted
    # Update the noise scheduler
    noise_scheduler.alphas = alpha_shifted
    noise_scheduler.betas = beta_shifted
    noise_scheduler.alphas_cumprod = alpha_bar_shifted
    return noise_scheduler
