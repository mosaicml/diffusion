# Copyright 2022 MosaicML Diffusion authors
# SPDX-License-Identifier: Apache-2.0

"""Logger for latent statistics."""

import math
from collections import defaultdict
from typing import Dict

import torch
from composer import Callback, Logger, State


class LogLatentStatistics(Callback):
    """Logging callback for latent statistics.

    Args:
        latent_key (str, optional): The key for the latents in the outputs dict. Defaults to 'latents'.
    """

    def __init__(self, latent_key: str = 'latents'):
        self.latent_key = latent_key
        self.counter = 0
        self.latent_statistics = defaultdict(float)

    def state_dict(self) -> Dict[str, float]:
        return dict(self.latent_statistics)

    def load_state_dict(self, state: Dict[str, float]) -> None:
        latent_statistics = defaultdict(float)
        for k, v in state.items():
            latent_statistics[k] = v
        self.latent_statistics = latent_statistics

    def _running_average(self, old_value: float, new_value: float) -> float:
        """Compute the running average of a value."""
        return old_value * self.counter / (self.counter + 1) + new_value / (self.counter + 1)

    def _calc_std(self, mean: float, second_moment: float) -> float:
        """Compute the standard deviation from the mean and the second moment."""
        return math.sqrt(second_moment - mean**2)

    def _get_latents(self, state: State) -> torch.Tensor:
        """Get the latents from the state."""
        # Get outputs from the eval forward pass
        outputs = state.outputs
        assert isinstance(outputs, dict), 'Outputs must be a dict, got {}'.format(type(outputs))
        # Get the latents
        if self.latent_key in outputs:
            latents = outputs[self.latent_key]  # type: ignore
        else:
            raise ValueError(f'Latent key {self.latent_key} not found in outputs.')
        return latents

    def eval_start(self, state: State, logger: Logger):
        # Reset the counter and wipe the latent statistics
        self.counter = 0
        self.latent_statistics = defaultdict(lambda: 0.0)

    def eval_batch_end(self, state: State, logger: Logger):
        # Get the latents
        latents = self._get_latents(state)
        # Get the global mean and standard deviation
        self.latent_statistics['global_mean'] = self._running_average(self.latent_statistics['global_mean'],
                                                                      latents.mean().item())
        self.latent_statistics['global_second_moment'] = self._running_average(
            self.latent_statistics['global_second_moment'], (latents**2).mean().item())
        # Get the channelwise mean and variance. Take the mean over all dims except the channel dim
        other_dims = [i for i in range(latents.ndim) if i != 1]
        channelwise_means = latents.mean(dim=other_dims)
        channelwise_second_moments = (latents**2).mean(dim=other_dims)
        for i in range(latents.shape[1]):
            self.latent_statistics[f'channel_mean_{i}'] = self._running_average(
                self.latent_statistics[f'channel_mean_{i}'], channelwise_means[i].item())
            self.latent_statistics[f'channel_second_moment_{i}'] = self._running_average(
                self.latent_statistics[f'channel_second_moment_{i}'], channelwise_second_moments[i].item())
        self.counter += 1

    def eval_end(self, state: State, logger: Logger):
        latents = self._get_latents(state)
        # Compute the standard deviations from the mean and the mean second moment
        self.latent_statistics['global_std'] = self._calc_std(self.latent_statistics['global_mean'],
                                                              self.latent_statistics['global_second_moment'])
        del self.latent_statistics['global_second_moment']
        for i in range(latents.shape[1]):
            self.latent_statistics[f'channel_std_{i}'] = self._calc_std(
                self.latent_statistics[f'channel_mean_{i}'], self.latent_statistics[f'channel_second_moment_{i}'])
            del self.latent_statistics[f'channel_second_moment_{i}']
        latent_statistics = {'latent_statistics/' + k: v for k, v in self.latent_statistics.items()}
        logger.log_metrics(latent_statistics)
