# Copyright 2022 MosaicML Diffusion authors
# SPDX-License-Identifier: Apache-2.0

"""Scheduler algorithm for a discriminator."""
import torch.nn as nn
from composer.core import Algorithm, Event, State, TimeUnit
from composer.loggers import Logger

__all__ = ['DiscriminatorSchedule']


class DiscriminatorSchedule(Algorithm):
    """Scheduling algorithm for the autoencoder discriminator.

    Args:
        start_iteration (int): The iteration to start training the discriminator. Default: `0`.
    """

    def __init__(self, start_iteration: int = 0) -> None:
        self.start_iteration = start_iteration
        self.lr = 0.0
        self.weight_decay = 0.0
        self.discriminator_weight = 0.0

    def match(self, event: Event, state: State) -> bool:
        return event in [Event.INIT, Event.BATCH_START, Event.AFTER_LOSS]

    def apply(self, event: Event, state: State, logger: Logger) -> None:
        if hasattr(state.model, 'autoencoder_loss') and isinstance(state.model.autoencoder_loss, nn.Module):
            autoencoder_loss = state.model.autoencoder_loss
        elif isinstance(state.model.module, nn.Module) and hasattr(state.model.module, 'autoencoder_loss'):
            autoencoder_loss = state.model.module.autoencoder_loss
        else:
            raise ValueError('Model does not have a discriminator weight')
        assert isinstance(autoencoder_loss, nn.Module) and callable(autoencoder_loss.set_discriminator_weight)

        # Grab the relevant scheduler params from the model and optimizer on init
        if event == Event.INIT:
            # Get the model's discriminator weight
            self.discriminator_weight = autoencoder_loss.discriminator_weight
            # Get the learning rate and weight decay from the optimizer
            self.lr = state.optimizers[0].param_groups[1]['lr']
            self.weight_decay = state.optimizers[0].param_groups[1]['weight_decay']

        # Ensure the discriminator is training/not training when appropriate
        elif event == Event.BATCH_START:
            if state.timestamp.get(TimeUnit.BATCH).value >= self.start_iteration:
                # Turn on the discriminator
                state.optimizers[0].param_groups[1]['lr'] = self.lr
                state.optimizers[0].param_groups[1]['weight_decay'] = self.weight_decay
                autoencoder_loss.set_discriminator_weight(self.discriminator_weight)
            else:
                # Turn off the discriminator
                state.optimizers[0].param_groups[1]['lr'] = 0.0
                state.optimizers[0].param_groups[1]['weight_decay'] = 0.0
                autoencoder_loss.set_discriminator_weight(0.0)
