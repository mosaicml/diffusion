# Copyright 2022 MosaicML Diffusion authors
# SPDX-License-Identifier: Apache-2.0

"""Algorithm to pin diffusion process noise."""

from typing import Any, Dict

import torch
from composer.core import Algorithm, Event, State
from composer.loggers import Logger
from composer.utils import dist

from diffusion.models import StableDiffusion


class PinDiffusionRNG(Algorithm):
    """Algorithm to pin diffusion process noise."""

    def __init__(self) -> None:
        self.seed = 0
        self.train_rng_state = None
        self.eval_rng_state = None
        self.train_rng_generator = torch.Generator()
        self.eval_rng_generator = torch.Generator()

    def match(self, event: Event, state: State) -> bool:
        return event in (Event.INIT, Event.EVAL_START, Event.EVAL_END)

    def apply(self, event: Event, state: State, logger: Logger) -> None:
        if hasattr(state.model, 'rng_generator'):
            model = state.model
        elif hasattr(state.model, 'module') and hasattr(state.model.module, 'rng_generator'):
            model = state.model.module
        else:
            raise ValueError('Model does not have an rng_generator')
        assert isinstance(model, StableDiffusion)

        if event == Event.INIT:
            # Create the RNG generators
            self.train_rng_generator = torch.Generator(device=state.device._device)
            self.eval_rng_generator = torch.Generator(device=state.device._device)
            # Set the seed
            self.seed = state.rank_zero_seed + dist.get_global_rank()
            self.train_rng_generator.manual_seed(self.seed)
            self.eval_rng_generator.manual_seed(self.seed)
            # Set the states if they exist
            if self.train_rng_state is not None:
                self.train_rng_generator.set_state(self.train_rng_state)
            if self.eval_rng_state is not None:
                self.eval_rng_generator.set_state(self.eval_rng_state)
            # Set the train rng generator
            model.set_rng_generator(self.train_rng_generator)
        elif event == Event.EVAL_START:
            # Reset the eval rng generator to ensure the same randomness is used every eval epoch
            self.eval_rng_generator = self.eval_rng_generator.manual_seed(self.seed)
            # Set the model's rng generator to the eval rng generator
            model.set_rng_generator(self.eval_rng_generator)
        elif event == Event.EVAL_END:
            # Set the model's rng generator to the train rng generator
            model.set_rng_generator(self.train_rng_generator)

    def state_dict(self) -> Dict[str, Any]:
        state_dict = super().state_dict()
        state_dict['train_rng_state'] = self.train_rng_generator.get_state()
        state_dict['eval_rng_state'] = self.eval_rng_generator.get_state()
        return state_dict

    def load_state_dict(self, state: Dict[str, Any], strict: bool = False):
        self.train_rng_state = state['train_rng_state']
        self.eval_rng_state = state['eval_rng_state']
