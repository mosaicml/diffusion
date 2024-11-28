# Copyright 2022 MosaicML Diffusion authors
# SPDX-License-Identifier: Apache-2.0

"""Logger for transformer activation statistics."""

from collections import defaultdict

import torch
from composer import Callback, Logger, State
from torch.nn.parallel import DistributedDataParallel


class LogActivationStatistics(Callback):
    """Logging callback for activation statistics."""

    def __init__(self):
        self.hook_handles = []
        self.activations = {}
        self.activation_norms = defaultdict(float)
        self.batch_counter = 0

    def activation_hook(self, name):

        def hook_fn(module, input, output):
            self.activations[name] = output.cpu()

        return hook_fn

    def register_hooks(self, model):
        for name, layer in model.named_modules():
            if 'autoencoder' not in name and 'adaLN' not in name and ('.attention' in name or 'linear' in name):
                handle = layer.register_forward_hook(self.activation_hook(name))
                self.hook_handles.append(handle)

    def remove_hooks(self):
        for handle in self.hook_handles:
            handle.remove()

    def eval_start(self, state: State, logger: Logger):
        if isinstance(state.model, DistributedDataParallel):
            model = state.model.module
        else:
            model = state.model
        self.register_hooks(model)

    def eval_batch_end(self, state: State, logger: Logger):
        for k, v in self.activations.items():
            self.activation_norms[k] = self.batch_counter * self.activation_norms[k] / (self.batch_counter + 1)
            stats = sum(torch.abs(t).mean().item() for t in v) / len(v)
            self.activation_norms[k] += stats / (self.batch_counter + 1)
        self.batch_counter += 1

    def eval_end(self, state: State, logger: Logger):
        norms = {}
        for k, v in self.activation_norms.items():
            norms[f'activation-statistics/{k}'] = v
        logger.log_metrics(norms)
        self.remove_hooks()
        self.activations.clear()
        self.batch_counter = 0
