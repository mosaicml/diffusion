# Copyright 2022 MosaicML Diffusion authors
# SPDX-License-Identifier: Apache-2.0

"""LoRA Planner."""
from torch.distributed.checkpoint._nested_dict import flatten_state_dict
from torch.distributed.checkpoint._sharded_tensor_utils import _flatten_sharded_tensors
from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner
from torch.distributed.checkpoint.metadata import STATE_DICT_TYPE, Metadata

__all__ = ['LoraPlanner']


class LoraPlanner(DefaultLoadPlanner):
    """Takes a Composer checkpoint and converts it to LoRA Checkpoint."""

    def set_up_planner(
        self,
        state_dict: STATE_DICT_TYPE,
        metadata: Metadata,
        is_coordinator: bool,
    ) -> None:
        """Sets up the planner for converting Composer to LoRA Checkpoint.

        Takes all targeted modules and checks whether they have been LoRA processed. If not,
        changes names of weights appropriately. If yes, doesn't change anything for autoresume
        compatibility.

        Args:
            state_dict (STATE_DICT_TYPE): Original torch state dict.
            metadata (METADATA): Any metadata associated with the state dict.
            is_coordinator (bool): Whether the machine this is running on is the coordinator of loading.
        """
        if 'state' not in state_dict:
            super().set_up_planner(state_dict, metadata, is_coordinator)
            return

        self.original_state_dict = state_dict

        state_dict = dict(state_dict.items())
        state_dict['state'] = dict(state_dict['state'].items())
        target_modules = ['to_k', 'to_v', 'to_q', 'to_out.0']

        for key in state_dict['state']['model'].keys():
            for mod in target_modules:
                if f'{mod}.weight' in key:
                    new_key = key.replace(mod, mod + '.base_layer')
                    state_dict['state']['model'][new_key] = state_dict['state']['model'].pop(key)
                    break

        if self.flatten_sharded_tensors:
            state_dict = _flatten_sharded_tensors(state_dict)

        if self.flatten_state_dict:
            state_dict, self.mappings = flatten_state_dict(state_dict)

        self.state_dict = state_dict
        self.metadata = metadata
        self.is_coordinator = is_coordinator
