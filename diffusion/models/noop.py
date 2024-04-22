# Copyright 2022 MosaicML Diffusion authors
# SPDX-License-Identifier: Apache-2.0

"""NoOpModel algorithm and class."""

from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torchmetrics import Metric
from diffusion.models.text_encoder import MultiTokenizer

from composer.models.base import ComposerModel

class NoOpModel(ComposerModel):
    """No-op model used to measure dataloader throughput.

    Args:
        tokenizer_names (str, Tuple[str, ...]): HuggingFace name(s) of the tokenizer(s) to load.
            Default: ``('stabilityai/stable-diffusion-xl-base-1.0/tokenizer',
            'stabilityai/stable-diffusion-xl-base-1.0/tokenizer_2')``.
    """

    def __init__(
            self,
            tokenizer_names: Union[str, Tuple[str, ...]] = ('stabilityai/stable-diffusion-xl-base-1.0/tokenizer',
                                                            'stabilityai/stable-diffusion-xl-base-1.0/tokenizer_2'),
        ):
        super().__init__()
        self.weight = torch.nn.Linear(in_features=1, out_features=16)
        self.tokenizer = MultiTokenizer(tokenizer_names_or_paths=tokenizer_names)

    def loss(self, outputs: torch.Tensor, batch):
        y = torch.randn_like(self.weight.weight)
        return F.mse_loss(outputs, y)

    def forward(self, batch):
        input = torch.randn_like(self.weight.weight).sum().unsqueeze(0)
        return self.weight(input)

    def get_metrics(self, is_train: bool) -> Dict[str, Metric]:
        return {}

    def eval_forward(self, batch, outputs: Optional[Any] = None):
        return self.forward(batch)

    def update_metric(self, batch: Any, outputs: Any, metric: Metric) -> None:
        pass
