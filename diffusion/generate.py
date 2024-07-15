# Copyright 2022 MosaicML Diffusion authors
# SPDX-License-Identifier: Apache-2.0

"""Generate images from a model."""

import operator
from typing import List

import hydra
from composer import Algorithm, ComposerModel
from composer.algorithms.low_precision_groupnorm import apply_low_precision_groupnorm
from composer.algorithms.low_precision_layernorm import apply_low_precision_layernorm
from composer.core import Precision
from composer.utils import reproducibility
from datasets import load_dataset
from omegaconf import DictConfig
from torch.utils.data import Dataset

from diffusion.evaluation.generate_images import ImageGenerator


def generate(config: DictConfig) -> None:
    """Evaluate a model.

    Args:
        config (DictConfig): Configuration composed by Hydra
    """
    reproducibility.seed_all(config.seed)

    # The model to evaluate
    if not config.hf_model:
        model: ComposerModel = hydra.utils.instantiate(config.model)
    else:
        model = config.model.name

    tokenizer = model.tokenizer if hasattr(model, 'tokenizer') else None

    # The dataset to use for evaluation

    if config.hf_model:
        dataset = load_dataset(config.dataset.name, split=config.dataset.split)
    elif tokenizer:
        dataset = hydra.utils.instantiate(config.dataset)

    else:
        dataset: Dataset = hydra.utils.instantiate(config.dataset)

    # Build list of algorithms.
    algorithms: List[Algorithm] = []

    # Some algorithms should also be applied at inference time
    if 'algorithms' in config:
        for ag_name, ag_conf in config.algorithms.items():
            if '_target_' in ag_conf:
                print(f'Instantiating algorithm <{ag_conf._target_}>')
                algorithms.append(hydra.utils.instantiate(ag_conf))
            elif ag_name == 'low_precision_groupnorm':
                surgery_target = model
                if 'attribute' in ag_conf:
                    surgery_target = operator.attrgetter(ag_conf.attribute)(model)
                apply_low_precision_groupnorm(
                    model=surgery_target,
                    precision=Precision(ag_conf['precision']),
                    optimizers=None,
                )
            elif ag_name == 'low_precision_layernorm':
                surgery_target = model
                if 'attribute' in ag_conf:
                    surgery_target = operator.attrgetter(ag_conf.attribute)(model)
                apply_low_precision_layernorm(
                    model=surgery_target,
                    precision=Precision(ag_conf['precision']),
                    optimizers=None,
                )

    image_generator: ImageGenerator = hydra.utils.instantiate(
        config.generator,
        model=model,
        dataset=dataset,
        hf_model=config.hf_model,
    )

    def generate_from_model():
        image_generator.generate()

    return generate_from_model()
