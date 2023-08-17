# Copyright 2022 MosaicML Diffusion authors
# SPDX-License-Identifier: Apache-2.0

"""Train model."""

import operator
import time
from collections.abc import Iterable
from typing import Any, Dict, List, Optional, Union

import hydra
from composer import Algorithm, Callback, ComposerModel, DataSpec, Evaluator, Trainer
from composer.algorithms.low_precision_groupnorm import apply_low_precision_groupnorm
from composer.algorithms.low_precision_layernorm import apply_low_precision_layernorm
from composer.core import Precision
from composer.loggers import LoggerDestination
from composer.utils import dist, reproducibility
from omegaconf import DictConfig, OmegaConf


def train(config: DictConfig) -> None:
    """Train a model.

    Args:
        config (DictConfig): Configuration composed by Hydra
    Returns:
        Optional[float]: Metric score for hyperparameter optimization
    """
    reproducibility.seed_all(config['seed'])

    model: ComposerModel = hydra.utils.instantiate(config.model)

    optimizer = hydra.utils.instantiate(config.optimizer, params=model.parameters())

    # Load train dataset. Currently this expects to load according to the datasetHparam method.
    # This means adding external datasets is currently not super easy. Will refactor or check for
    # upstream composer changes that could make this easier.
    train_dataloader: Union[Iterable, DataSpec, Dict[str, Any]] = hydra.utils.instantiate(
        config.dataset.train_dataset,
        batch_size=config.dataset.train_batch_size // dist.get_world_size(),
    )
    # Need to sleep for a bit to avoid dataloader crash
    time.sleep(10)

    # Composer can take dataloaders, dataspecs, evaluators, or list of evaluators
    eval_set: Optional[Union[DataSpec, List[Evaluator]]] = None

    # Assumes that evaluators is a nested dictionary with evalutor / dataloader pairs
    if 'evaluators' in config.dataset:
        evaluators = []
        for _, eval_conf in config.dataset.evaluators.items():
            print(OmegaConf.to_yaml(eval_conf))
            eval_dataloader = hydra.utils.instantiate(
                eval_conf.eval_dataset,
                config.dataset.eval_batch_size // dist.get_world_size(),
            )
            evaluator = hydra.utils.instantiate(eval_conf.evaluator, dataloader=eval_dataloader)
            # Need to sleep for a bit to avoid dataloader crash
            time.sleep(10)
            evaluators.append(evaluator)

        eval_set = evaluators

    else:
        eval_set = hydra.utils.instantiate(config.dataset.eval_dataset,
                                           batch_size=config.dataset.eval_batch_size // dist.get_world_size())
        # Need to sleep for a bit to avoid dataloader crash
        time.sleep(10)

    # Build list of loggers, callbacks, and algorithms to pass to trainer
    logger: List[LoggerDestination] = []
    callbacks: List[Callback] = []
    algorithms: List[Algorithm] = []

    if 'logger' in config:
        for log, lg_conf in config.logger.items():
            if '_target_' in lg_conf:
                print(f'Instantiating logger <{lg_conf._target_}>')
                if log == 'wandb':
                    container = OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
                    # use _partial_ so it doesn't try to init everything
                    wandb_logger = hydra.utils.instantiate(lg_conf, _partial_=True)
                    logger.append(wandb_logger(init_kwargs={'config': container}))
                else:
                    logger.append(hydra.utils.instantiate(lg_conf))

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
                    optimizers=optimizer,
                )
            elif ag_name == 'low_precision_layernorm':
                surgery_target = model
                if 'attribute' in ag_conf:
                    surgery_target = operator.attrgetter(ag_conf.attribute)(model)
                apply_low_precision_layernorm(
                    model=surgery_target,
                    precision=Precision(ag_conf['precision']),
                    optimizers=optimizer,
                )

    if 'callbacks' in config:
        for _, call_conf in config.callbacks.items():
            if '_target_' in call_conf:
                print(f'Instantiating callbacks <{call_conf._target_}>')
                callbacks.append(hydra.utils.instantiate(call_conf))

    scheduler = hydra.utils.instantiate(config.scheduler)

    trainer: Trainer = hydra.utils.instantiate(
        config.trainer,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_set,
        optimizers=optimizer,
        model=model,
        loggers=logger,
        algorithms=algorithms,
        schedulers=scheduler,
        callbacks=callbacks,
    )

    def eval_and_then_train():
        if config.get('eval_first', True):
            if hasattr(config.trainer, 'eval_subset_num_batches'):
                trainer.eval(subset_num_batches=config.trainer.eval_subset_num_batches)
            else:
                trainer.eval()
        trainer.fit()

    return eval_and_then_train()
