# Copyright 2022 MosaicML Diffusion authors
# SPDX-License-Identifier: Apache-2.0

"""Train model."""

import operator
import time
from collections.abc import Iterable
from itertools import chain
from typing import Any, Dict, List, Optional, Union

import hydra
from composer import Algorithm, Callback, ComposerModel, DataSpec, Evaluator, Trainer
from composer.algorithms.low_precision_groupnorm import apply_low_precision_groupnorm
from composer.algorithms.low_precision_layernorm import apply_low_precision_layernorm
from composer.core import Precision
from composer.loggers import LoggerDestination
from composer.utils import dist, reproducibility
from omegaconf import DictConfig, OmegaConf
from torch.optim import Optimizer

from diffusion.models.autoencoder import ComposerAutoEncoder, ComposerDiffusersAutoEncoder
from diffusion.models.t2i_transformer import ComposerPrecomputedTextLatentsToImageMMDiT, ComposerTextToImageMMDiT
from diffusion.models.transformer import MuOutputLinear


def make_autoencoder_optimizer(config: DictConfig, model: ComposerModel) -> Optimizer:
    """Configures the optimizer for use with an autoencoder + discriminator loss."""
    print('Configuring opimizer for autoencoder+discriminator')
    assert isinstance(model, (ComposerAutoEncoder, ComposerDiffusersAutoEncoder))

    # Configure optimizer settings for the autoencoder
    if hasattr(config, 'autoencoder_optimizer'):
        autoencoder_param_dict = dict(config.autoencoder_optimizer.items())
    else:
        autoencoder_param_dict = dict(config.optimizer.items())

    if model.autoencoder_loss.learn_log_var:
        autoencoder_param_dict['params'] = chain(model.model.parameters(), [model.autoencoder_loss.log_var])
    else:
        autoencoder_param_dict['params'] = model.model.parameters()

    # Configure optimizer settings for the discriminator
    if hasattr(config, 'discriminator_optimizer'):
        discriminator_param_dict = dict(config.discriminator_optimizer.items())
    else:
        discriminator_param_dict = dict(config.optimizer.items())
    discriminator_param_dict['params'] = model.autoencoder_loss.discriminator.parameters()

    params = [autoencoder_param_dict, discriminator_param_dict]
    optimizer = hydra.utils.instantiate(config.optimizer, params)
    return optimizer


def make_transformer_optimizer(config: DictConfig, model: ComposerModel) -> Optimizer:
    """Configures the optimizer for use with a transformer model."""
    print('Configuring optimizer for transformer')
    assert isinstance(model, (ComposerTextToImageMMDiT, ComposerPrecomputedTextLatentsToImageMMDiT))
    # Grab the width scaling factor from the model if it's been given
    if hasattr(model, 'width_scale'):
        width_scale = model.width_scale
    else:
        width_scale = 1.0

    # Turn off weight decay for biases, norms, and positional embeddings.
    # Also set up learning rates for mu-parameterization
    no_decay = ['bias', 'norm', 'position_embedding']
    params_with_no_decay = []
    params_with_decay = []
    mu_input_params = []
    mu_hidden_params = []
    mu_output_params = []
    for name, param in model.named_parameters():
        if 'mu_input_linear.weight' in name:
            mu_input_params.append(param)
        elif 'mu_hidden_linear.weight' in name:
            mu_hidden_params.append(param)
        elif 'mu_output_linear.weight' in name:
            mu_output_params.append(param)
        elif any(nd in name for nd in no_decay):
            params_with_no_decay.append(param)
        else:
            params_with_decay.append(param)
    no_decay_dict = dict(config.optimizer.items())
    no_decay_dict['params'] = params_with_no_decay
    no_decay_dict['weight_decay'] = 0.0

    decay_dict = dict(config.optimizer.items())
    decay_dict['params'] = params_with_decay

    mu_input_dict = dict(config.optimizer.items())
    mu_input_dict['params'] = mu_input_params

    mu_hidden_dict = dict(config.optimizer.items())
    mu_hidden_dict['params'] = mu_hidden_params
    mu_hidden_dict['lr'] *= 1 / width_scale

    mu_output_dict = dict(config.optimizer.items())
    mu_output_dict['params'] = mu_output_params
    mu_output_dict['lr'] *= 1 / width_scale

    # Rescaling of output inits
    for module in model.modules():
        if isinstance(module, MuOutputLinear):
            module.rescale_init(width_scale)

    optimizer = hydra.utils.instantiate(config.optimizer,
                                        [no_decay_dict, decay_dict, mu_input_dict, mu_hidden_dict, mu_output_dict])
    return optimizer


def train(config: DictConfig) -> None:
    """Train a model.

    Args:
        config (DictConfig): Configuration composed by Hydra
    Returns:
        Optional[float]: Metric score for hyperparameter optimization
    """
    reproducibility.seed_all(config['seed'])

    model: ComposerModel = hydra.utils.instantiate(config.model)

    # If the model has a tokenizer, we'll need it for the dataset
    if hasattr(model, 'tokenizer'):
        tokenizer = model.tokenizer
    else:
        tokenizer = None

    if hasattr(model, 'autoencoder_loss'):
        # Check if this is training an autoencoder. If so, the optimizer needs different param groups
        optimizer = make_autoencoder_optimizer(config, model)
    elif isinstance(model, (ComposerTextToImageMMDiT, ComposerPrecomputedTextLatentsToImageMMDiT)):
        # Check if this is training a transformer. If so, the optimizer needs different param groups
        optimizer = make_transformer_optimizer(config, model)
    else:
        optimizer = hydra.utils.instantiate(config.optimizer, params=model.parameters())

    # Load train dataset. Currently this expects to load according to the datasetHparam method.
    # This means adding external datasets is currently not super easy. Will refactor or check for
    # upstream composer changes that could make this easier.
    if tokenizer:
        train_dataloader: Union[Iterable, DataSpec, Dict[str, Any]] = hydra.utils.instantiate(
            config.dataset.train_dataset,
            tokenizer=tokenizer,
            batch_size=config.dataset.train_batch_size // dist.get_world_size(),
        )
    else:
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
        for eval_conf in config.dataset.evaluators.values():
            print(OmegaConf.to_yaml(eval_conf))
            if tokenizer:
                eval_dataloader = hydra.utils.instantiate(
                    eval_conf.eval_dataset,
                    tokenizer=tokenizer,
                    batch_size=config.dataset.eval_batch_size // dist.get_world_size(),
                )
            else:
                eval_dataloader = hydra.utils.instantiate(
                    eval_conf.eval_dataset,
                    batch_size=config.dataset.eval_batch_size // dist.get_world_size(),
                )

            evaluator = hydra.utils.instantiate(eval_conf.evaluator, dataloader=eval_dataloader)
            # Need to sleep for a bit to avoid dataloader crash
            time.sleep(10)
            evaluators.append(evaluator)

        eval_set = evaluators

    else:
        if tokenizer:
            eval_set = hydra.utils.instantiate(config.dataset.eval_dataset,
                                               tokenizer=model.tokenizer,
                                               batch_size=config.dataset.eval_batch_size // dist.get_world_size())
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
        for call_conf in config.callbacks.values():
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
