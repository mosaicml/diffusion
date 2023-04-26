# Copyright 2022 MosaicML Diffusion authors
# SPDX-License-Identifier: Apache-2.0

"""Run training."""

import argparse

import hydra
from omegaconf import DictConfig

from diffusion.train import train

parser = argparse.ArgumentParser()
parser.add_argument('--config_path', type=str, default='/mnt/config')
parser.add_argument('--config_name', type=str, default='parameters')
args = parser.parse_args()


@hydra.main(version_base=None, config_path=args.config_path, config_name=args.config_name)
def main(config: DictConfig) -> None:
    """Hydra wrapper for train."""
    return train(config)


if __name__ == '__main__':
    main()
