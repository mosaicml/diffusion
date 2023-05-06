# Copyright 2022 MosaicML Diffusion authors
# SPDX-License-Identifier: Apache-2.0

"""Run training."""

import textwrap

import hydra
from omegaconf import DictConfig

from diffusion.train import train


@hydra.main(version_base=None)
def main(config: DictConfig) -> None:
    """Hydra wrapper for train."""
    if not config:
        raise ValueError(
            textwrap.dedent("""\
                            Config path and name not specified!
                            Please specify these by using --config-path and --config-name, respectively."""))
    return train(config)


if __name__ == '__main__':
    main()
