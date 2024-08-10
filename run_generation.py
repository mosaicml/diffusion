# Copyright 2022 MosaicML Diffusion authors
# SPDX-License-Identifier: Apache-2.0

"""Run image generation."""

import textwrap

import hydra
from omegaconf import DictConfig

from diffusion.generate import generate


@hydra.main(version_base=None)
def main(config: DictConfig) -> None:
    """Hydra wrapper for evaluation."""
    if not config:
        raise ValueError(
            textwrap.dedent("""\
                            Config path and name not specified!
                            Please specify these by using --config-path and --config-name, respectively."""))
    return generate(config)


if __name__ == '__main__':
    main()
