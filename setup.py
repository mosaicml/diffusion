# Copyright 2022 MosaicML Diffusion authors
# SPDX-License-Identifier: Apache-2.0

"""Diffusion package setup."""

from setuptools import find_packages, setup

install_requires = [
    'mosaicml',
    'mosaicml-streaming>=0.4.0,<1.0',
    'hydra-core>=1.2',
    'hydra-colorlog>=1.1.0',
    'diffusers[torch]==0.19.0',
    'transformers[torch]',
    'wandb==0.15.4',
    'xformers',
    'triton',
    'torchmetrics[image]',
    'clean-fid',
    'clip@git+https://github.com/openai/CLIP.git',
]

extras_require = {}

extras_require['dev'] = {
    'pre-commit>=2.18.1,<3',
    'pytest==7.3.0',
    'coverage[toml]==7.2.2',
    'pyarrow==11.0.0',
}

extras_require['all'] = set(dep for deps in extras_require.values() for dep in deps)

setup(
    name='diffusion',
    version='0.0.1',
    packages=find_packages(),
    install_requires=install_requires,
    extras_require=extras_require,
)
