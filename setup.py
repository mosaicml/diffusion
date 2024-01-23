# Copyright 2022 MosaicML Diffusion authors
# SPDX-License-Identifier: Apache-2.0

"""Diffusion package setup."""

from setuptools import find_packages, setup

install_requires = [
    'composer==0.17.2',
    'mosaicml-streaming==0.7.3',
    'hydra-core>=1.2',
    'hydra-colorlog>=1.1.0',
    'diffusers[torch]==0.25.0',
    'transformers[torch]==4.36.2',
    'wandb==0.16.2',
    'xformers==0.0.23',
    'triton==2.1.0',
    'torchmetrics[image]==1.0.3',
    'lpips==0.1.4',
    'clean-fid==0.1.35',
    'clip@git+https://github.com/openai/CLIP.git',
    'gradio==4.14.0',
]

extras_require = {}

extras_require['dev'] = {
    'pre-commit>=2.18.1,<3',
    'pytest==7.3.0',
    'coverage[toml]==7.2.2',
    'pyarrow==14.0.1',
}

extras_require['all'] = set(dep for deps in extras_require.values() for dep in deps)

setup(
    name='diffusion',
    version='0.0.1',
    packages=find_packages(),
    install_requires=install_requires,
    extras_require=extras_require,
)
