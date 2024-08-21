# Copyright 2022 MosaicML Diffusion authors
# SPDX-License-Identifier: Apache-2.0

"""Diffusion package setup."""

from setuptools import find_packages, setup

install_requires = [
    'mosaicml==0.20.1', 'mosaicml-streaming==0.7.4', 'hydra-core>=1.2', 'hydra-colorlog>=1.1.0',
    'diffusers[torch]==0.26.3', 'transformers[torch]==4.38.2', 'huggingface_hub==0.21.2', 'wandb==0.16.3',
    'xformers==0.0.23.post1', 'triton==2.1.0', 'torchmetrics[image]==1.3.1', 'lpips==0.1.4', 'clean-fid==0.1.35',
    'clip@git+https://github.com/openai/CLIP.git@a1d071733d7111c9c014f024669f959182114e33', 'gradio==4.19.2',
    'datasets==2.19.2', 'peft==0.12.0'
]

extras_require = {}

extras_require['dev'] = {
    'pre-commit>=2.18.1,<3',
    'pytest==7.3.0',
    'coverage[toml]==7.2.2',
    'pyarrow==14.0.1',
}

extras_require['all'] = {dep for deps in extras_require.values() for dep in deps}

setup(
    name='diffusion',
    version='0.0.1',
    packages=find_packages(),
    install_requires=install_requires,
    extras_require=extras_require,
)
