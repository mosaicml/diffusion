# Copyright 2022 MosaicML Diffusion authors
# SPDX-License-Identifier: Apache-2.0

"""Composer algorithms."""

from diffusion.algorithms.discriminator_schedule import DiscriminatorSchedule
from diffusion.algorithms.ema import EMA

__all__ = ['EMA', 'DiscriminatorSchedule']
