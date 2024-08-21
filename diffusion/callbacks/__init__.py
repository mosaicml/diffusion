# Copyright 2022 MosaicML Diffusion authors
# SPDX-License-Identifier: Apache-2.0

"""Custom callbacks for Diffusion."""

from diffusion.callbacks.log_diffusion_images import LogAutoencoderImages, LogDiffusionImages
from diffusion.callbacks.log_latent_statistics import LogLatentStatistics
from diffusion.callbacks.nan_catcher import NaNCatcher
from diffusion.callbacks.scheduled_garbage_collector import ScheduledGarbageCollector
from diffusion.callbacks.assign_controlnet_weight import AssignControlNet

__all__ = [
    'AssignControlNet',
    'LogAutoencoderImages',
    'LogDiffusionImages',
    'LogLatentStatistics',
    'NaNCatcher',
    'ScheduledGarbageCollector',
]
