# Copyright 2022 MosaicML Diffusion authors
# SPDX-License-Identifier: Apache-2.0

"""COCO."""

from diffusion.datasets.coco.coco_captions import StreamingCOCOCaption, build_streaming_cocoval_dataloader

__all__ = [
    'build_streaming_cocoval_dataloader',
    'StreamingCOCOCaption',
]
