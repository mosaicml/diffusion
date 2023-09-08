# Copyright 2022 MosaicML Diffusion authors
# SPDX-License-Identifier: Apache-2.0

"""Synthetic Image-Caption dataset."""

from typing import Dict, Optional

import torch
from composer.utils import dist
from torch.utils.data import DataLoader, Dataset


class SyntheticImageCaptionDataset(Dataset):
    """Synthetic dataset imitating a dataset containing image-caption pairs.

    Args:
        image_size (int): Size of the synthetic images. Default: ``512``.
        caption_length (int): Length of the synthetic captions. Default: ``77``.
        num_samples (int): Number of samples in the synthetic dataset. Default: ``100_000``.
    """

    def __init__(self, image_size: int = 512, caption_length: int = 77, num_samples: int = 100_000):
        
        super().__init__()
        self.num_samples = num_samples
        self.images = torch.randn(num_samples, 3, image_size, image_size)
        self.captions = torch.randint(0, 128, (num_samples, caption_length), dtype=torch.long)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return {'image': self.images[idx], 'captions': self.captions[idx]}


def build_synthetic_image_caption_dataloader(
    batch_size: int,
    image_size: int = 512,
    caption_length: int = 77,
    num_samples: int = 100_000,
    dataloader_kwargs: Optional[Dict] = None,
):
    """Builds a dataloader for the Synthetic Image-Caption dataset.

    Args:
        batch_size (int): Batch size for the dataloader.
        image_size (int): Size of the synthetic images. Default: ``512``.
        caption_length (int): Length of the synthetic captions. Default: ``77``.
        num_samples (int): Number of samples in the synthetic dataset. Default: ``100_000``.
        **dataloader_kwargs: Additional arguments to pass to the dataloader.
    """

    if dataloader_kwargs is None:
        dataloader_kwargs = {}

    dataset = SyntheticImageCaptionDataset(
        image_size=image_size,
        caption_length=caption_length,
        num_samples=num_samples,
    )

    dataloader = DataLoader(
        dataset=dataset,
        sampler=dist.get_sampler(dataset),
        batch_size=batch_size,
        **dataloader_kwargs,
    )

    return dataloader
