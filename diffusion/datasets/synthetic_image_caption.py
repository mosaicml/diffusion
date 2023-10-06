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

    def __init__(self, image_size: int = 512, caption_length: int = 77, num_samples: int = 100_000, sdxl: bool = False):

        super().__init__()
        self.image_size = image_size
        self.num_samples = num_samples
        self.sdxl = sdxl

        self.images = torch.randn(num_samples, 3, image_size, image_size)
        caption_shape = (num_samples, 2, caption_length) if self.sdxl else (num_samples, caption_length)
        self.captions = torch.randint(0, 128, caption_shape, dtype=torch.long)


    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        out = {}
        if self.sdxl:
            out['cond_crops_coords_top_left'] = torch.tensor([0, 0])
            out['cond_original_size'] = torch.tensor([self.image_size, self.image_size])
            out['cond_target_size'] = torch.tensor([self.image_size, self.image_size])
        out['image'] = self.image[idx]
        out['captions'] = self.captions[idx]
        return out


def build_synthetic_image_caption_dataloader(
    batch_size: int,
    image_size: int = 512,
    caption_length: int = 77,
    num_samples: int = 100_000,
    sdxl: bool = False,
    dataloader_kwargs: Optional[Dict] = None,
):
    """Builds a dataloader for the synthetic image-caption dataset.

    Args:
        batch_size (int): Batch size for the dataloader.
        image_size (int): Size of the synthetic images. Default: ``512``.
        caption_length (int): Length of the synthetic captions. Default: ``77``.
        num_samples (int): Number of samples in the synthetic dataset. Default: ``100_000``.
        dataloader_kwargs (optional, dict): Additional arguments to pass to the dataloader. Default ``None``.
    """
    if dataloader_kwargs is None:
        dataloader_kwargs = {}

    dataset = SyntheticImageCaptionDataset(
        image_size=image_size,
        caption_length=caption_length,
        num_samples=num_samples,
        sdxl=sdxl,
    )

    dataloader = DataLoader(
        dataset=dataset,
        sampler=dist.get_sampler(dataset),
        batch_size=batch_size,
        **dataloader_kwargs,
    )

    return dataloader
