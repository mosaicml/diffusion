# Copyright 2022 MosaicML Diffusion authors
# SPDX-License-Identifier: Apache-2.0

"""Synthetic LAION dataset."""

import torch
from torch.utils.data import DataLoader, Dataset


class SyntheticImageCaptionDataset(Dataset):
    """Synthetic dataset imitating a dataset of images plus captions."""

    def __init__(self, image_size: int = 512, caption_length: int = 77, num_samples: int = 100_000):
        """Synthetic LAION dataset for testing.

        Args:
            image_size (int): Size of the synthetic images. Default: ``512``.
            caption_length (int): Length of the synthetic captions. Default: ``77``.
            num_samples (int): Number of samples in the synthetic dataset. Default: ``100_000``.
        """
        self.num_samples = num_samples
        self.images = torch.randn(num_samples, 3, image_size, image_size)
        self.captions = torch.randint(0, 128, (num_samples, caption_length), dtype=torch.long)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return {'image': self.images[idx], 'caption': self.captions[idx]}


def build_synthetic_laion_dataloader(
    batch_size: int,
    image_size: int = 512,
    caption_length: int = 77,
    num_samples: int = 100_000,
    **dataloader_kwargs,
):
    """Builds a dataloader for the Synthetic LAION dataset.

    Args:
        batch_size (int): Batch size for the dataloader.
        image_size (int): Size of the synthetic images. Default: ``512``.
        caption_length (int): Length of the synthetic captions. Default: ``77``.
        num_samples (int): Number of samples in the synthetic dataset. Default: ``100_000``.
        **dataloader_kwargs: Additional arguments to pass to the dataloader.
    """
    dataset = SyntheticImageCaptionDataset(
        image_size=image_size,
        caption_length=caption_length,
        num_samples=num_samples,
    )

    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, **dataloader_kwargs)
    return dataloader
