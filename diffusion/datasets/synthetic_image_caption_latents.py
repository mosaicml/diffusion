# Copyright 2022 MosaicML Diffusion authors
# SPDX-License-Identifier: Apache-2.0

"""Synthetic Image-Caption dataset."""

from typing import Dict, Optional

import torch
from composer.utils import dist
from torch.utils.data import DataLoader, Dataset


class SyntheticImageCaptionLatentsDataset(Dataset):
    """Synthetic dataset imitating a dataset containing image-caption pairs.

    Args:
        image_size (int): Size of the synthetic images. Default: ``512``.
        clip_length (int): Length of the synthetic clip embeddings. Default: ``77``.
        clip_dim (int): Dimension of the synthetic clip embeddings. Default: ``768``.
        t5_length (int): Length of the synthetic T5 embeddings. Default: ``512``.
        t5_dim (int): Dimension of the synthetic T5 embeddings. Default: ``4096``.
    """

    def __init__(self,
                 image_size: int = 512,
                 clip_length: int = 77,
                 clip_dim: int = 768,
                 t5_length: int = 512,
                 t5_dim: int = 4096):

        super().__init__()
        self.image_size = image_size
        self.clip_length = clip_length
        self.clip_dim = clip_dim
        self.t5_length = t5_length
        self.t5_dim = t5_dim

    def __len__(self):
        return 100_000

    def __getitem__(self, idx):
        out = {}
        out['cond_crops_coords_top_left'] = torch.tensor([0, 0], dtype=torch.float)
        out['cond_original_size'] = torch.tensor([self.image_size, self.image_size], dtype=torch.float)
        out['cond_target_size'] = torch.tensor([self.image_size, self.image_size], dtype=torch.float)
        out['image'] = torch.randn(3, self.image_size, self.image_size)
        out['CLIP_LATENTS'] = torch.randn(self.clip_length, self.clip_dim, dtype=torch.float)
        out['CLIP_POOLED'] = torch.randn(self.clip_dim, dtype=torch.float)
        out['CLIP_ATTENTION_MASK'] = torch.ones(self.clip_length)
        out['T5_LATENTS'] = torch.randn(self.t5_length, self.t5_dim, dtype=torch.float)
        out['T5_ATTENTION_MASK'] = torch.ones(self.t5_length)
        return out


def build_synthetic_image_caption_latents_dataloader(
    batch_size: int,
    image_size: int = 512,
    clip_length: int = 77,
    clip_dim: int = 768,
    t5_length: int = 512,
    t5_dim: int = 4096,
    dataloader_kwargs: Optional[Dict] = None,
):
    """Builds a dataloader for the synthetic image-caption dataset.

    Args:
        batch_size (int): Batch size for the dataloader.
        image_size (int): Size of the synthetic images. Default: ``512``.
        clip_length (int): Length of the synthetic clip embeddings. Default: ``77``.
        clip_dim (int): Dimension of the synthetic clip embeddings. Default: ``768``.
        t5_length (int): Length of the synthetic T5 embeddings. Default: ``512``.
        t5_dim (int): Dimension of the synthetic T5 embeddings. Default: ``4096``.
        dataloader_kwargs (optional, dict): Additional arguments to pass to the dataloader. Default ``None``.
    """
    if dataloader_kwargs is None:
        dataloader_kwargs = {}

    dataset = SyntheticImageCaptionLatentsDataset(image_size=image_size,
                                                  clip_length=clip_length,
                                                  clip_dim=clip_dim,
                                                  t5_length=t5_length,
                                                  t5_dim=t5_dim)

    dataloader = DataLoader(
        dataset=dataset,
        sampler=dist.get_sampler(dataset),
        batch_size=batch_size,
        **dataloader_kwargs,
    )

    return dataloader
