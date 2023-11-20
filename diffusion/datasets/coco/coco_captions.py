# Copyright 2022 MosaicML Diffusion authors
# SPDX-License-Identifier: Apache-2.0

"""Streaming COCO dataset."""

import random
from typing import Optional

from streaming.base import StreamingDataset
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import CLIPTokenizer

from diffusion.datasets.laion.transforms import LargestCenterSquare


class StreamingCOCOCaption(StreamingDataset):
    """Streaming COCO dataset.

    Args:
        remote (str, optional): Remote directory (S3 or local filesystem) where dataset is stored.
            Default: ``None``.
        local (str, optional): Local filesystem directory where dataset is cached during operation.
            Default: ``None``.
        shuffle (bool): Whether to shuffle the samples in this dataset.
            Default: ``False``.
        shuffle_algo (str): What shuffle algorithm to use.
            Default: ``'py1s'``.
        shuffle_block_size (int): Unit of shuffling.
            Default: ``1 << 18``.
        batch_size (Optional[int]):  batch_size that will be used on each device's DataLoader.
            Default: ``None``.
        tokenizer_name_or_path (str): The name or path of the tokenizer to use.
            Default: ``'stabilityai/stable-diffusion-2-base'``.
        caption_selection (str): Which caption to use for the image. Must be 'random' or 'first'.
            Default: ``'first'``.
        download_timeout (Optional[float]): The timeout for a download.
            Default: ``120``.
        transform (Optional[Callable]): The transforms to apply to the image.
            Default: ``None``.
        num_canonical_nodes (int, optional): The number of canonical nodes for shuffle.
            Default: ``None``.

    """

    def __init__(
        self,
        remote,
        local,
        shuffle,
        shuffle_algo: str = 'py1s',
        shuffle_block_size: int = 1 << 18,
        batch_size: Optional[int] = None,
        tokenizer_name_or_path='stabilityai/stable-diffusion-2-base',
        caption_selection='first',
        download_timeout=120,
        transform=None,
        num_canonical_nodes: Optional[int] = None,
    ):
        super().__init__(
            remote=remote,
            local=local,
            shuffle=shuffle,
            shuffle_algo=shuffle_algo,
            shuffle_block_size=shuffle_block_size,
            batch_size=batch_size,
            download_timeout=download_timeout,
            num_canonical_nodes=num_canonical_nodes,
        )
        self.tokenizer = CLIPTokenizer.from_pretrained(tokenizer_name_or_path, subfolder='tokenizer')
        self.transform = transform
        self.caption_selection = caption_selection.lower()
        if caption_selection not in ['random', 'first']:
            raise ValueError(f'Invalid caption selection: {caption_selection}. Must be one of [random, first]')

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        image = sample['image']
        if image.mode != 'RGB':
            image = image.convert('RGB')

        if self.transform is not None:
            image = self.transform(image)

        if self.caption_selection == 'first':
            captions = sample['captions'][0]
        elif self.caption_selection == 'random':
            captions = random.sample(sample['captions'], k=1)[0]
        else:
            raise ValueError(f'Invalid caption selection: {self.caption_selection}. Must be one of [random, first].')
        captions = self.tokenizer(captions, padding='max_length', truncation=True, return_tensors='pt')['input_ids'][0]
        return {'image': image, 'captions': captions}


def build_streaming_cocoval_dataloader(
    batch_size: int,
    remote: str,
    local: str = '/tmp/mds-cache/mds-coco-val/',
    shuffle: bool = False,
    resize_size: int = 512,
    use_crop: bool = False,
    caption_selection='first',
    num_canonical_nodes: Optional[int] = None,
    **dataloader_kwargs,
):
    """Builds a streaming dataloader for the COCO validation set."""
    if use_crop:
        transform = transforms.Compose([LargestCenterSquare(resize_size), transforms.ToTensor()])
    else:
        transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((resize_size, resize_size))])

    dataset = StreamingCOCOCaption(
        remote=remote,
        local=local,
        shuffle=shuffle,
        batch_size=batch_size,
        caption_selection=caption_selection,
        transform=transform,
        num_canonical_nodes=num_canonical_nodes,
    )

    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, drop_last=False, **dataloader_kwargs)

    return dataloader
