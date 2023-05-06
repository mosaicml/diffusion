# Copyright 2022 MosaicML Diffusion authors
# SPDX-License-Identifier: Apache-2.0

"""Streaming LAION dataset."""

from io import BytesIO
from typing import Callable, List, Optional, Sequence, Union

import numpy as np
import torch
from PIL import Image
from streaming import Stream, StreamingDataset
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import CLIPTokenizer

from diffusion.datasets.laion.transforms import LargestCenterSquare

# Disable PIL max image size limit
Image.MAX_IMAGE_PIXELS = None


class StreamingLAIONDataset(StreamingDataset):
    """Implementation of the LAION dataset as a streaming dataset.

    Args:
        streams (Sequence[Stream], optional): One or more Streams to stream/cache samples from. StreamingLAIONDataset
            uses either ``streams`` or ``remote``/``local``. Default:``None``.
        remote (str, optional): Remote directory (S3 or local filesystem) where dataset is stored. Default: ``None``.
        local (str, optional): Local filesystem directory where dataset is cached during operation. Default: ``None``.
        split (str, optional): The dataset split to use. Currently, only ``None`` is supported. Default: ``None``.
        shuffle (bool): Whether to shuffle the samples in this dataset. Default: ``False``.
        tokenizer_name_or_path (str): The name or path of the tokenizer to use. Default: ``'stabilityai/stable-diffusion-2-base'``.
        transform (Optional[Callable]): The transforms to apply to the image. Default: ``None``.
        predownload (Optional[int]): The number of samples to prefetch. Default: ``100_000``.
        download_retry (Optional[int]): The number of times to retry a download. Default: ``2``.
        download_timeout (Optional[float]): The timeout for a download. Default: ``120``.
        batch_size (Optional[int]): Hint batch_size that will be used on each device's DataLoader. Default: ``None``.
        image_size (Optional[int]): The size to resize the image to. Default: ``None``.
        num_canonical_nodes (int, optional): The number of canonical nodes for shuffle. Default: ``None``.
    """

    def __init__(
        self,
        streams: Optional[Sequence[Stream]] = None,
        remote: Optional[str] = None,
        local: Optional[str] = None,
        split: Optional[str] = None,
        shuffle: Optional[bool] = False,
        tokenizer_name_or_path: Optional[str] = 'stabilityai/stable-diffusion-2-base',
        caption_drop_prob: Optional[float] = 0.0,
        transform: Optional[Callable] = None,
        predownload: Optional[int] = 100_000,
        download_retry: Optional[int] = 2,
        download_timeout: Optional[float] = 120,
        batch_size: Optional[int] = None,
        image_size: Optional[int] = None,
        num_canonical_nodes: Optional[int] = None,
    ) -> None:

        super().__init__(
            streams=streams,
            remote=remote,
            local=local,
            split=split,
            shuffle=shuffle,
            predownload=predownload,
            keep_zip=False,
            download_retry=download_retry,
            download_timeout=download_timeout,
            validate_hash=None,
            batch_size=batch_size,
            num_canonical_nodes=num_canonical_nodes,
        )

        self.transform = transform
        self.tokenizer = CLIPTokenizer.from_pretrained(tokenizer_name_or_path, subfolder='tokenizer')
        self.caption_drop_prob = caption_drop_prob
        self.image_size = image_size

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        img = Image.open(BytesIO(sample['jpg']))
        if img.mode != 'RGB':
            img = img.convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        # Drop the caption with probability `caption_drop_prob`
        if torch.rand(1) < self.caption_drop_prob:
            caption = ''
        else:
            caption = sample['caption']
        tokenized_caption = self.tokenizer(
            caption,
            padding='max_length',
            max_length=self.tokenizer.model_max_length,
            truncation=True,
        )['input_ids']
        tokenized_caption = torch.tensor(tokenized_caption)
        out = {'image': img, 'captions': tokenized_caption}
        if 'caption_latents' in sample:
            out['caption_latents'] = torch.from_numpy(
                np.frombuffer(sample['caption_latents'], dtype=np.float16).copy()).reshape(77, 1024)
        if self.image_size == 256 and 'latents_256' in sample:
            out['image_latents'] = torch.from_numpy(np.frombuffer(sample['latents_256'],
                                                                  dtype=np.float16).copy()).reshape(4, 32, 32)
        if self.image_size == 512 and 'latents_512' in sample:
            out['image_latents'] = torch.from_numpy(np.frombuffer(sample['latents_512'],
                                                                  dtype=np.float16).copy()).reshape(4, 64, 64)
        return out


def build_streaming_laion_dataloader(
    remote: Union[str, List],
    local: Union[str, List],
    batch_size: int,
    tokenizer_name_or_path: str = 'stabilityai/stable-diffusion-2-base',
    caption_drop_prob: float = 0.0,
    resize_size: int = 256,
    num_samples: Optional[int] = None,
    predownload: Optional[int] = 100_000,
    download_retry: Optional[int] = 2,
    download_timeout: Optional[float] = 120,
    drop_last: bool = True,
    shuffle: bool = True,
    num_canonical_nodes: Optional[int] = None,
    **dataloader_kwargs,
):
    """Builds a streaming LAION dataloader.

    Args:
        remote (str, Sequence[str]): One or more remote directories (S3 or local filesystem) where dataset is stored.
        local (str, Sequence[str]): One or more local filesystem directories where dataset is cached during operation.
        batch_size (int): The batch size to use.
        tokenizer_name_or_path (str): The name or path of the tokenizer to use. Default: ``'stabilityai/stable-diffusion-2-base'``.
        caption_drop_prob (float): The probability of dropping a caption. Default: ``0.0``.
        resize_size (int): The size to resize the image to. Default: ``256``.
        num_samples (Optional[int]): The number of samples to use. Default: ``None`` uses all available samples.
        predownload (Optional[int]): The number of samples to prefetch. Default: ``100_000``.
        download_retry (Optional[int]): The number of times to retry a download. Default: ``2``.
        download_timeout (Optional[float]): The timeout for a download. Default: ``120``.
        drop_last (bool): Whether to drop the last batch if it is incomplete. Default: ``True``.
        shuffle (bool): Whether to shuffle the samples in this dataset. Default: ``True``.
        num_canonical_nodes (int, optional): The number of canonical nodes for shuffle. Default: ``None``.
        **dataloader_kwargs: Additional arguments to pass to the dataloader.
    """
    if isinstance(remote, str) or isinstance(local, str):
        assert isinstance(remote, str) and isinstance(
            local, str), 'If either remote or local is a single string, both must be single strings'
        # Hacky... make remote and local lists to simplify downstream code
        remote, local = [
            remote,
        ], [
            local,
        ]
    elif isinstance(remote, Sequence) or isinstance(local, Sequence):
        assert isinstance(remote, Sequence) and isinstance(
            local, Sequence), 'If either remote or local is a sequence, both must be sequences'
        assert len(remote) == len(
            local), f'remote and local must be lists of the same length, got lengths {len(remote)} and {len(local)}'
    else:
        ValueError('remote and local must both be a single string or a Sequence of strings.')

    # Create a Stream for each (remote, local) pair
    streams = []
    for r, l in zip(remote, local):
        streams.append(Stream(remote=r, local=l, download_retry=download_retry, download_timeout=download_timeout))

    center_square_crop = LargestCenterSquare(resize_size)
    # Normalize from 0 to 1 to -1 to 1
    normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    transform = transforms.Compose([center_square_crop, transforms.ToTensor(), normalize])
    dataset = StreamingLAIONDataset(
        streams=streams,
        split=None,
        shuffle=shuffle,
        tokenizer_name_or_path=tokenizer_name_or_path,
        caption_drop_prob=caption_drop_prob,
        transform=transform,
        predownload=predownload,
        download_retry=download_retry,
        download_timeout=download_timeout,
        batch_size=batch_size,
        image_size=resize_size,
        num_canonical_nodes=num_canonical_nodes,
    )
    # Create a subset of the dataset
    if num_samples is not None:
        dataset = torch.utils.data.Subset(dataset, range(num_samples))  # type: ignore

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        sampler=None,
        drop_last=drop_last,
        **dataloader_kwargs,
    )

    return dataloader
