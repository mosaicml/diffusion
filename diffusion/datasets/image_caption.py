# Copyright 2022 MosaicML Diffusion authors
# SPDX-License-Identifier: Apache-2.0

"""Streaming Image-Caption dataset."""

import random
from io import BytesIO
from typing import Callable, List, Optional, Sequence, Union

import torch
from PIL import Image
from streaming import Stream, StreamingDataset
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import AutoTokenizer

from diffusion.datasets.laion.transforms import LargestCenterSquare

# Disable PIL max image size limit
Image.MAX_IMAGE_PIXELS = None


class StreamingImageCaptionDataset(StreamingDataset):
    """Streaming dataset for image-caption pairs.

    Args:
        streams (Sequence[Stream], optional): One or more Streams to stream/cache samples from.
            ``StreamingImageCaptionDataset`` uses either ``streams`` or ``remote``/``local``. Default:``None``.
        remote (str, optional): Remote directory (S3 or local filesystem) where dataset is stored. Default: ``None``.
        local (str, optional): Local filesystem directory where dataset is cached during operation. Default: ``None``.
        split (str, optional): The dataset split to use. Currently, only ``None`` is supported. Default: ``None``.
        shuffle (bool): Whether to shuffle the samples in this dataset. Default: ``False``.
        tokenizer_name_or_path (str): The name or path of the tokenizer to use. Default: ``'stabilityai/stable-diffusion-2-base'``.
        caption_selection (str): If there are multiple captions, specifies how to select a single caption.
            'first' selects the first caption in the list and 'random' selects a random caption in the list.
            If there is only one caption, this argument is ignored. Default: ``'first'``.
        transform (Optional[Callable]): The transforms to apply to the image. Default: ``None``.
        image_size (Optional[int]): The size to resize the image to. Default: ``None``.
        image_key (str): Key associated with the image in the streaming dataset. Default: ``'image'``.
        caption_key (str): Key associated with the caption in the streaming dataset. Default: ``'caption'``.
        **streaming_kwargs: Additional arguments to pass in the construction of the StreamingDataloader
    """

    def __init__(
        self,
        streams: Optional[Sequence[Stream]] = None,
        remote: Optional[str] = None,
        local: Optional[str] = None,
        split: Optional[str] = None,
        shuffle: bool = False,
        tokenizer_name_or_path: str = 'stabilityai/stable-diffusion-2-base',
        caption_drop_prob: float = 0.0,
        caption_selection: str = 'first',
        transform: Optional[Callable] = None,
        image_size: Optional[int] = None,
        image_key: str = 'image',
        caption_key: str = 'caption',
        **streaming_kwargs,
    ) -> None:

        super().__init__(
            streams=streams,
            remote=remote,
            local=local,
            split=split,
            shuffle=shuffle,
            **streaming_kwargs,
        )

        if caption_selection not in ['first', 'random']:
            raise ValueError(f'Invalid caption selection: {caption_selection}. Must be one of [random, first]')

        self.transform = transform
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, subfolder='tokenizer')
        self.caption_drop_prob = caption_drop_prob
        self.caption_selection = caption_selection.lower()
        self.image_size = image_size
        self.image_key = image_key
        self.caption_key = caption_key

    def __getitem__(self, index):
        sample = super().__getitem__(index)

        # Image
        img = Image.open(BytesIO(sample[self.image_key]))
        if img.mode != 'RGB':
            img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        # Caption
        if torch.rand(1) < self.caption_drop_prob:
            caption = ''
        else:
            caption = sample[self.caption_key]
            if isinstance(caption, List) and self.caption_selection == 'first':
                caption = caption[0]
            if isinstance(caption, List) and self.caption_selection == 'random':
                caption = random.sample(caption, k=1)[0]
        tokenized_caption = self.tokenizer(caption,
                                           padding='max_length',
                                           max_length=self.tokenizer.model_max_length,
                                           truncation=True,
                                           return_tensor='pt')['input_ids'][0]

        return {'image': img, 'captions': tokenized_caption}


def build_streaming_image_caption_dataloader(
    remote: Union[str, List],
    local: Union[str, List],
    batch_size: int,
    tokenizer_name_or_path: str = 'stabilityai/stable-diffusion-2-base',
    caption_drop_prob: float = 0.0,
    resize_size: int = 256,
    num_samples: Optional[int] = None,
    predownload: Optional[int] = 100_000,
    download_retry: int = 2,
    download_timeout: float = 120,
    drop_last: bool = True,
    shuffle: bool = True,
    num_canonical_nodes: Optional[int] = None,
    caption_selection: str = 'first',
    transform: Optional[List[torch.nn.Module]] = None,
    image_key: str = 'image',
    caption_key: str = 'caption',
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
        num_samples (int, optional): The number of samples to use. Default: ``None`` uses all available samples.
        predownload (int, optional): The number of samples to prefetch. Default: ``100_000``.
        download_retry (int): The number of times to retry a download. Default: ``2``.
        download_timeout (float): The timeout for a download. Default: ``120``.
        drop_last (bool): Whether to drop the last batch if it is incomplete. Default: ``True``.
        shuffle (bool): Whether to shuffle the samples in this dataset. Default: ``True``.
        num_canonical_nodes (int, optional): The number of canonical nodes for shuffle. Default: ``None``.
        caption_selection (str): If there are multiple captions, specifies how to select a single caption.
            'first' selects the first caption in the list and 'random' selects a random caption in the list.
            If there is only one caption, this argument is ignored. Default: ``'first'``.
        transform (Optional[Callable]): The transforms to apply to the image. Default: ``None``.
        image_key (str): Key associated with the image in the streaming dataset. Default: ``'image'``.
        caption_key (str): Key associated with the caption in the streaming dataset. Default: ``'caption'``.
        is_precomputed_latents (bool): Whether or not to use the precomputed latents streaming dataset.
            Default: ``False``
        **dataloader_kwargs: Additional arguments to pass to the dataloader.
    """
    if isinstance(remote, str) and isinstance(local, str):
        # Hacky... make remote and local lists to simplify downstream code
        remote, local = [remote], [local]
    elif isinstance(remote, Sequence) and isinstance(local, Sequence):
        if len(remote) != len(local):
            ValueError(
                f'remote and local Sequences must be the same length, got lengths {len(remote)} and {len(local)}')
    else:
        ValueError(f'remote and local must be both Strings or Sequences, got types {type(remote)} and {type(local)}.')

    # Create a Stream for each (remote, local) pair
    streams = []
    for r, l in zip(remote, local):
        streams.append(Stream(remote=r, local=l, download_retry=download_retry, download_timeout=download_timeout))

    # Setup the transforms to apply
    if transform is None:
        transform = [
            LargestCenterSquare(resize_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # # Normalize from 0 to 1 to -1 to 1
        ]
    transform = transforms.Compose(transform)

    dataset = StreamingImageCaptionDataset(
        streams=streams,
        split=None,
        shuffle=shuffle,
        tokenizer_name_or_path=tokenizer_name_or_path,
        caption_drop_prob=caption_drop_prob,
        caption_selection=caption_selection,
        transform=transform,
        image_size=resize_size,
        image_key=image_key,
        caption_key=caption_key,
        predownload=predownload,
        download_retry=download_retry,
        download_timeout=download_timeout,
        batch_size=batch_size,
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
