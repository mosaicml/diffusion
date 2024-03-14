# Copyright 2022 MosaicML Diffusion authors
# SPDX-License-Identifier: Apache-2.0

"""Streaming Image dataset."""

import logging
from io import BytesIO
from typing import Callable, Dict, List, Optional, Sequence, Union

from PIL import Image
from streaming import Stream, StreamingDataset
from torch.utils.data import DataLoader
from torchvision import transforms

log = logging.getLogger(__name__)

# Disable PIL max image size limit
Image.MAX_IMAGE_PIXELS = None


class StreamingImageDataset(StreamingDataset):
    """Streaming dataset for images.

    Args:
        streams (Sequence[Stream], optional): One or more Streams to stream/cache samples from.
            ``StreamingImageCaptionDataset`` uses either ``streams`` or ``remote``/``local``. Default:``None``.
        remote (str, optional): Remote directory (S3 or local filesystem) where dataset is stored. Default: ``None``.
        local (str, optional): Local filesystem directory where dataset is cached during operation. Default: ``None``.
        transform (Callable, optional): The transforms to apply to the image. Default: ``None``.
        image_key (str): Key associated with the image in the streaming dataset. Default: ``'image'``.
        image_output_key (optional, str): Optional output key for the image. If none, the value of `image_key` will
            be used. Default: ``None``.
        return_all_fields (bool, optional): If ``True``, return all fields in the sample. If ``False``, only return
            the image. Default: ``False``.

        **streaming_kwargs: Additional arguments to pass in the construction of the StreamingDataloader
    """

    def __init__(
        self,
        streams: Optional[Sequence[Stream]] = None,
        remote: Optional[str] = None,
        local: Optional[str] = None,
        transform: Optional[Callable] = None,
        image_key: str = 'image',
        image_output_key: Optional[str] = None,
        return_all_fields: bool = False,
        **streaming_kwargs,
    ) -> None:

        # Set defaults for vision-friendly streaming args.
        streaming_kwargs.setdefault('shuffle_block_size', 1 << 18)
        streaming_kwargs.setdefault('shuffle_algo', 'py1s')

        super().__init__(
            streams=streams,
            remote=remote,
            local=local,
            **streaming_kwargs,
        )

        self.transform = transform
        self.image_key = image_key
        self.image_output_key = image_output_key if image_output_key is not None else image_key
        self.return_all_fields = return_all_fields

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        # Image
        if not isinstance(sample[self.image_key], Image.Image):
            img = Image.open(BytesIO(sample[self.image_key]))
        else:
            img = sample[self.image_key].copy()
        if img.mode != 'RGB':
            img = img.convert('RGB')
        # Image transforms
        if self.transform is not None:
            img = self.transform(img)
        # Make the output
        if self.return_all_fields:
            if self.image_output_key != self.image_key and self.image_output_key in sample:
                raise ValueError(f'Output key {self.image_output_key} already exists in the sample.')
            output = {**sample, self.image_output_key: img}
        else:
            output = {self.image_output_key: img}
        return output


def build_streaming_image_dataloader(
    remote: Union[str, List],
    local: Union[str, List],
    batch_size: int,
    transform: Optional[List[Callable]] = None,
    image_key: str = 'image',
    image_output_key: Optional[str] = 'image',
    streaming_kwargs: Optional[Dict] = None,
    dataloader_kwargs: Optional[Dict] = None,
):
    """Builds a streaming LAION dataloader.

    Args:
        remote (str, Sequence[str]): One or more remote directories (S3 or local filesystem) where dataset is stored.
        local (str, Sequence[str]): One or more local filesystem directories where dataset is cached during operation.
        batch_size (int): The batch size to use for both the ``StreamingDataset`` and ``DataLoader``.
        transform (Optional[Callable]): The transforms to apply to the image. Default: ``None``.
        image_key (str): Key associated with the image in the streaming dataset. Default: ``'image'``.
        image_output_key (optional, str): Optional output key for the image. If none, the value of `image_key` will
            be used. Default: ``image``.
        streaming_kwargs (dict, optional): Additional arguments to pass to the ``StreamingDataset``. Default: ``None``.
        dataloader_kwargs (dict, optional): Additional arguments to pass to the ``DataLoader``. Default: ``None``.
    """
    # Handle ``None`` kwargs
    if streaming_kwargs is None:
        streaming_kwargs = {}
    if dataloader_kwargs is None:
        dataloader_kwargs = {}

    # Check types for remote and local
    if isinstance(remote, str) and isinstance(local, str):
        remote, local = [remote], [local]
    elif isinstance(remote, Sequence) and isinstance(local, Sequence):
        if len(remote) != len(local):
            raise ValueError(
                f'remote and local Sequences must be the same length, got lengths {len(remote)} and {len(local)}')
    else:
        raise ValueError(
            f'remote and local must be both Strings or Sequences, got types {type(remote)} and {type(local)}.')

    # Create a Stream for each (remote, local) pair
    streams = []
    for r, l in zip(remote, local):
        streams.append(Stream(remote=r, local=l))

    if transform is None:
        transform = [transforms.ToTensor()]
    transform = transforms.Compose(transform)
    assert isinstance(transform, Callable)

    dataset = StreamingImageDataset(
        streams=streams,
        transform=transform,
        image_key=image_key,
        image_output_key=image_output_key,
        batch_size=batch_size,
        **streaming_kwargs,
    )

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        sampler=None,
        **dataloader_kwargs,
    )

    return dataloader
