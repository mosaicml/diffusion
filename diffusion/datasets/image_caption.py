# Copyright 2022 MosaicML Diffusion authors
# SPDX-License-Identifier: Apache-2.0

"""Streaming Image-Caption dataset."""

import logging
import random
from io import BytesIO
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
import transformers
from PIL import Image
from streaming import Stream, StreamingDataset
from torch.utils.data import DataLoader
from torchvision import transforms

from diffusion.datasets.laion.transforms import (LargestCenterSquare, RandomCropAspectRatioTransform,
                                                 RandomCropBucketedAspectRatioTransform, RandomCropSquare)
from diffusion.datasets.utils import make_streams
from diffusion.models.text_encoder import MultiTokenizer

log = logging.getLogger(__name__)

# Disable PIL max image size limit
Image.MAX_IMAGE_PIXELS = None


class StreamingImageCaptionDataset(StreamingDataset):
    """Streaming dataset for image-caption pairs.

    Args:
        tokenizer (transformers.PreTrainedTokenizer, MultiTokenizer): Tokenizer used for text input.
            Should be the same tokenizer passed to the model being trained.
            Can be accessed with model.tokenizer on Diffusion models. Default: ``None``.
        streams (Sequence[Stream], optional): One or more Streams to stream/cache samples from.
            ``StreamingImageCaptionDataset`` uses either ``streams`` or ``remote``/``local``. Default:``None``.
        remote (str, optional): Remote directory (S3 or local filesystem) where dataset is stored. Default: ``None``.
        local (str, optional): Local filesystem directory where dataset is cached during operation. Default: ``None``.
        caption_drop_prob (float): The probability of dropping a caption. Default: ``0.0``.
        microcond_drop_prob (float): The probability of dropping microconditioning. Only relevant for SDXL. Default: ``0.0``.
        caption_selection (str): If there are multiple captions, specifies how to select a single caption.
            'first' selects the first caption in the list and 'random' selects a random caption in the list.
            If there is only one caption, this argument is ignored. Default: ``'first'``.
        crop (Callable, optional): The crop transform to apply to the image before ``transform``. Default: ``None``
        transform (Callable, optional): The transforms to apply to the image. Default: ``None``.
        image_key (str): Key associated with the image in the streaming dataset. Default: ``'image'``.
        caption_key (str): Key associated with the caption in the streaming dataset. Default: ``'caption'``.
        aspect_ratio_bucket_key (str, optional): Key associated with the aspect ratio bucket in the streaming dataset. Default: ``None``.
        sdxl_conditioning (bool): Whether or not to include SDXL microconditioning in a sample. Default: `False`.
        zero_dropped_captions (bool): If True, zero out text embeddings for dropped captions. Default: ``False``.
        **streaming_kwargs: Additional arguments to pass in the construction of the StreamingDataloader
    """

    def __init__(
        self,
        tokenizer: Optional[Union[transformers.PreTrainedTokenizer, MultiTokenizer]] = None,
        streams: Optional[Sequence[Stream]] = None,
        remote: Optional[str] = None,
        local: Optional[str] = None,
        caption_drop_prob: float = 0.0,
        microcond_drop_prob: float = 0.0,
        caption_selection: str = 'first',
        crop: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        image_key: str = 'image',
        caption_key: str = 'caption',
        aspect_ratio_bucket_key: Optional[str] = None,
        sdxl_conditioning: bool = False,
        zero_dropped_captions: bool = False,
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
        caption_selection = caption_selection.lower()
        if caption_selection not in ['first', 'random']:
            raise ValueError(f'Invalid caption selection: {caption_selection}. Must be one of [random, first]')

        self.crop = crop
        self.transform = transform
        self.sdxl_conditioning = sdxl_conditioning
        self.caption_drop_prob = caption_drop_prob
        self.microcond_drop_prob = microcond_drop_prob
        self.caption_selection = caption_selection
        self.image_key = image_key
        self.caption_key = caption_key
        self.aspect_ratio_bucket_key = aspect_ratio_bucket_key
        if isinstance(self.crop, RandomCropBucketedAspectRatioTransform):
            assert self.aspect_ratio_bucket_key is not None, 'aspect_ratio_bucket_key must be provided when using RandomCropBucketedAspectRatioTransform'
        self.zero_dropped_captions = zero_dropped_captions

        self.tokenizer = tokenizer

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        out = {}

        # Image
        img = sample[self.image_key]
        if not isinstance(img, Image.Image):
            img = Image.open(BytesIO(sample[self.image_key]))
        if img.mode != 'RGB':
            img = img.convert('RGB')
        orig_w, orig_h = img.size

        # Image transforms
        if isinstance(self.crop, RandomCropBucketedAspectRatioTransform):
            img, crop_top, crop_left = self.crop(img, sample[self.aspect_ratio_bucket_key])
        elif self.crop is not None:
            img, crop_top, crop_left = self.crop(img)
        else:
            crop_top, crop_left = 0, 0
        if self.transform is not None:
            img = self.transform(img)
        out['image'] = img

        # SDXL microconditioning on image characteristics
        if self.sdxl_conditioning:
            # Get the new height and width
            if isinstance(img, torch.Tensor):
                img_h, img_w = img.shape[-2], img.shape[-1]
            elif isinstance(img, Image.Image):
                img_w, img_h = img.size
            else:
                raise ValueError('Image after transformations must either be a PIL Image or Torch Tensor')

            out['cond_crops_coords_top_left'] = torch.tensor([crop_top, crop_left])
            out['cond_original_size'] = torch.tensor([orig_w, orig_h])
            out['cond_target_size'] = torch.tensor([img_w, img_h])

            # Microconditioning dropout as in Stability repo
            # https://github.com/Stability-AI/generative-models/blob/477d8b9a7730d9b2e92b326a770c0420d00308c9/sgm/modules/encoders/modules.py#L151-L160
            if torch.rand(1) < self.microcond_drop_prob:
                out['cond_crops_coords_top_left'] = out['cond_crops_coords_top_left'] * 0
            if torch.rand(1) < self.microcond_drop_prob:
                out['cond_original_size'] = out['cond_original_size'] * 0
            if torch.rand(1) < self.microcond_drop_prob:
                out['cond_target_size'] = out['cond_target_size'] * 0

        # Caption
        if torch.rand(1) < self.caption_drop_prob:
            caption = ''
            if self.zero_dropped_captions:
                out['drop_caption_mask'] = 0.0
            else:
                out['drop_caption_mask'] = 1.0
        else:
            caption = sample[self.caption_key]
            if isinstance(caption, List) and self.caption_selection == 'first':
                caption = caption[0]
            if isinstance(caption, List) and self.caption_selection == 'random':
                caption = random.sample(caption, k=1)[0]
            out['drop_caption_mask'] = 1.0

        if self.tokenizer:
            tokenizer_out = self.tokenizer(caption,
                                           padding='max_length',
                                           max_length=self.tokenizer.model_max_length,
                                           truncation=True,
                                           return_tensors='pt')
            out['captions'] = tokenizer_out['input_ids'].squeeze()
            out['attention_mask'] = tokenizer_out['attention_mask'].squeeze()
        else:
            out['captions'] = caption
        return out


def build_streaming_image_caption_dataloader(
    remote: Union[str, List],
    batch_size: int,
    tokenizer: Optional[Union[transformers.PreTrainedTokenizer, MultiTokenizer]] = None,
    local: Optional[Union[str, List]] = None,
    caption_drop_prob: float = 0.0,
    microcond_drop_prob: float = 0.0,
    resize_size: Union[int, Tuple[int, int], Tuple[Tuple[int, int], ...]] = 256,
    ar_bucket_boundaries: Optional[Tuple[float, ...]] = None,
    caption_selection: str = 'first',
    transform: Optional[List[Callable]] = None,
    image_key: str = 'image',
    caption_key: str = 'caption',
    aspect_ratio_bucket_key: Optional[str] = None,
    crop_type: Optional[str] = 'square',
    zero_dropped_captions: bool = True,
    sdxl_conditioning: bool = False,
    proportion: Optional[list] = None,
    repeat: Optional[list] = None,
    choose: Optional[list] = None,
    streaming_kwargs: Optional[Dict] = None,
    dataloader_kwargs: Optional[Dict] = None,
):
    """Builds a streaming LAION dataloader.

    Args:
        remote (str, Sequence[str]): One or more remote directories (S3 or local filesystem) where dataset is stored.
        local (str, Sequence[str]): One or more local filesystem directories where dataset is cached during operation.
        batch_size (int): The batch size to use for both the ``StreamingDataset`` and ``DataLoader``.
        tokenizer (transformers.PreTrainedTokenizer, MultiTokenizer): Tokenizer used for text input.
            Should be the same tokenizer passed to the model being trained.
            Can be accessed with model.tokenizer on Diffusion models. Default: ``None``.
        caption_drop_prob (float): The probability of dropping a caption. Default: ``0.0``.
        microcond_drop_prob (float): The probability of dropping microconditioning. Only relevant for SDXL.
            Default:``0.0``.
        resize_size (int, Tuple[int, int], Tuple[Tuple[int, int], ...): The size to resize the image to. Specify a
            tuple of tuples if using 'aspect_ratio' crop_type. Default: ``256``.
        ar_bucket_boundaries (Tuple[float, ...], optional): When using ``crop_type='aspect_ratio'``, specifies the
            boundary points for bucket assignment. This tuple should be of length len(resize_size) - 1. If set to
            ``None``, the bucket with the smallest distance to the current sample's aspect ratio is selected.
            Default: ``None``.
        caption_selection (str): If there are multiple captions, specifies how to select a single caption.
            'first' selects the first caption in the list and 'random' selects a random caption in the list.
            If there is only one caption, this argument is ignored. Default: ``'first'``.
        transform (Optional[Callable]): The transforms to apply to the image. Default: ``None``.
        image_key (str): Key associated with the image in the streaming dataset. Default: ``'image'``.
        caption_key (str): Key associated with the caption in the streaming dataset. Default: ``'caption'``.
        aspect_ratio_bucket_key (str, optional): Key associated with the aspect ratio bucket in the streaming dataset. Default: ``None``.
        crop_type (str, optional): Type of crop to perform, either ['square', 'random', 'aspect_ratio', 'bucketed_aspect_ratio'].
            Default: ``'square'``.
        zero_dropped_captions (bool): If True, zero out text embeddings for dropped captions. Default: ``True``.
        sdxl_conditioning (bool): Whether or not to include SDXL microconditioning in a sample. Default: `False`.
        proportion (list, optional): Specifies how to sample this Stream relative to other Streams. Default: ``None``.
        repeat (list, optional): Specifies the degree to which a Stream is upsampled or downsampled. Default: ``None``.
        choose (list, optional): Specifies the number of samples to choose from a Stream. Default: ``None``.
        streaming_kwargs (dict, optional): Additional arguments to pass to the ``StreamingDataset``. Default: ``None``.
        dataloader_kwargs (dict, optional): Additional arguments to pass to the ``DataLoader``. Default: ``None``.
    """
    # Check crop type
    if crop_type is not None:
        crop_type = crop_type.lower()
        if crop_type not in ['square', 'random', 'aspect_ratio', 'bucketed_aspect_ratio']:
            raise ValueError(
                f'Invalid crop_type: {crop_type}. Must be ["square", "random", "aspect_ratio", "bucketed_aspect_ratio", None]'
            )
        if crop_type in ['aspect_ratio', 'bucketed_aspect_ratio'] and (isinstance(resize_size, int) or
                                                                       isinstance(resize_size[0], int)):
            raise ValueError(
                'If using aspect ratio bucketing, specify aspect ratio buckets in resize_size as a tuple of tuples.')
    # Handle ``None`` kwargs
    if streaming_kwargs is None:
        streaming_kwargs = {}
    if dataloader_kwargs is None:
        dataloader_kwargs = {}

    # Set up streams
    streams = make_streams(remote, local=local, proportion=proportion, repeat=repeat, choose=choose)

    # Set the crop to apply
    if crop_type == 'square':
        crop = LargestCenterSquare(resize_size)
    elif crop_type == 'random':
        crop = RandomCropSquare(resize_size)
    elif crop_type == 'aspect_ratio':
        crop = RandomCropAspectRatioTransform(resize_size, ar_bucket_boundaries)  # type: ignore
    elif crop_type == 'bucketed_aspect_ratio':
        assert aspect_ratio_bucket_key is not None, 'aspect_ratio_bucket_key must be provided when using bucketed_aspect_ratio crop type'
        crop = RandomCropBucketedAspectRatioTransform(resize_size)  # type: ignore
    else:
        crop = None

    if transform is None:
        transform = [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    transform = transforms.Compose(transform)
    assert isinstance(transform, Callable)

    dataset = StreamingImageCaptionDataset(
        streams=streams,
        tokenizer=tokenizer,
        caption_drop_prob=caption_drop_prob,
        microcond_drop_prob=microcond_drop_prob,
        caption_selection=caption_selection,
        crop=crop,
        transform=transform,
        image_key=image_key,
        caption_key=caption_key,
        aspect_ratio_bucket_key=aspect_ratio_bucket_key,
        batch_size=batch_size,
        sdxl_conditioning=sdxl_conditioning,
        zero_dropped_captions=zero_dropped_captions,
        **streaming_kwargs,
    )

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        sampler=None,
        **dataloader_kwargs,
    )

    return dataloader
