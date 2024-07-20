# Copyright 2022 MosaicML Diffusion authors
# SPDX-License-Identifier: Apache-2.0

"""Transforms for the training and eval dataset."""

import math
from typing import Optional, Tuple

import torch
import torchvision.transforms as transforms
from torchvision.transforms.functional import crop


class LargestCenterSquare:
    """Center crop to the largest square of a PIL image."""

    def __init__(self, size):
        self.size = size
        self.center_crop = transforms.CenterCrop(self.size)

    def __call__(self, img):
        # First, resize the image such that the smallest side is self.size while preserving aspect ratio.
        img = transforms.functional.resize(img, self.size, antialias=True)

        # Then take a center crop to a square.
        w, h = img.size
        c_top = (h - self.size) // 2
        c_left = (w - self.size) // 2
        img = crop(img, c_top, c_left, self.size, self.size)
        return img, c_top, c_left


class RandomCropSquare:
    """Randomly crop square of a PIL image and return the crop parameters."""

    def __init__(self, size):
        self.size = size
        self.random_crop = transforms.RandomCrop(size)

    def __call__(self, img):
        # First, resize the image such that the smallest side is self.size while preserving aspect ratio.
        img = transforms.functional.resize(img, self.size, antialias=True)
        # Then take a center crop to a square & return crop params.
        c_top, c_left, h, w = self.random_crop.get_params(img, (self.size, self.size))
        img = crop(img, c_top, c_left, h, w)
        return img, c_top, c_left


class RandomCropAspectRatioTransorm:
    """Assigns an image to a arbitrary set of aspect ratio buckets, then resizes and crops to fit into the bucket.

    Args:
        resize_size (Tuple[Tuple[int, int], ...): A tuple of 2-tuple integers representing the aspect ratio buckets.
            The format is ((height_bucket1, width_bucket1), (height_bucket2, width_bucket2), ...). These must be
            ordered based on ascending aspect ratio.
        ar_bucket_boundaries (Tuple[float, ...], optional): Specifies the boundary points for bucket assignment. This
            tuple should be of length len(resize_size) - 1. If set to ``None``, the bucket with the smallest distance
            to the current sample's aspect ratio is selected. These must be ordered based on ascending aspect ratio.
            Default: ``None``
    """

    def __init__(
        self,
        resize_size: Tuple[Tuple[int, int], ...],
        ar_bucket_boundaries: Optional[Tuple[float, ...]] = None,
    ):

        if ar_bucket_boundaries is not None and (len(resize_size) - 1 != len(ar_bucket_boundaries)):
            raise ValueError(
                f'Bucket boundaries ({len(ar_bucket_boundaries)}) must equal resize sizes ({len(resize_size)}) - 1')

        self.height_buckets = torch.tensor([size[0] for size in resize_size])
        self.width_buckets = torch.tensor([size[1] for size in resize_size])
        self.aspect_ratio_buckets = self.height_buckets / self.width_buckets

        # If ar_bucket_boundaries is not None, add 0 and inf endpoints
        self.ar_bucket_boundaries = (0.0, *ar_bucket_boundaries, float('inf')) if ar_bucket_boundaries else None

    def __call__(self, img):
        orig_w, orig_h = img.size
        orig_aspect_ratio = orig_h / orig_w

        # Assign sample to an aspect ratio bucket
        if self.ar_bucket_boundaries is None:
            bucket_ind = torch.abs(self.aspect_ratio_buckets - orig_aspect_ratio).argmin()
        else:
            bucket_ind = None
            for i, (low, high) in enumerate(zip(self.ar_bucket_boundaries[:-1], self.ar_bucket_boundaries[1:])):
                if (i < len(self.aspect_ratio_buckets) // 2) and (low <= orig_aspect_ratio < high):
                    bucket_ind = i
                elif (i == len(self.aspect_ratio_buckets) // 2) and (low <= orig_aspect_ratio <= high):
                    bucket_ind = i
                elif (i > len(self.aspect_ratio_buckets) // 2) and (low < orig_aspect_ratio <= high):
                    bucket_ind = i
            assert bucket_ind is not None, f'Sample with aspect ratio ({orig_aspect_ratio}) was not assigned to a bucket. Check the bucket boundaries.'

        target_width, target_height = self.width_buckets[bucket_ind].item(), self.height_buckets[bucket_ind].item()
        target_aspect_ratio = target_height / target_width

        # Determine resize size
        if orig_aspect_ratio > target_aspect_ratio:
            w_scale = target_width / orig_w
            resize_size = (round(w_scale * orig_h), target_width)
        elif orig_aspect_ratio < target_aspect_ratio:
            h_scale = target_height / orig_h
            resize_size = (target_height, round(h_scale * orig_w))
        else:
            resize_size = (target_height, target_width)
        img = transforms.functional.resize(img, resize_size, antialias=True)

        # Crop based on aspect ratio
        c_top, c_left, height, width = transforms.RandomCrop.get_params(img, output_size=(target_height, target_width))
        img = crop(img, c_top, c_left, height, width)
        return img, c_top, c_left


class RandomCropBucketedAspectRatioTransorm:
    """Assigns an image to a arbitrary set of aspect ratio buckets, then resizes and crops to fit into the bucket.

    This transform requires the desired aspect ratio bucket to be specified manually in the call to the transform.

    Args:
        resize_size (Tuple[Tuple[int, int], ...): A tuple of 2-tuple integers representing the aspect ratio buckets.
            The format is ((height_bucket1, width_bucket1), (height_bucket2, width_bucket2), ...).
    """

    def __init__(
        self,
        resize_size: Tuple[Tuple[int, int], ...],
    ):
        self.height_buckets = torch.tensor([size[0] for size in resize_size])
        self.width_buckets = torch.tensor([size[1] for size in resize_size])
        self.aspect_ratio_buckets = self.height_buckets / self.width_buckets
        self.log_aspect_ratio_buckets = torch.log(self.aspect_ratio_buckets)

    def __call__(self, img, aspect_ratio):
        orig_w, orig_h = img.size
        orig_aspect_ratio = orig_h / orig_w
        # Figure out target H/W given the input aspect ratio
        bucket_ind = torch.abs(self.log_aspect_ratio_buckets - math.log(aspect_ratio)).argmin()
        target_width, target_height = self.width_buckets[bucket_ind].item(), self.height_buckets[bucket_ind].item()
        target_aspect_ratio = target_height / target_width

        # Determine resize size
        if orig_aspect_ratio > target_aspect_ratio:
            # Resize width and crop height
            w_scale = target_width / orig_w
            resize_size = (round(w_scale * orig_h), target_width)
        elif orig_aspect_ratio < target_aspect_ratio:
            # Resize height and crop width
            h_scale = target_height / orig_h
            resize_size = (target_height, round(h_scale * orig_w))
        else:
            resize_size = (target_height, target_width)
        img = transforms.functional.resize(img, resize_size, antialias=True)

        # Crop based on aspect ratio
        c_top, c_left, height, width = transforms.RandomCrop.get_params(img, output_size=(target_height, target_width))
        img = crop(img, c_top, c_left, height, width)
        return img, c_top, c_left
