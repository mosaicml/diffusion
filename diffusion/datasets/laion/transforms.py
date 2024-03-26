# Copyright 2022 MosaicML Diffusion authors
# SPDX-License-Identifier: Apache-2.0

"""Transforms for the training and eval dataset."""

from typing import Tuple

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
    """Assigns an image to a pre-defined set of aspect ratio buckets, then resizes and crops to fit into the bucket."""

    def __init__(self, resize_size: Tuple[Tuple[int, int], ...]):
        self.height_buckets = torch.tensor([size[0] for size in resize_size])
        self.width_buckets = torch.tensor([size[1] for size in resize_size])
        self.aspect_ratio_buckets = self.height_buckets / self.width_buckets

    def __call__(self, img):
        orig_w, orig_h = img.size
        orig_aspect_ratio = orig_h / orig_w
        bucket_ind = torch.abs(self.aspect_ratio_buckets - orig_aspect_ratio).argmin()
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
