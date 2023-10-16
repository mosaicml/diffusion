# Copyright 2022 MosaicML Diffusion authors
# SPDX-License-Identifier: Apache-2.0

"""Transforms for the training and eval dataset."""

try:
    import torchvision.transforms.v2 as transforms
except ImportError:
    import torchvision.transforms as transforms

RandomCrop = transforms.RandomCrop
crop = transforms.functional


class LargestCenterSquare:
    """Center crop to the largest square of a PIL image."""

    def __init__(self, size):
        self.size = size
        self.center_crop = transforms.CenterCrop(self.size)

    def __call__(self, img):
        # First, resize the image such that the smallest side is self.size while preserving aspect ratio.
        img = transforms.functional.resize(img, self.size, antialias=True)
        # Then take a center crop to a square.
        img = self.center_crop(img)
        return img


class RandomCropSquare:
    """Randomly crop square of a PIL image."""

    def __init__(self, size):
        self.size = size
        self.random_crop = RandomCrop(size)

    def __call__(self, img):
        # First, resize the image such that the smallest side is self.size while preserving aspect ratio.
        img = transforms.functional.resize(img, self.size, antialias=True)
        # Then take a center crop to a square & return crop params.
        c_top, c_left, h, w = self.random_crop.get_params(img, (self.size, self.size))
        img = crop(img, c_top, c_left, h, w)
        return img


class RandomCropSquareReturnTransform:
    """Randomly crop square of a PIL image and return the crop parameters."""

    def __init__(self, size):
        self.size = size
        self.random_crop = RandomCrop(size)

    def __call__(self, img):
        # First, resize the image such that the smallest side is self.size while preserving aspect ratio.
        orig_w, orig_h = img.size
        img = transforms.functional.resize(img, self.size, antialias=True)
        # Then take a center crop to a square & return crop params.
        c_top, c_left, h, w = self.random_crop.get_params(img, (self.size, self.size))
        img = crop(img, c_top, c_left, h, w)
        return img, c_top, c_left, orig_h, orig_w
