# Copyright 2022 MosaicML Diffusion authors
# SPDX-License-Identifier: Apache-2.0

"""Generates images based on a prompt dataset and uploads them for evaluation."""

import json
import os
from typing import Dict, Optional

from composer import ComposerModel, Trainer
from composer.core import get_precision_context
from composer.utils import dist
from composer.utils.object_store import OCIObjectStore
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_pil_image
from tqdm.auto import tqdm
from transformers import PreTrainedTokenizerBase

os.environ['TOKENIZERS_PARALLELISM'] = 'false'


class ImageGenerator:
    """Image generator that generates images from a dataset and saves them.

    Args:
        model (ComposerModel): The model to evaluate.
        dataloader (DataLoader): The dataloader to use the prompts from.
        load_path (str, optional): The path to load the model from. Default: ``None``.
        guidance_scale (float): The guidance scale to use for evaluation. Default: ``7.0``.
        height (int): The height of the generated images. Default: ``1024``.
        width (int): The width of the generated images. Default: ``1024``.
        batch_size (int): The per-device batch size to use for generation. Default: ``1``.
        caption_key (str): The key to use for captions in the dataloader. Default: ``'caption'``.
        load_strict_model_weights (bool): Whether or not to strict load model weights. Default: ``True``.
        seed (int): The seed to use for generation. Default: ``17``.
        output_bucket (str, Optional): The remote to save images to. Default: ``None``.
        output_prefix (str, Optional): The prefix to save images to. Default: ``None``.
        additional_generate_kwargs (Dict, optional): Additional keyword arguments to pass to the model.generate method.

    """

    def __init__(self,
                 model: ComposerModel,
                 dataloader: DataLoader,
                 load_path: Optional[str] = None,
                 guidance_scale: float = 7.0,
                 height: int = 1024,
                 width: int = 1024,
                 batch_size: int = 1,
                 caption_key: str = 'caption',
                 load_strict_model_weights: bool = True,
                 seed: int = 17,
                 output_bucket: Optional[str] = None,
                 output_prefix: Optional[str] = None,
                 additional_generate_kwargs: Optional[Dict] = None):
        self.model = model
        self.tokenizer: PreTrainedTokenizerBase = model.tokenizer
        self.dataloader = dataloader
        self.load_path = load_path
        self.guidance_scale = guidance_scale
        self.height = height
        self.width = width
        self.batch_size = batch_size
        self.caption_key = caption_key
        self.seed = seed
        self.output_bucket = output_bucket
        self.output_prefix = output_prefix if output_prefix is not None else ''
        self.additional_generate_kwargs = additional_generate_kwargs if additional_generate_kwargs is not None else {}
        self.sdxl = model.sdxl

        # Object
        if self.output_bucket is not None:
            self.object_store = OCIObjectStore(self.output_bucket.replace('oci://', ''), self.output_prefix)

        # Load the model
        Trainer(model=self.model,
                load_path=self.load_path,
                load_weights_only=True,
                load_strict_model_weights=load_strict_model_weights,
                eval_dataloader=self.dataloader,
                seed=self.seed)

    def generate(self):
        """Core image generation function. Generates images at a given guidance scale.

        Args:
            guidance_scale (float): The guidance scale to use for image generation.
        """
        # Iterate over the eval dataloader
        os.makedirs(os.path.join('/tmp', self.output_prefix), exist_ok=True)
        for batch_id, batch in tqdm(enumerate(self.dataloader)):
            captions = batch[self.caption_key]
            # Generate images from the captions
            with get_precision_context('amp_fp16'):
                generated_images = self.model.generate(tokenized_prompts=captions,
                                                       height=self.height,
                                                       width=self.width,
                                                       guidance_scale=self.guidance_scale,
                                                       seed=self.seed,
                                                       progress_bar=False,
                                                       **self.additional_generate_kwargs)  # type: ignore
            # Get the prompts from the tokens
            text_captions = self.tokenizer.batch_decode(captions, skip_special_tokens=True)
            # Save the images
            for i, img in enumerate(generated_images):
                image_name = f'rank_{dist.get_global_rank()}_batch_{batch_id}_image_{i}.png'
                data_name = f'rank_{dist.get_global_rank()}_batch_{batch_id}_image_{i}.json'
                img_local_path = os.path.join('/tmp', self.output_prefix, image_name)
                data_local_path = os.path.join('/tmp', self.output_prefix, data_name)
                # Save the image
                img = to_pil_image(img)
                img.save(img_local_path)
                # Save the metadata
                metadata = {
                    'image_name': image_name,
                    'prompt': text_captions[i],
                    'guidance_scale': self.guidance_scale,
                    'seed': self.seed
                }
                json.dump(metadata, open(f'{data_local_path}', 'w'))
                # Upload the image
                if self.output_bucket is not None:
                    self.object_store.upload_object(object_name=os.path.join(self.output_prefix, image_name),
                                                    filename=img_local_path)
                    # Upload the metadata
                    self.object_store.upload_object(object_name=os.path.join(self.output_prefix, data_name),
                                                    filename=data_local_path)
