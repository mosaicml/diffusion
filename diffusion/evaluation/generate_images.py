# Copyright 2022 MosaicML Diffusion authors
# SPDX-License-Identifier: Apache-2.0

"""Generates images based on a prompt dataset and uploads them for evaluation."""

import json
import os
from typing import Dict, Optional, Union
from urllib.parse import urlparse

import torch
from composer.core import get_precision_context
from composer.utils import dist
from composer.utils.file_helpers import get_file
from composer.utils.object_store import OCIObjectStore
from datasets import DatasetDict
from diffusers import AutoPipelineForText2Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_pil_image
from tqdm.auto import tqdm


class ImageGenerator:
    """Image generator that generates images from a dataset and saves them.

    Args:
        model (torch.nn.Module): The model to evaluate.
        dataset (Dataset): The dataset to use the prompts from.
        load_path (str, optional): The path to load the model from. Default: ``None``.
        local_checkpoint_path (str, optional): The local path to save the model checkpoint. Default: ``'/tmp/model.pt'``.
        load_strict_model_weights (bool): Whether or not to strict load model weights. Default: ``True``.
        guidance_scale (float): The guidance scale to use for evaluation. Default: ``7.0``.
        height (int): The height of the generated images. Default: ``1024``.
        width (int): The width of the generated images. Default: ``1024``.
        caption_key (str): The key to use for captions in the dataloader. Default: ``'caption'``.
        load_strict_model_weights (bool): Whether or not to strict load model weights. Default: ``True``.
        seed (int): The seed to use for generation. Default: ``17``.
        output_bucket (str, Optional): The remote to save images to. Default: ``None``.
        output_prefix (str, Optional): The prefix to save images to. Default: ``None``.
        extra_keys (list, Optional): Extra keys from the dataset to include in the metadata. Default: ``None``.
        additional_generate_kwargs (Dict, optional): Additional keyword arguments to pass to the model.generate method.
        hf_model: (bool, Optional): whether the model is HF or not. Default: ``False``.
        hf_dataset: (bool, Optional): whether the dataset is HF formatted or not. Default: ``False``.
    """

    def __init__(self,
                 model: Union[torch.nn.Module, str],
                 dataset: Union[Dataset, DatasetDict],
                 load_path: Optional[str] = None,
                 local_checkpoint_path: str = '/tmp/model.pt',
                 load_strict_model_weights: bool = True,
                 guidance_scale: float = 7.0,
                 height: int = 1024,
                 width: int = 1024,
                 caption_key: str = 'caption',
                 seed: int = 17,
                 output_bucket: Optional[str] = None,
                 output_prefix: Optional[str] = None,
                 extra_keys: Optional[list] = None,
                 additional_generate_kwargs: Optional[Dict] = None,
                 hf_model: Optional[bool] = False,
                 hf_dataset: Optional[bool] = False):

        if isinstance(model, str) and hf_model == False:
            raise ValueError('Can only use strings for model with hf models!')
        self.hf_model = hf_model
        self.hf_dataset = hf_dataset
        if hf_model or isinstance(model, str):
            if dist.get_local_rank() == 0:
                self.model = AutoPipelineForText2Image.from_pretrained(
                    model, torch_dtype=torch.float16).to(f'cuda:{dist.get_local_rank()}')
            dist.barrier()
            self.model = AutoPipelineForText2Image.from_pretrained(
                model, torch_dtype=torch.float16).to(f'cuda:{dist.get_local_rank()}')
            dist.barrier()
        else:
            self.model = model
        self.dataset = dataset
        self.load_path = load_path
        self.local_checkpoint_path = local_checkpoint_path
        self.load_strict_model_weights = load_strict_model_weights
        self.guidance_scale = guidance_scale
        self.height = height
        self.width = width
        self.caption_key = caption_key
        self.seed = seed
        self.generator = torch.Generator(device='cuda').manual_seed(self.seed)

        self.output_bucket = output_bucket
        self.output_prefix = output_prefix if output_prefix is not None else ''
        self.extra_keys = extra_keys if extra_keys is not None else []
        self.additional_generate_kwargs = additional_generate_kwargs if additional_generate_kwargs is not None else {}

        # Object store for uploading images
        if self.output_bucket is not None:
            parsed_remote_bucket = urlparse(self.output_bucket)
            if parsed_remote_bucket.scheme != 'oci':
                raise ValueError(f'Currently only OCI object stores are supported. Got {parsed_remote_bucket.scheme}.')
            self.object_store = OCIObjectStore(self.output_bucket.replace('oci://', ''), self.output_prefix)

        # Download the model checkpoint if needed
        if self.load_path is not None and not isinstance(self.model, str):
            if dist.get_local_rank() == 0:
                get_file(path=self.load_path, destination=self.local_checkpoint_path, overwrite=True)
            with dist.local_rank_zero_download_and_wait(self.local_checkpoint_path):
                # Load the model
                state_dict = torch.load(self.local_checkpoint_path)
            for key in list(state_dict['state']['model'].keys()):
                if 'val_metrics.' in key:
                    del state_dict['state']['model'][key]
            self.model.load_state_dict(state_dict['state']['model'], strict=self.load_strict_model_weights)
            self.model = self.model.cuda().eval()

    def generate(self):
        """Core image generation function. Generates images at a given guidance scale.

        Args:
            guidance_scale (float): The guidance scale to use for image generation.
        """
        os.makedirs(os.path.join('/tmp', self.output_prefix), exist_ok=True)
        # Partition the dataset across the ranks
        if self.hf_dataset:
            dataset_len = self.dataset.num_rows  # type: ignore
        else:
            dataset_len = self.dataset.num_samples  # type: ignore
        samples_per_rank, remainder = divmod(dataset_len, dist.get_world_size())
        start_idx = dist.get_global_rank() * samples_per_rank + min(remainder, dist.get_global_rank())
        end_idx = start_idx + samples_per_rank
        if dist.get_global_rank() < remainder:
            end_idx += 1
        print(f'Rank {dist.get_global_rank()} processing samples {start_idx} to {end_idx} of {dataset_len} total.')
        # Iterate over the dataset
        for sample_id in tqdm(range(start_idx, end_idx)):
            sample = self.dataset[sample_id]
            caption = sample[self.caption_key]
            # Generate images from the captions
            if self.hf_model:
                generated_image = self.model(prompt=caption,
                                             height=self.height,
                                             width=self.width,
                                             guidance_scale=self.guidance_scale,
                                             generator=self.generator,
                                             **self.additional_generate_kwargs).images[0]
            else:
                with get_precision_context('amp_fp16'):
                    generated_image = self.model.generate(prompt=caption,
                                                          height=self.height,
                                                          width=self.width,
                                                          guidance_scale=self.guidance_scale,
                                                          seed=self.seed,
                                                          progress_bar=False,
                                                          **self.additional_generate_kwargs)  # type: ignore
            # Save the images
            image_name = f'{sample_id}.png'
            data_name = f'{sample_id}.json'
            img_local_path = os.path.join('/tmp', self.output_prefix, image_name)
            data_local_path = os.path.join('/tmp', self.output_prefix, data_name)
            # Save the image
            if self.hf_model:
                img = generated_image
            else:
                img = to_pil_image(generated_image[0])
            img.save(img_local_path)
            # Save the metadata
            metadata = {
                'image_name': image_name,
                'prompt': caption,
                'guidance_scale': self.guidance_scale,
                'seed': self.seed
            }
            for key in self.extra_keys:
                metadata[key] = sample[key]
            json.dump(metadata, open(f'{data_local_path}', 'w'))
            # Upload the image
            if self.output_bucket is not None:
                self.object_store.upload_object(object_name=os.path.join(self.output_prefix, image_name),
                                                filename=img_local_path)
                # Upload the metadata
                self.object_store.upload_object(object_name=os.path.join(self.output_prefix, data_name),
                                                filename=data_local_path)
