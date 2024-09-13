# Copyright 2022 MosaicML Diffusion authors
# SPDX-License-Identifier: Apache-2.0

"""Image generation for runnning evaluation with geneval."""

import json
import os
from typing import Dict, Optional, Union
from urllib.parse import urlparse

import torch
from composer.core import get_precision_context
from composer.utils import dist
from composer.utils.file_helpers import get_file
from composer.utils.object_store import OCIObjectStore
from diffusers import AutoPipelineForText2Image
from torchvision.transforms.functional import to_pil_image
from tqdm.auto import tqdm


class GenevalImageGenerator:
    """Image generator that generates images from the geneval prompt set and saves them.

    Args:
        model (torch.nn.Module): The model to evaluate.
        geneval_prompts (str): Path to the prompts to use for geneval (ex: `geneval/prompts/evaluation_metadata.json`).
        load_path (str, optional): The path to load the model from. Default: ``None``.
        local_checkpoint_path (str, optional): The local path to save the model checkpoint. Default: ``'/tmp/model.pt'``.
        load_strict_model_weights (bool): Whether or not to strict load model weights. Default: ``True``.
        guidance_scale (float): The guidance scale to use for evaluation. Default: ``7.0``.
        height (int): The height of the generated images. Default: ``1024``.
        width (int): The width of the generated images. Default: ``1024``.
        images_per_prompt (int): The number of images to generate per prompt. Default: ``4``.
        load_strict_model_weights (bool): Whether or not to strict load model weights. Default: ``True``.
        seed (int): The seed to use for generation. Default: ``17``.
        output_bucket (str, Optional): The remote to save images to. Default: ``None``.
        output_prefix (str, Optional): The prefix to save images to. Default: ``None``.
        local_prefix (str): The local prefix to save images to. Default: ``/tmp``.
        additional_generate_kwargs (Dict, optional): Additional keyword arguments to pass to the model.generate method.
        hf_model: (bool, Optional): whether the model is HF or not. Default: ``False``.
    """

    def __init__(self,
                 model: Union[torch.nn.Module, str],
                 geneval_prompts: str,
                 load_path: Optional[str] = None,
                 local_checkpoint_path: str = '/tmp/model.pt',
                 load_strict_model_weights: bool = True,
                 guidance_scale: float = 7.0,
                 height: int = 1024,
                 width: int = 1024,
                 images_per_prompt: int = 4,
                 seed: int = 17,
                 output_bucket: Optional[str] = None,
                 output_prefix: Optional[str] = None,
                 local_prefix: str = '/tmp',
                 additional_generate_kwargs: Optional[Dict] = None,
                 hf_model: Optional[bool] = False):

        if isinstance(model, str) and hf_model == False:
            raise ValueError('Can only use strings for model with hf models!')
        self.hf_model = hf_model
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
        # Load the geneval prompts
        self.geneval_prompts = geneval_prompts
        with open(geneval_prompts) as f:
            self.prompt_metadata = [json.loads(line) for line in f]
        self.load_path = load_path
        self.local_checkpoint_path = local_checkpoint_path
        self.load_strict_model_weights = load_strict_model_weights
        self.guidance_scale = guidance_scale
        self.height = height
        self.width = width
        self.images_per_prompt = images_per_prompt
        self.seed = seed
        self.generator = torch.Generator(device='cuda').manual_seed(self.seed)

        self.output_bucket = output_bucket
        self.output_prefix = output_prefix if output_prefix is not None else ''
        self.local_prefix = local_prefix
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
                state_dict = torch.load(self.local_checkpoint_path, map_location='cpu')
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
        os.makedirs(os.path.join(self.local_prefix, self.output_prefix), exist_ok=True)
        # Partition the dataset across the ranks. Note this partitions prompts, not repeats.
        dataset_len = len(self.prompt_metadata)
        samples_per_rank, remainder = divmod(dataset_len, dist.get_world_size())
        start_idx = dist.get_global_rank() * samples_per_rank + min(remainder, dist.get_global_rank())
        end_idx = start_idx + samples_per_rank
        if dist.get_global_rank() < remainder:
            end_idx += 1
        print(f'Rank {dist.get_global_rank()} processing samples {start_idx} to {end_idx} of {dataset_len} total.')
        # Iterate over the dataset
        for sample_id in tqdm(range(start_idx, end_idx)):
            metadata = self.prompt_metadata[sample_id]
            # Write the metadata jsonl
            output_dir = os.path.join(self.local_prefix, self.output_prefix, f'{sample_id:0>5}')
            os.makedirs(output_dir, exist_ok=True)
            with open(os.path.join(output_dir, 'metadata.jsonl'), 'w') as f:
                json.dump(metadata, f)
            caption = metadata['prompt']
            # Create dir for samples to live in
            sample_dir = os.path.join(output_dir, 'samples')
            os.makedirs(sample_dir, exist_ok=True)
            # Generate images from the captions. Take care to use a different seed for each image
            for i in range(self.images_per_prompt):
                seed = self.seed + i
                if self.hf_model:
                    generated_image = self.model(prompt=caption,
                                                 height=self.height,
                                                 width=self.width,
                                                 guidance_scale=self.guidance_scale,
                                                 generator=self.generator,
                                                 **self.additional_generate_kwargs).images[0]
                    img = generated_image
                else:
                    with get_precision_context('amp_fp16'):
                        generated_image = self.model.generate(prompt=caption,
                                                              height=self.height,
                                                              width=self.width,
                                                              guidance_scale=self.guidance_scale,
                                                              seed=seed,
                                                              progress_bar=False,
                                                              **self.additional_generate_kwargs)  # type: ignore
                    img = to_pil_image(generated_image[0])
                # Save the images and metadata locally
                image_name = f'{sample_id:05}.png'
                data_name = f'{sample_id:05}.json'
                img_local_path = os.path.join(sample_dir, image_name)
                data_local_path = os.path.join(sample_dir, data_name)
                img.save(img_local_path)
                metadata = {
                    'image_name': image_name,
                    'prompt': caption,
                    'guidance_scale': self.guidance_scale,
                    'seed': seed
                }
                json.dump(metadata, open(f'{data_local_path}', 'w'))
                # Upload the image and metadata to cloud storage
                output_sample_prefix = os.path.join(self.output_prefix, f'{sample_id:0>5}', 'samples')
                if self.output_bucket is not None:
                    self.object_store.upload_object(object_name=os.path.join(output_sample_prefix, image_name),
                                                    filename=img_local_path)
                    # Upload the metadata
                    self.object_store.upload_object(object_name=os.path.join(output_sample_prefix, data_name),
                                                    filename=data_local_path)
