# Copyright 2022 MosaicML Diffusion authors
# SPDX-License-Identifier: Apache-2.0

"""Evaluation using the clean-fid package."""

import json
import os
from typing import Dict, List, Optional

import clip
import torch
from cleanfid import fid
from composer import ComposerModel, Trainer
from composer.core import get_precision_context
from composer.loggers import LoggerDestination
from composer.utils import dist
from torch.utils.data import Dataset
from torchmetrics.multimodal import CLIPScore
from torchvision.transforms.functional import pil_to_tensor, to_pil_image
from tqdm.auto import tqdm

os.environ['TOKENIZERS_PARALLELISM'] = 'false'


class CleanFIDEvaluator:
    """Evaluator for CLIP, FID, KID, CLIP-FID scores using clean-fid.

    See https://github.com/GaParmar/clean-fid for more information on clean-fid.

    CLIP scores are computed using the torchmetrics CLIPScore metric.

    Args:
        model (ComposerModel): The model to evaluate.
        dataset (Dataset): The dataset to use the prompts from.
        clip_metric (CLIPScore): The CLIPScore metric to use for evaluation.
        load_path (str, optional): The path to load the model from. Default: ``None``.
        guidance_scales (List[float]): The guidance scales to use for evaluation.
            Default: ``[1.0]``.
        size (int): The size of the images to generate. Default: ``256``.
        batch_size (int): The per-device batch size to use for evaluation. Default: ``16``.
        load_strict_model_weights (bool): Whether or not to strict load model weights. Default: ``True``.
        loggers (List[LoggerDestination], optional): The loggers to use for logging results. Default: ``None``.
        seed (int): The seed to use for evaluation. Default: ``17``.
        output_dir (str): The directory to save results to. Default: ``/tmp/``.
        num_samples (int, optional): The maximum number of samples to generate. Depending on batch size, actual
            number may be slightly higher. If not specified, all the samples in the dataloader will be used.
            Default: ``None``.
        precision (str): The precision to use for evaluation. Default: ``'amp_fp16'``.
        prompts (List[str], optional): The prompts to use for image visualtization.
            Default: ``["A shiba inu wearing a blue sweater]``.
        default_prompt (Optional[str]): An optional default prompt to add before each eval prompt. Default: ``None``.
        default_negative_prompt (Optional[str]): An optional default negative prompt to add before each
            negative prompt. Default: ``None``.
        additional_generate_kwargs (Dict, optional): Additional keyword arguments to pass to the model.generate method.

    """

    def __init__(self,
                 model: ComposerModel,
                 dataset: Dataset,
                 clip_metric: CLIPScore,
                 load_path: Optional[str] = None,
                 guidance_scales: Optional[List[float]] = None,
                 size: int = 256,
                 batch_size: int = 16,
                 image_key: str = 'image',
                 caption_key: str = 'caption',
                 load_strict_model_weights: bool = True,
                 loggers: Optional[List[LoggerDestination]] = None,
                 seed: int = 17,
                 output_dir: str = '/tmp/',
                 num_samples: Optional[int] = None,
                 precision: str = 'amp_fp16',
                 prompts: Optional[List[str]] = None,
                 default_prompt: Optional[str] = None,
                 default_negative_prompt: Optional[str] = None,
                 additional_generate_kwargs: Optional[Dict] = None):
        self.model = model
        self.dataset = dataset
        self.clip_metric = clip_metric
        self.load_path = load_path
        self.guidance_scales = guidance_scales if guidance_scales is not None else [1.0]
        self.size = size
        self.batch_size = batch_size
        self.image_key = image_key
        self.caption_key = caption_key
        self.loggers = loggers
        self.seed = seed
        self.output_dir = output_dir
        self.num_samples = num_samples
        self.precision = precision
        self.prompts = prompts if prompts is not None else ['A shiba inu wearing a blue sweater']
        self.default_prompt = default_prompt
        self.default_negative_prompt = default_negative_prompt
        self.additional_generate_kwargs = additional_generate_kwargs if additional_generate_kwargs is not None else {}
        self.sdxl = model.sdxl

        # Load the model
        trainer = Trainer(model=self.model,
                          load_path=self.load_path,
                          load_weights_only=True,
                          load_strict_model_weights=load_strict_model_weights,
                          seed=self.seed,
                          loggers=self.loggers)
        self.trainer = trainer

        # Move CLIP metric to device
        self.device = dist.get_local_rank()
        self.clip_metric = self.clip_metric.to(self.device)

        # Predownload the CLIP model for computing clip-fid
        clip_url = 'https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt'
        clip_name = os.path.basename(clip_url)
        clip_path = os.path.expanduser('~/.cache/clip')
        if dist.get_local_rank() == 0:
            clip.clip._download(clip_url, clip_path)
        with dist.local_rank_zero_download_and_wait(os.path.join(clip_path, clip_name)):
            clip.load('ViT-B/32', device=self.device)

    def _generate_images(self, guidance_scale: float):
        """Core image generation function. Generates images at a given guidance scale.

        Args:
            guidance_scale (float): The guidance scale to use for image generation.
        """
        # Verify output dirs exist, if they don't, create them
        real_image_path = os.path.join(self.output_dir, f'real_images_gs_{guidance_scale}')
        gen_image_path = os.path.join(self.output_dir, f'gen_images_gs_{guidance_scale}')
        if not os.path.exists(real_image_path) and dist.get_local_rank() == 0:
            os.makedirs(real_image_path)
        if not os.path.exists(gen_image_path) and dist.get_local_rank() == 0:
            os.makedirs(gen_image_path)

        # Reset the CLIP metric
        self.clip_metric.reset()

        # Storage for prompts
        prompts = {}
        # Partition the dataset across the ranks
        dataset_len = self.dataset.num_samples  # type: ignore
        # Truncate the dataset if num_samples is specified
        if self.num_samples is not None and self.num_samples <= dataset_len:
            dataset_len = self.num_samples
        elif self.num_samples is not None and self.num_samples > dataset_len:
            raise ValueError(f'num_samples {self.num_samples} is greater than the dataset length {dataset_len}.')
        samples_per_rank, remainder = divmod(dataset_len, dist.get_world_size())
        start_idx = dist.get_global_rank() * samples_per_rank + min(remainder, dist.get_global_rank())
        end_idx = start_idx + samples_per_rank
        if dist.get_global_rank() < remainder:
            end_idx += 1
        print(f'Rank {dist.get_global_rank()} processing samples {start_idx} to {end_idx} of {dataset_len} total.')
        # Iterate over the dataset
        for sample_id in tqdm(range(start_idx, end_idx)):
            # Set a unique seed for this sample to ensure reproducible but different randomness
            seed = self.seed + sample_id
            # Image and caption come from the dataset. Note the caption is untokenized
            sample = self.dataset[sample_id]
            real_images = pil_to_tensor(sample[self.image_key]).unsqueeze(0) / 255.0
            text_captions = sample[self.caption_key]
            # Add default prompts if specified
            augmented_captions = text_captions
            augmented_negative_prompt = None
            if self.default_prompt:
                augmented_captions = [f'{self.default_prompt} {caption}' for caption in text_captions]
            if self.default_negative_prompt:
                augmented_negative_prompt = [f'{self.default_negative_prompt}' for _ in text_captions]

            if self.sdxl:
                crop_params = torch.tensor([0, 0]).unsqueeze(0)
                input_size_params = torch.tensor([self.size, self.size]).unsqueeze(0)
            else:
                crop_params = None
                input_size_params = None
            # Generate images from the captions
            with get_precision_context(self.precision):
                generated_images = self.model.generate(prompt=augmented_captions,
                                                       negative_prompt=augmented_negative_prompt,
                                                       height=self.size,
                                                       width=self.size,
                                                       guidance_scale=guidance_scale,
                                                       seed=seed,
                                                       crop_params=crop_params,
                                                       input_size_params=input_size_params,
                                                       progress_bar=False,
                                                       **self.additional_generate_kwargs)  # type: ignore
            self.clip_metric.update((generated_images * 255).to(torch.uint8), text_captions)
            # Save the real images
            # Verify that the real images are in the proper range
            if real_images.min() < 0.0 or real_images.max() > 1.0:
                raise ValueError(
                    f'Images are expected to be in the range [0, 1]. Got max {real_images.max()} and min {real_images.min()}'
                )
            for i, img in enumerate(real_images):
                to_pil_image(img).save(f'{real_image_path}/{sample_id}_rank_{dist.get_local_rank()}.png')
                prompts[f'{sample_id}_rank_{dist.get_local_rank()}'] = text_captions[i]
            # Save the generated images
            for i, img in enumerate(generated_images):
                to_pil_image(img).save(f'{gen_image_path}/{sample_id}_rank_{dist.get_local_rank()}.png')

        # Save the prompts as json
        json.dump(prompts, open(f'{real_image_path}/prompts_rank_{dist.get_local_rank()}.json', 'w'))

    def _compute_metrics(self, guidance_scale: float):
        """Compute metrics for the generated images at a given guidance scale.

        Args:
            guidance_scale (float): The guidance scale to use for image generation.

        Returns:
            Dict[str, float]: The computed metrics.
        """
        # Path to find the generated images in
        real_image_path = os.path.join(self.output_dir, f'real_images_gs_{guidance_scale}')
        gen_image_path = os.path.join(self.output_dir, f'gen_images_gs_{guidance_scale}')

        metrics = {}
        # CLIP score
        clip_score = self.clip_metric.compute()
        metrics['CLIP-score'] = clip_score
        print(f'{guidance_scale} CLIP score: {clip_score}')

        # Need to tell clean-fid which device to use
        device = torch.device(self.device)
        # Standard FID
        fid_score = fid.compute_fid(real_image_path,
                                    gen_image_path,
                                    device=device,
                                    use_dataparallel=False,
                                    verbose=False)
        metrics['FID'] = fid_score
        print(f'{guidance_scale} FID: {fid_score}')
        # CLIP-FID from https://arxiv.org/abs/2203.06026
        clip_fid_score = fid.compute_fid(real_image_path,
                                         gen_image_path,
                                         mode='clean',
                                         model_name='clip_vit_b_32',
                                         device=device,
                                         use_dataparallel=False,
                                         verbose=False)
        metrics['CLIP-FID'] = clip_fid_score
        print(f'{guidance_scale} CLIP-FID: {clip_fid_score}')
        # KID
        kid_score = fid.compute_kid(real_image_path,
                                    gen_image_path,
                                    device=device,
                                    use_dataparallel=False,
                                    verbose=False)
        metrics['KID'] = kid_score
        print(f'{guidance_scale} KID: {kid_score}')
        return metrics

    def _generate_images_from_prompts(self, guidance_scale: float):
        """Generate images from prompts for visualization."""
        if self.prompts:
            # Augment the prompt
            augmented_prompts = self.prompts
            if self.default_prompt:
                augmented_prompts = [f'{self.default_prompt} {prompt}' for prompt in self.prompts]
            # Augment the negative prompt
            augmented_negative_prompts = None
            if 'negative prompt' in self.additional_generate_kwargs:
                negative_prompts = self.additional_generate_kwargs['negative prompt']
                augmented_negative_prompts = [
                    f'{self.default_negative_prompt} {neg_prompt}' for neg_prompt in negative_prompts
                ]
            if self.default_negative_prompt and augmented_negative_prompts is None:
                augmented_negative_prompts = [f'{self.default_negative_prompt}' for _ in self.prompts]

            with get_precision_context(self.precision):
                generated_images = self.model.generate(prompt=augmented_prompts,
                                                       negative_prompt=augmented_negative_prompts,
                                                       height=self.size,
                                                       width=self.size,
                                                       guidance_scale=guidance_scale,
                                                       seed=self.seed,
                                                       **self.additional_generate_kwargs)  # type: ignore
        else:
            generated_images = []
        return generated_images

    def evaluate(self):
        # Generate images and compute metrics for each guidance scale
        for guidance_scale in self.guidance_scales:
            dist.barrier()
            # Generate images and compute metrics
            self._generate_images(guidance_scale=guidance_scale)
            # Need to wait until all ranks have finished generating images before computing metrics
            dist.barrier()
            # Compute the metrics on the generated images
            metrics = self._compute_metrics(guidance_scale=guidance_scale)
            # Generate images from prompts for visualization
            generated_images = self._generate_images_from_prompts(guidance_scale=guidance_scale)
            # Log metrics and images on rank 0
            if self.loggers and dist.get_local_rank() == 0:
                for logger in self.loggers:
                    for metric, value in metrics.items():
                        logger.log_metrics({f'{guidance_scale}/{metric}': value})
                    for prompt, image in zip(self.prompts, generated_images):
                        logger.log_images(images=image, name=f'{prompt}_gs_{guidance_scale}')
