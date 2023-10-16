# Copyright 2022 MosaicML Diffusion authors
# SPDX-License-Identifier: Apache-2.0

"""Evaluation using the clean-fid package."""

import json
import os
from typing import List, Optional

import clip
import torch
import wandb
from cleanfid import fid
from composer import ComposerModel, Trainer
from composer.core import get_precision_context
from composer.loggers import LoggerDestination, WandBLogger
from composer.utils import dist
from torch.utils.data import DataLoader
from torchmetrics.multimodal import CLIPScore
from tqdm.auto import tqdm
from transformers import PreTrainedTokenizerBase

try:
    from torchvision.transforms.v2.functional import to_pil_image
except ImportError:
    from torchvision.transforms.functional import to_pil_image

os.environ['TOKENIZERS_PARALLELISM'] = 'false'


class CleanFIDEvaluator:
    """Evaluator for CLIP, FID, KID, CLIP-FID scores using clean-fid.

    See https://github.com/GaParmar/clean-fid for more information on clean-fid.

    CLIP scores are computed using the torchmetrics CLIPScore metric.

    Args:
        model (ComposerModel): The model to evaluate.
        eval_dataloader (DataLoader): The dataloader to use for evaluation.
        clip_metric (CLIPScore): The CLIPScore metric to use for evaluation.
        load_path (str, optional): The path to load the model from. Default: ``None``.
        guidance_scales (List[float]): The guidance scales to use for evaluation.
            Default: ``[1.0]``.
        size (int): The size of the images to generate. Default: ``256``.
        batch_size (int): The per-device batch size to use for evaluation. Default: ``16``.
        loggers (List[LoggerDestination], optional): The loggers to use for logging results. Default: ``None``.
        seed (int): The seed to use for evaluation. Default: ``17``.
        output_dir (str): The directory to save results to. Default: ``/tmp/``.
        num_samples (int, optional): The maximum number of samples to generate. Depending on batch size, actual
            number may be slightly higher. If not specified, all the samples in the dataloader will be used.
            Default: ``None``.
        precision (str): The precision to use for evaluation. Default: ``'amp_fp16'``.
        prompts (List[str], optional): The prompts to use for image visualtization.
            Default: ``["A shiba inu wearing a blue sweater]``.

    """

    def __init__(self,
                 model: ComposerModel,
                 eval_dataloader: DataLoader,
                 clip_metric: CLIPScore,
                 load_path: Optional[str] = None,
                 guidance_scales: Optional[List[float]] = None,
                 size: int = 256,
                 batch_size: int = 16,
                 image_key: str = 'image',
                 caption_key: str = 'caption',
                 loggers: Optional[List[LoggerDestination]] = None,
                 seed: int = 17,
                 output_dir: str = '/tmp/',
                 num_samples: Optional[int] = None,
                 precision: str = 'amp_fp16',
                 prompts: Optional[List[str]] = None):
        self.model = model
        self.tokenizer: PreTrainedTokenizerBase = model.tokenizer
        self.eval_dataloader = eval_dataloader
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
        self.num_samples = num_samples if num_samples is not None else float('inf')
        self.precision = precision
        self.prompts = prompts if prompts is not None else ['A shiba inu wearing a blue sweater']

        # Init loggers
        if self.loggers and dist.get_local_rank() == 0:
            for logger in self.loggers:
                if isinstance(logger, WandBLogger):
                    wandb.init(**logger._init_kwargs)

        # Load the model
        Trainer(model=self.model,
                load_path=self.load_path,
                load_weights_only=True,
                eval_dataloader=self.eval_dataloader,
                seed=self.seed)

        # Move CLIP metric to device
        self.device = dist.get_local_rank()
        self.clip_metric = self.clip_metric.to(self.device)

        # Predownload the CLIP model for computing clip-fid
        _, _ = clip.load('ViT-B/32', device=self.device)

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
        # Iterate over the eval dataloader
        num_batches = len(self.eval_dataloader)
        starting_seed = self.seed + num_batches * dist.get_local_rank()
        for batch_id, batch in tqdm(enumerate(self.eval_dataloader)):
            # Break if enough samples have been generated
            if batch_id * self.batch_size * dist.get_world_size() >= self.num_samples:
                break

            real_images = batch[self.image_key]
            captions = batch[self.caption_key]
            # Ensure a new seed for each batch, as randomness in model.generate is fixed.
            seed = starting_seed + batch_id
            # Generate images from the captions
            with get_precision_context(self.precision):
                generated_images = self.model.generate(tokenized_prompts=captions,
                                                       height=self.size,
                                                       width=self.size,
                                                       guidance_scale=guidance_scale,
                                                       seed=seed,
                                                       progress_bar=False)  # type: ignore
            # Get the prompts from the tokens
            text_captions = self.tokenizer.batch_decode(captions, skip_special_tokens=True)
            self.clip_metric.update((generated_images * 255).to(torch.uint8), text_captions)
            # Save the real images
            # Verify that the real images are in the proper range
            if real_images.min() < 0.0 or real_images.max() > 1.0:
                raise ValueError(
                    f'Images are expected to be in the range [0, 1]. Got max {real_images.max()} and min {real_images.min()}'
                )
            for i, img in enumerate(real_images):
                to_pil_image(img).save(f'{real_image_path}/{batch_id}_{i}_rank_{dist.get_local_rank()}.png')
                prompts[f'{batch_id}_{i}_rank_{dist.get_local_rank()}'] = text_captions[i]
            # Save the generated images
            for i, img in enumerate(generated_images):
                to_pil_image(img).save(f'{gen_image_path}/{batch_id}_{i}_rank_{dist.get_local_rank()}.png')

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
            with get_precision_context(self.precision):
                generated_images = self.model.generate(prompt=self.prompts,
                                                       height=self.size,
                                                       width=self.size,
                                                       guidance_scale=guidance_scale,
                                                       seed=self.seed)  # type: ignore
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
