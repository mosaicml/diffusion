# Copyright 2022 MosaicML Diffusion authors
# SPDX-License-Identifier: Apache-2.0

"""Logger for generated images."""

from typing import List, Optional

import torch
from composer import Callback, Logger, State
from composer.core import TimeUnit, get_precision_context
from torch.nn.parallel import DistributedDataParallel


class LogDiffusionImages(Callback):
    """Logs images generated from the evaluation prompts to a logger.

    Logs eval prompts and generated images to a table at
    the end of an evaluation batch.

    Args:
        prompts (List[str]): List of prompts to use for evaluation.
        size (int, optional): Image size to use during generation. Default: ``256``.
        num_inference_steps (int, optional): Number of inference steps to use during generation. Default: ``50``.
        guidance_scale (float, optional): guidance_scale is defined as w of equation 2
            of the Imagen Paper. Guidance scale is enabled by setting guidance_scale > 1.
            A larger guidance scale generates images that are more aligned to
            the text prompt, usually at the expense of lower image quality.
            Default: ``0.0``.
        text_key (str, optional): Key in the batch to use for text prompts. Default: ``'captions'``.
        tokenized_prompts (torch.LongTensor or List[torch.LongTensor], optional): Batch of pre-tokenized prompts
            to use for evaluation. If SDXL, this will be a list of two pre-tokenized prompts Default: ``None``.
        seed (int, optional): Random seed to use for generation. Set a seed for reproducible generation.
            Default: ``1138``.
        use_table (bool): Whether to make a table of the images or not. Default: ``False``.
    """

    def __init__(self,
                 prompts: List[str],
                 size: Optional[int] = 256,
                 num_inference_steps=50,
                 guidance_scale: Optional[float] = 0.0,
                 text_key: Optional[str] = 'captions',
                 tokenized_prompts: Optional[torch.LongTensor] = None,
                 seed: Optional[int] = 1138,
                 use_table: bool = False):
        self.prompts = prompts
        self.size = size
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.text_key = text_key
        self.seed = seed
        self.tokenized_prompts = tokenized_prompts
        self.use_table = use_table

    def eval_batch_end(self, state: State, logger: Logger):
        # Only log once per eval epoch
        if state.eval_timestamp.get(TimeUnit.BATCH).value == 1:
            # Get the model object if it has been wrapped by DDP
            # We need this to access the text keys, tokenizer, and image generation function.
            if isinstance(state.model, DistributedDataParallel):
                model = state.model.module
            else:
                model = state.model

            if self.tokenized_prompts is None:
                self.tokenized_prompts = [
                    model.tokenizer(p, padding='max_length', truncation=True,
                                    return_tensors='pt')['input_ids']  # type: ignore
                    for p in self.prompts
                ]
                if model.sdxl:
                    self.tokenized_prompts = [
                        torch.cat([tp[0] for tp in self.tokenized_prompts]),
                        torch.cat([tp[1] for tp in self.tokenized_prompts])
                    ]
                else:
                    self.tokenized_prompts = torch.cat(self.tokenized_prompts)  # type: ignore
            if model.sdxl:
                self.tokenized_prompts[0] = self.tokenized_prompts[0].to(state.batch[self.text_key].device)
                self.tokenized_prompts[1] = self.tokenized_prompts[1].to(state.batch[self.text_key].device)
            else:
                self.tokenized_prompts = self.tokenized_prompts.to(state.batch[self.text_key].device)  # type: ignore

            # Generate images
            with get_precision_context(state.precision):
                gen_images = model.generate(
                    tokenized_prompts=self.tokenized_prompts,  # type: ignore
                    height=self.size,
                    width=self.size,
                    guidance_scale=self.guidance_scale,
                    progress_bar=False,
                    num_inference_steps=self.num_inference_steps,
                    seed=self.seed)

            # Log images to wandb
            for prompt, image in zip(self.prompts, gen_images):
                logger.log_images(images=image, name=prompt, step=state.timestamp.batch.value, use_table=self.use_table)
