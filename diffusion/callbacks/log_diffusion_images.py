# Copyright 2022 MosaicML Diffusion authors
# SPDX-License-Identifier: Apache-2.0

"""Logger for generated images."""

import gc
from math import ceil
from typing import List, Optional, Tuple, Union

import torch
from composer import Callback, Logger, State
from composer.core import TimeUnit, get_precision_context
from torch.nn.parallel import DistributedDataParallel
from transformers import AutoModel, AutoTokenizer, CLIPTextModel


class LogDiffusionImages(Callback):
    """Logs images generated from the evaluation prompts to a logger.

    Logs eval prompts and generated images to a table at
    the end of an evaluation batch.

    Args:
        prompts (List[str]): List of prompts to use for evaluation.
        size (int, Tuple[int, int]): Image size to use during generation.
            If using a tuple, specify as (height, width).  Default: ``256``.
        batch_size (int, optional): The batch size of the prompts passed to the generate function. If set to ``None``,
            batch_size is equal to the number of prompts. Default: ``1``.
        num_inference_steps (int): Number of inference steps to use during generation. Default: ``50``.
        guidance_scale (float): guidance_scale is defined as w of equation 2
            of the Imagen Paper. Guidance scale is enabled by setting guidance_scale > 1.
            A larger guidance scale generates images that are more aligned to
            the text prompt, usually at the expense of lower image quality.
            Default: ``0.0``.
        rescaled_guidance (float, optional): Rescaled guidance scale. If not specified, rescaled guidance
            will not be used. Default: ``None``.
        seed (int, optional): Random seed to use for generation. Set a seed for reproducible generation.
            Default: ``1138``.
        use_table (bool): Whether to make a table of the images or not. Default: ``False``.
        t5_encoder (str, optional): path to the T5 encoder to as a second text encoder.
        clip_encoder (str, optional): path to the CLIP encoder as the first text encoder.
        cache_dir: (str, optional): path for HF to cache files while downloading model
    """

    def __init__(self,
                 prompts: List[str],
                 size: Union[Tuple[int, int], int] = 256,
                 batch_size: Optional[int] = 1,
                 num_inference_steps: int = 50,
                 guidance_scale: float = 0.0,
                 rescaled_guidance: Optional[float] = None,
                 seed: Optional[int] = 1138,
                 use_table: bool = False,
                 t5_encoder: Optional[str] = None,
                 clip_encoder: Optional[str] = None,
                 cache_dir: Optional[str] = '/tmp/hf_files'):
        self.prompts = prompts
        self.size = (size, size) if isinstance(size, int) else size
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.rescaled_guidance = rescaled_guidance
        self.seed = seed
        self.use_table = use_table
        self.cache_dir = cache_dir

        # Batch prompts
        batch_size = len(prompts) if batch_size is None else batch_size
        num_batches = ceil(len(prompts) / batch_size)
        self.batched_prompts = []
        for i in range(num_batches):
            start, end = i * batch_size, (i + 1) * batch_size
            self.batched_prompts.append(prompts[start:end])

        if t5_encoder is not None and clip_encoder is None or t5_encoder is None and clip_encoder is not None:
            raise ValueError('Cannot specify only one of text encoder and CLIP encoder.')

        self.precomputed_latents = False
        self.batched_latents = []
        if t5_encoder:
            self.precomputed_latents = True
            t5_tokenizer = AutoTokenizer.from_pretrained(t5_encoder, cache_dir=self.cache_dir, local_files_only=True)
            clip_tokenizer = AutoTokenizer.from_pretrained(clip_encoder,
                                                           subfolder='tokenizer',
                                                           cache_dir=self.cache_dir,
                                                           local_files_only=True)

            t5_model = AutoModel.from_pretrained(t5_encoder,
                                                 torch_dtype=torch.float16,
                                                 cache_dir=self.cache_dir,
                                                 local_files_only=True).encoder.cuda().eval()
            clip_model = CLIPTextModel.from_pretrained(clip_encoder,
                                                       subfolder='text_encoder',
                                                       torch_dtype=torch.float16,
                                                       cache_dir=self.cache_dir,
                                                       local_files_only=True).cuda().eval()

            for batch in self.batched_prompts:
                latent_batch = {}
                tokenized_t5 = t5_tokenizer(batch,
                                            padding='max_length',
                                            max_length=t5_tokenizer.model_max_length,
                                            truncation=True,
                                            return_tensors='pt')
                t5_attention_mask = tokenized_t5['attention_mask'].to(torch.bool).cuda()
                t5_ids = tokenized_t5['input_ids'].cuda()
                t5_latents = t5_model(input_ids=t5_ids, attention_mask=t5_attention_mask)[0].cpu()
                t5_attention_mask = t5_attention_mask.cpu().to(torch.long)

                tokenized_clip = clip_tokenizer(batch,
                                                padding='max_length',
                                                max_length=clip_tokenizer.model_max_length,
                                                truncation=True,
                                                return_tensors='pt')
                clip_attention_mask = tokenized_clip['attention_mask'].cuda()
                clip_ids = tokenized_clip['input_ids'].cuda()
                clip_outputs = clip_model(input_ids=clip_ids,
                                          attention_mask=clip_attention_mask,
                                          output_hidden_states=True)
                clip_latents = clip_outputs.hidden_states[-2].cpu()
                clip_pooled = clip_outputs[1].cpu()
                clip_attention_mask = clip_attention_mask.cpu().to(torch.long)

                latent_batch['T5_LATENTS'] = t5_latents
                latent_batch['CLIP_LATENTS'] = clip_latents
                latent_batch['ATTENTION_MASK'] = torch.cat([t5_attention_mask, clip_attention_mask], dim=1)
                latent_batch['CLIP_POOLED'] = clip_pooled
                self.batched_latents.append(latent_batch)

            del t5_model
            del clip_model
            gc.collect()
            torch.cuda.empty_cache()

    def eval_start(self, state: State, logger: Logger):
        # Get the model object if it has been wrapped by DDP to access the image generation function.
        if isinstance(state.model, DistributedDataParallel):
            model = state.model.module
        else:
            model = state.model

        # Generate images
        with get_precision_context(state.precision):
            all_gen_images = []
            if self.precomputed_latents:
                for batch in self.batched_latents:
                    pooled_prompt = batch['CLIP_POOLED'].cuda()
                    prompt_mask = batch['ATTENTION_MASK'].cuda()
                    t5_embeds = model.t5_proj(batch['T5_LATENTS'].cuda())
                    clip_embeds = model.clip_proj(batch['CLIP_LATENTS'].cuda())
                    prompt_embeds = torch.cat([t5_embeds, clip_embeds], dim=1)

                    gen_images = model.generate(prompt_embeds=prompt_embeds,
                                                pooled_prompt=pooled_prompt,
                                                prompt_mask=prompt_mask,
                                                height=self.size[0],
                                                width=self.size[1],
                                                guidance_scale=self.guidance_scale,
                                                rescaled_guidance=self.rescaled_guidance,
                                                progress_bar=False,
                                                num_inference_steps=self.num_inference_steps,
                                                seed=self.seed)
                    all_gen_images.append(gen_images)
            else:
                for batch in self.batched_prompts:
                    gen_images = model.generate(
                        prompt=batch,  # type: ignore
                        height=self.size[0],
                        width=self.size[1],
                        guidance_scale=self.guidance_scale,
                        rescaled_guidance=self.rescaled_guidance,
                        progress_bar=False,
                        num_inference_steps=self.num_inference_steps,
                        seed=self.seed)
                    all_gen_images.append(gen_images)
            gen_images = torch.cat(all_gen_images)

        # Log images to wandb
        for prompt, image in zip(self.prompts, gen_images):
            logger.log_images(images=image, name=prompt, step=state.timestamp.batch.value, use_table=self.use_table)


class LogAutoencoderImages(Callback):
    """Logs images from an autoencoder to compare real inputs to their autoencoded outputs.

    Args:
        image_key (str): Key in the batch to use for images. Default: ``'image'``.
        max_images (int): Maximum number of images to log. Default: ``10``.
        log_latents (bool): Whether to log the latents or not. Default: ``True``.
        use_table (bool): Whether to make a table of the images or not. Default: ``False``.
    """

    def __init__(self,
                 image_key: str = 'image',
                 max_images: int = 10,
                 log_latents: bool = True,
                 use_table: bool = False):
        self.image_key = image_key
        self.max_images = max_images
        self.log_latents = log_latents
        self.use_table = use_table

    def _scale_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Scale latents to be between 0 and 1 for visualization."""
        latents = latents - latents.min()
        latents = latents / latents.max()
        return latents

    def eval_batch_end(self, state: State, logger: Logger):
        # Only log once per eval epoch
        if state.eval_timestamp.get(TimeUnit.BATCH).value == 1:
            # Get the inputs
            images = state.batch[self.image_key]
            if self.max_images > images.shape[0]:
                max_images = images.shape[0]
            else:
                max_images = self.max_images

            # Get the model reconstruction
            outputs = state.model(state.batch)
            recon = outputs['x_recon']
            latents = outputs['latents']

            # Log images to wandb
            for i, image in enumerate(images[:max_images]):
                # Clamp the reconstructed image to be between 0 and 1
                recon_img = (recon[i] / 2 + 0.5).clamp(0, 1)
                logged_images = [image, recon_img]
                if self.log_latents:
                    logged_images += [self._scale_latents(latents[i][j]) for j in range(latents.shape[1])]
                logger.log_images(images=logged_images,
                                  name=f'Image (input, reconstruction, latents) {i}',
                                  step=state.timestamp.batch.value,
                                  use_table=self.use_table)
