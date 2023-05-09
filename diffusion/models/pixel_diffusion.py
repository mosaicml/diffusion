# Copyright 2022 MosaicML Diffusion authors
# SPDX-License-Identifier: Apache-2.0

"""Pixel space diffusion models."""

from typing import List, Optional

import torch
from composer.models import ComposerModel
from torchmetrics import MeanSquaredError, Metric
from tqdm.auto import tqdm


class PixelSpaceDiffusion(ComposerModel):
    """
    TODO: Docstring
    """

    def __init__(self,
                 model,
                 text_encoder,
                 tokenizer,
                 scheduler,
                 inference_scheduler=None,
                 continuous_time=False,
                 input_key='image',
                 conditioning_key='captions',
                 prediction_type='epsilon',
                 train_metrics: Optional[List] = None,
                 val_metrics: Optional[List] = None,
                 val_guidance_scales: List = [],
                 negative_conditioning: Optional[torch.FloatTensor] = None):
        super().__init__()
        self.model = model
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.scheduler = scheduler
        self.inference_scheduler = inference_scheduler
        self.input_key = input_key
        self.conditioning_key = conditioning_key
        self.prediction_type = prediction_type
        self.train_metrics = train_metrics
        self.val_metrics = val_metrics
        self.val_guidance_scales = val_guidance_scales
        self.negative_conditioning = negative_conditioning

        # freeze text_encoder training
        self.text_encoder.requires_grad_(False)

    def forward(self, batch):
        inputs, conditioning = batch[self.input_key], batch[self.conditioning_key]
        # Encode the conditioning
        conditioning = self.text_encoder(conditioning)[0]
        # Sample the diffusion timesteps
        timesteps = torch.randint(0, len(self.scheduler), (inputs.shape[0],), device=inputs.device)
        # Add noise to the inputs (forward diffusion)
        noise = torch.randn_like(inputs)
        noised_inputs = self.scheduler.add_noise(inputs, noise, timesteps)
        # Generate the targets
        if self.prediction_type == 'epsilon':
            targets = noise
        elif self.prediction_type == 'sample':
            targets = inputs
        elif self.prediction_type == 'v_prediction':
            targets = self.scheduler.get_velocity(inputs, noise, timesteps)
        # Forward through the model
        return self.model(noised_inputs, timesteps, conditioning)['sample'], targets, timesteps

    def loss(self, outputs, batch):
        return torch.nn.functional.mse_loss(outputs[0], outputs[1])

    def eval_forward(self, batch, outputs=None):
        if outputs is not None:
            return outputs
        # Get model outputs
        model_out, targets, timesteps = self.forward(batch)
        # Sample images from the conditioning in the batch
        images = batch[self.input_key]
        conditioning = batch[self.conditioning_key]
        height, width = images.shape[-2], images.shape[-1]
        generated_images = {}
        for guidance_scale in self.val_guidance_scales:
            gen_images = self.generate(tokenized_prompts=conditioning,
                                       height=height,
                                       width=width,
                                       guidance_scale=guidance_scale,
                                       seed=self.val_seed,
                                       progress_bar=False)
            generated_images[guidance_scale] = gen_images
        return model_out, targets, timesteps, generated_images

    def get_metrics(self, is_train: bool = False):
        if is_train:
            metrics = self.train_metrics
        else:
            metrics = self.val_metrics

        if isinstance(metrics, Metric):
            metrics_dict = {metrics.__class__.__name__: metrics}
        elif isinstance(metrics, list):
            metrics_dict = {metric.__class__.__name__: metric for metric in metrics}
        else:
            metrics_dict = {}
            for name, metric in metrics.items():
                assert isinstance(metric, Metric)
                metrics_dict[name] = metric

        return metrics_dict

    def update_metric(self, batch, outputs, metric):
        if isinstance(metric, MeanSquaredError):
            metric.update(outputs[0], outputs[1])
        else:
            raise NotImplementedError(f'Metric {metric.__class__.__name__} not implemented.')

    @torch.no_grad()
    def generate(
        self,
        prompt: Optional[list] = None,
        negative_prompt: Optional[list] = None,
        tokenized_prompts: Optional[torch.LongTensor] = None,
        tokenized_negative_prompts: Optional[torch.LongTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        height: int = 64,
        width: int = 64,
        num_inference_steps: Optional[int] = 50,
        guidance_scale: Optional[float] = 3.0,
        num_images_per_prompt: Optional[int] = 1,
        seed: Optional[int] = None,
        progress_bar: Optional[bool] = True,
    ):
        # Create rng for the generation
        device = self.model.device
        rng_generator = torch.Generator(device=device)
        if seed:
            rng_generator = rng_generator.manual_seed(seed)  # type: ignore

        do_classifier_free_guidance = guidance_scale > 1.0  # type: ignore

        text_embeddings = self._prepare_text_embeddings(prompt, tokenized_prompts, prompt_embeds, num_images_per_prompt)
        batch_size = len(text_embeddings)  # len prompts * num_images_per_prompt
        # classifier free guidance + negative prompts
        # negative prompt is given in place of the unconditional input in classifier free guidance
        if do_classifier_free_guidance:
            negative_prompt = negative_prompt or ([''] * (batch_size // num_images_per_prompt))  # type: ignore
            unconditional_embeddings = self._prepare_text_embeddings(negative_prompt, tokenized_negative_prompts,
                                                                     negative_prompt_embeds, num_images_per_prompt)
            # concat uncond + prompt
            text_embeddings = torch.cat([unconditional_embeddings, text_embeddings])

        # prepare for diffusion generation process
        images = torch.randn((batch_size, 3, height, width), device=device, generator=rng_generator)

        self.inference_scheduler.set_timesteps(num_inference_steps)
        # scale the initial noise by the standard deviation required by the scheduler
        images = images * self.inference_scheduler.init_noise_sigma

        # backward diffusion process
        for t in tqdm(self.inference_scheduler.timesteps, disable=not progress_bar):
            if do_classifier_free_guidance:
                model_input = torch.cat([images] * 2)
            else:
                model_input = images

            model_input = self.inference_scheduler.scale_model_input(model_input, t)
            # get model's predicted output
            model_output = self.model(model_input, t, encoder_hidden_states=text_embeddings).sample

            if do_classifier_free_guidance:
                # perform guidance. Not this is technically incorrect unless prediction_type is 'epsilon'
                pred_uncond, pred_text = model_output.chunk(2)
                model_output = pred_uncond + guidance_scale * (pred_text - pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            images = self.inference_scheduler.step(model_output, t, images, generator=rng_generator).prev_sample

        # Rescale to (0, 1)
        images = (images / 2 + 0.5).clamp(0, 1)
        return images.detach()  # (batch*num_images_per_prompt, channel, h, w)

    def _prepare_text_embeddings(self, prompt, tokenized_prompts, prompt_embeds, num_images_per_prompt):
        """Tokenizes and embeds prompts if needed, then duplicates embeddings to support multiple generations per prompt."""
        device = self.text_encoder.device
        if prompt_embeds is None:
            if tokenized_prompts is None:
                tokenized_prompts = self.tokenizer(prompt,
                                                   padding='max_length',
                                                   max_length=self.tokenizer.model_max_length,
                                                   truncation=True,
                                                   return_tensors='pt').input_ids
            text_embeddings = self.text_encoder(tokenized_prompts.to(device))[0]  # type: ignore
        else:
            text_embeddings = prompt_embeds

        # duplicate text embeddings for each generation per prompt
        bs_embed, seq_len, _ = text_embeddings.shape
        text_embeddings = text_embeddings.repeat(1, num_images_per_prompt, 1)  # type: ignore
        text_embeddings = text_embeddings.view(bs_embed * num_images_per_prompt, seq_len, -1)
        return text_embeddings
