# Copyright 2022 MosaicML Diffusion authors
# SPDX-License-Identifier: Apache-2.0

"""Inference endpoint for Stable Diffusion."""

import base64
import io

import torch
from composer.utils.file_helpers import get_file
from PIL import Image

from diffusion.models import stable_diffusion_2

# Local checkpoint params
LOCAL_CHECKPOINT_PATH = '/tmp/model.pt'


def download_model():
    """Download model from remote storage."""
    model_uri = "oci://mosaicml-internal-checkpoints/stable-diffusion-hero-run/4-13-512-ema/ep5-ba850000-rank0.pt"
    get_file(path=model_uri, destination=LOCAL_CHECKPOINT_PATH)


class StableDiffusionInference():
    """Inference endpoint class for Stable Diffusion."""

    def __init__(self):
        pretrained_flag = False
        self.device = torch.cuda.current_device()

        model = stable_diffusion_2(pretrained=pretrained_flag, encode_latents_in_fp16=True, fsdp=False)
        if not pretrained_flag:
            download_model()
            state_dict = torch.load(LOCAL_CHECKPOINT_PATH)
            for key in list(state_dict['state']['model'].keys()):
                if 'val_metrics.' in key:
                    del state_dict['state']['model'][key]
            model.load_state_dict(state_dict['state']['model'], strict=False)
        model.to(self.device)
        self.model = model.eval()

    def predict(self, **inputs):
        if 'prompt' not in inputs:
            print('No prompt provided, returning nothing')
            return

        # Parse and cast args
        kwargs = {}
        for arg in ['prompt', 'negative_prompt']:
            if arg in inputs:
                kwargs[arg] = inputs[arg]
        for arg in ['height', 'width', 'num_inference_steps', 'num_images_per_prompt', 'seed']:
            if arg in inputs:
                kwargs[arg] = int(inputs[arg])
        for arg in ['guidance_scale']:
            if arg in inputs:
                kwargs[arg] = float(inputs[arg])

        prompt = kwargs.pop('prompt')
        prompts = [prompt] if isinstance(prompt, str) else prompt

        # Generate images
        with torch.cuda.amp.autocast(True):
            imgs = self.model.generate(prompt=prompts, **kwargs).cpu()

        # Send as bytes
        png_images = []
        for i in range(imgs.shape[0]):
            img = (imgs[i].permute(1, 2, 0).numpy() * 255).round().astype('uint8')
            pil_image = Image.fromarray(img, 'RGB')
            img_byte_arr = io.BytesIO()
            pil_image.save(img_byte_arr, format='PNG')
            base64_encoded_image = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')
            png_images.append(bytes(base64_encoded_image, 'utf-8'))
        return png_images
