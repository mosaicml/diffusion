# Copyright 2022 MosaicML Diffusion authors
# SPDX-License-Identifier: Apache-2.0

"""Inference endpoint for Stable Diffusion."""

import base64
import io
from typing import Any, Dict, List, Optional

import torch
from composer.utils.file_helpers import get_file
from PIL import Image

from diffusion.models import stable_diffusion_2, stable_diffusion_xl

# Local checkpoint params
LOCAL_CHECKPOINT_PATH = '/tmp/model.pt'


def download_checkpoint(chkpt_path: str):
    """Downloads the Stable Diffusion checkpoint to the local filesystem.

    Args:
        chkpt_path (str): The path to the local folder, URL or object score that contains the checkpoint.
    """
    get_file(path=chkpt_path, destination=LOCAL_CHECKPOINT_PATH)


class StableDiffusionInference():
    """Inference endpoint class for Stable Diffusion 2.

    Args:
        model_name (str, optional): Name of the model to load. Default: 'stabilityai/stable-diffusion-2-base'.
        pretrained (bool): Whether to load pretrained weights. Default: True.
        prediction_type (str): The type of prediction to use. Must be one of 'sample',
            'epsilon', or 'v_prediction'. Default: `epsilon`.
        **kwargs: Additional keyword arguments to pass to the model.
    """

    def __init__(self,
                 model_name: str = 'stabilityai/stable-diffusion-2-base',
                 pretrained: bool = False,
                 prediction_type: str = 'epsilon',
                 **kwargs):
        self.device = torch.cuda.current_device()

        model = stable_diffusion_2(
            model_name=model_name,
            pretrained=pretrained,
            prediction_type=prediction_type,
            encode_latents_in_fp16=True,
            fsdp=False,
            **kwargs,
        )

        if not pretrained:
            state_dict = torch.load(LOCAL_CHECKPOINT_PATH)
            for key in list(state_dict['state']['model'].keys()):
                if 'val_metrics.' in key:
                    del state_dict['state']['model'][key]
            model.load_state_dict(state_dict['state']['model'], strict=False)
        model.to(self.device)
        self.model = model.eval()

    def predict(self, model_requests: List[Dict[str, Any]]):
        prompts = []
        negative_prompts = []
        generate_kwargs = {}

        # assumes the same generate_kwargs across all samples
        for req in model_requests:
            if 'input' not in req:
                raise RuntimeError('"input" must be provided to generate call')
            inputs = req['input']

            # Prompts and negative prompts if available
            if isinstance(inputs, str):
                prompts.append(inputs)
            elif isinstance(inputs, Dict):
                if 'prompt' not in inputs:
                    raise RuntimeError('"prompt" must be provided to generate call if using a dict as input')
                prompts.append(inputs['prompt'])
                if 'negative_prompt' in inputs:
                    negative_prompts.append(inputs['negative_prompt'])
            else:
                raise RuntimeError(f'Input must be of type string or dict, but it is type: {type(inputs)}')

            generate_kwargs = req['parameters']

        # Check for prompts
        if len(prompts) == 0:
            raise RuntimeError('No prompts provided, must be either a string or dictionary with "prompt"')

        # Check negative prompt length
        if len(negative_prompts) == 0:
            negative_prompts = None
        elif len(prompts) != len(negative_prompts):
            raise RuntimeError('There must be the same number of negative prompts as prompts.')

        # Generate images
        with torch.cuda.amp.autocast(True):
            imgs = self.model.generate(prompt=prompts, negative_prompt=negative_prompts, **generate_kwargs).cpu()

        # Send as bytes
        png_images = []
        for i in range(imgs.shape[0]):
            img = (imgs[i].permute(1, 2, 0).numpy() * 255).round().astype('uint8')
            pil_image = Image.fromarray(img, 'RGB')
            img_byte_arr = io.BytesIO()
            pil_image.save(img_byte_arr, format='PNG')
            base64_encoded_image = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')
            png_images.append(base64_encoded_image)
        return png_images


class StableDiffusionXLInference():
    """Inference endpoint class for Stable Diffusion XL.

    Args:
        model_name (str): Name of the model to load. Default: 'stabilityai/stable-diffusion-xl-base-1.0'.
        unet_model_name (str): Name of the UNet model to load. Default: 'stabilityai/stable-diffusion-xl-base-1.0'.
        vae_model_name (str): Name of the VAE model to load. Defaults to
            'madebyollin/sdxl-vae-fp16-fix' as the official VAE checkpoint (from
            'stabilityai/stable-diffusion-xl-base-1.0') is not compatible with fp16.
        clip_qkv (float, optional): If not None, clip the qkv values to this value. Defaults to 6.0. Improves stability
            of training.
        pretrained (bool): Whether to load pretrained weights. Default: True.
        prediction_type (str): The type of prediction to use. Must be one of 'sample',
            'epsilon', or 'v_prediction'. Default: `epsilon`.
        **kwargs: Additional keyword arguments to pass to the model.
    """

    def __init__(self,
                 model_name: str = 'stabilityai/stable-diffusion-xl-base-1.0',
                 unet_model_name: str = 'stabilityai/stable-diffusion-xl-base-1.0',
                 vae_model_name: str = 'madebyollin/sdxl-vae-fp16-fix',
                 clip_qkv: Optional[float] = None,
                 pretrained: bool = False,
                 prediction_type: str = 'epsilon',
                 **kwargs):
        self.device = torch.cuda.current_device()

        model = stable_diffusion_xl(
            model_name=model_name,
            unet_model_name=unet_model_name,
            vae_model_name=vae_model_name,
            clip_qkv=clip_qkv,
            pretrained=pretrained,
            prediction_type=prediction_type,
            encode_latents_in_fp16=True,
            fsdp=False,
            **kwargs,
        )

        if not pretrained:
            state_dict = torch.load(LOCAL_CHECKPOINT_PATH)
            for key in list(state_dict['state']['model'].keys()):
                if 'val_metrics.' in key:
                    del state_dict['state']['model'][key]
            model.load_state_dict(state_dict['state']['model'], strict=False)
        model.to(self.device)
        self.model = model.eval()

    def predict(self, model_requests: List[Dict[str, Any]]):
        prompts = []
        negative_prompts = []
        generate_kwargs = {}

        # assumes the same generate_kwargs across all samples
        for req in model_requests:
            if 'input' not in req:
                raise RuntimeError('"input" must be provided to generate call')
            inputs = req['input']

            # Prompts and negative prompts if available
            if isinstance(inputs, str):
                prompts.append(inputs)
            elif isinstance(inputs, Dict):
                if 'prompt' not in inputs:
                    raise RuntimeError('"prompt" must be provided to generate call if using a dict as input')
                prompts.append(inputs['prompt'])
                if 'negative_prompt' in inputs:
                    negative_prompts.append(inputs['negative_prompt'])
            else:
                raise RuntimeError(f'Input must be of type string or dict, but it is type: {type(inputs)}')

            generate_kwargs = req['parameters']

        # Check for prompts
        if len(prompts) == 0:
            raise RuntimeError('No prompts provided, must be either a string or dictionary with "prompt"')

        # Check negative prompt length
        if len(negative_prompts) == 0:
            negative_prompts = None
        elif len(prompts) != len(negative_prompts):
            raise RuntimeError('There must be the same number of negative prompts as prompts.')

        # Generate images
        with torch.cuda.amp.autocast(True):
            imgs = self.model.generate(prompt=prompts, negative_prompt=negative_prompts, **generate_kwargs).cpu()

        # Send as bytes
        png_images = []
        for i in range(imgs.shape[0]):
            img = (imgs[i].permute(1, 2, 0).numpy() * 255).round().astype('uint8')
            pil_image = Image.fromarray(img, 'RGB')
            img_byte_arr = io.BytesIO()
            pil_image.save(img_byte_arr, format='PNG')
            base64_encoded_image = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')
            png_images.append(base64_encoded_image)
        return png_images
