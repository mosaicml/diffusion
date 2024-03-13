# Copyright 2022 MosaicML Diffusion authors
# SPDX-License-Identifier: Apache-2.0

# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from diffusion.models.models import stable_diffusion_2, stable_diffusion_xl


def test_sd2_forward():
    # fp16 vae does not run on cpu
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = stable_diffusion_2(pretrained=False, fsdp=False, encode_latents_in_fp16=False)
    batch_size = 1
    H = 8
    W = 8
    image = torch.randn(batch_size, 3, H, W, device=device)
    latent = torch.randn(batch_size, 4, H // 8, W // 8)
    caption = torch.randint(low=0, high=128, size=(
        batch_size,
        77,
    ), dtype=torch.long, device=device)
    batch = {'image': image, 'captions': caption}
    output, target, _ = model(batch)  # model.forward generates the unet output noise or v_pred target.
    assert output.shape == latent.shape
    assert target.shape == latent.shape


@pytest.mark.parametrize('guidance_scale', [0.0, 3.0])
@pytest.mark.parametrize('negative_prompt', [None, 'so cool'])
def test_sd2_generate(guidance_scale, negative_prompt):
    # fp16 vae does not run on cpu
    model = stable_diffusion_2(pretrained=False, fsdp=False, encode_latents_in_fp16=False)
    output = model.generate(
        prompt='a cool doge',
        negative_prompt=negative_prompt,
        num_inference_steps=1,
        num_images_per_prompt=1,
        height=8,
        width=8,
        guidance_scale=guidance_scale,
        progress_bar=False,
    )
    assert output.shape == (1, 3, 8, 8)


def test_sdxl_forward():
    # fp16 vae does not run on cpu
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = stable_diffusion_xl(pretrained=False, fsdp=False, encode_latents_in_fp16=False, use_xformers=False)
    batch_size = 1
    H = 16
    W = 16
    image = torch.randn(batch_size, 3, H, W, device=device)
    prompt = 'a cool doge'
    tokenized_prompt = model.tokenizer(prompt,
                                       padding='max_length',
                                       max_length=model.tokenizer.model_max_length,
                                       truncation=True,
                                       return_tensors='pt')
    print(tokenized_prompt['attention_mask'].shape)
    batch = {
        'image': image,
        'captions': tokenized_prompt['input_ids'],
        'attention_mask': tokenized_prompt['attention_mask'],
        'cond_original_size': torch.tensor([[H, W]]),
        'cond_crops_coords_top_left': torch.tensor([[0, 0]]),
        'cond_target_size': torch.tensor([[H, W]])
    }
    output, target, _ = model(batch)  # model.forward generates the unet output noise or v_pred target.
    assert output.shape == torch.Size([batch_size, 4, H // 8, W // 8])
    assert target.shape == torch.Size([batch_size, 4, H // 8, W // 8])


@pytest.mark.parametrize('guidance_scale', [0.0, 3.0])
@pytest.mark.parametrize('negative_prompt', [None, 'so cool'])
def test_sdxl_generate(guidance_scale, negative_prompt):
    # fp16 vae does not run on cpu
    model = stable_diffusion_xl(pretrained=False,
                                fsdp=False,
                                encode_latents_in_fp16=False,
                                use_xformers=False,
                                mask_pad_tokens=True)
    output = model.generate(
        prompt='a cool doge',
        negative_prompt=negative_prompt,
        num_inference_steps=1,
        num_images_per_prompt=1,
        height=16,
        width=16,
        guidance_scale=guidance_scale,
        progress_bar=False,
    )
    assert output.shape == (1, 3, 16, 16)
