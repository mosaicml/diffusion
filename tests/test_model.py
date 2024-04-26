# Copyright 2022 MosaicML Diffusion authors
# SPDX-License-Identifier: Apache-2.0

# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from diffusion.models.models import stable_diffusion_2, stable_diffusion_xl


@pytest.mark.parametrize('latent_mean', [0.1, (0.1, 0.2, 0.3, 0.4)])
@pytest.mark.parametrize('latent_std', [5.5, (1.1, 2.2, 3.3, 4.4)])
def test_sd2_latent_scales(latent_mean, latent_std):
    model = stable_diffusion_2(pretrained=False,
                               fsdp=False,
                               encode_latents_in_fp16=False,
                               latent_mean=latent_mean,
                               latent_std=latent_std)

    if isinstance(latent_mean, float):
        latent_mean_comp = [latent_mean] * 4
    else:
        latent_mean_comp = latent_mean
    latent_mean_comp = torch.tensor(latent_mean_comp).view(1, -1, 1, 1).to(model.latent_mean.device)
    if isinstance(latent_std, float):
        latent_std_comp = [latent_std] * 4
    else:
        latent_std_comp = latent_std
    latent_std_comp = torch.tensor(latent_std_comp).view(1, -1, 1, 1).to(model.latent_std.device)

    torch.testing.assert_close(model.latent_mean, latent_mean_comp)
    torch.testing.assert_close(model.latent_std, latent_std_comp)


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


@pytest.mark.parametrize('latent_mean', [0.1, (0.1, 0.2, 0.3, 0.4)])
@pytest.mark.parametrize('latent_std', [5.5, (1.1, 2.2, 3.3, 4.4)])
def test_sdxl_latent_scales(latent_mean, latent_std):
    model = stable_diffusion_xl(pretrained=False,
                                mask_pad_tokens=True,
                                fsdp=False,
                                encode_latents_in_fp16=False,
                                use_xformers=False,
                                latent_mean=latent_mean,
                                latent_std=latent_std)
    if isinstance(latent_mean, float):
        latent_mean_comp = [latent_mean] * 4
    else:
        latent_mean_comp = latent_mean
    latent_mean_comp = torch.tensor(latent_mean_comp).view(1, -1, 1, 1).to(model.latent_mean.device)
    if isinstance(latent_std, float):
        latent_std_comp = [latent_std] * 4
    else:
        latent_std_comp = latent_std
    latent_std_comp = torch.tensor(latent_std_comp).view(1, -1, 1, 1).to(model.latent_std.device)

    torch.testing.assert_close(model.latent_mean, latent_mean_comp)
    torch.testing.assert_close(model.latent_std, latent_std_comp)


@pytest.mark.parametrize('mask_pad_tokens', [True, False])
def test_sdxl_forward(mask_pad_tokens):
    # fp16 vae does not run on cpu
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = stable_diffusion_xl(pretrained=False,
                                mask_pad_tokens=mask_pad_tokens,
                                fsdp=False,
                                encode_latents_in_fp16=False,
                                use_xformers=False)
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
@pytest.mark.parametrize('mask_pad_tokens', [True, False])
def test_sdxl_generate(guidance_scale, negative_prompt, mask_pad_tokens):
    # fp16 vae does not run on cpu
    model = stable_diffusion_xl(pretrained=False,
                                fsdp=False,
                                encode_latents_in_fp16=False,
                                use_xformers=False,
                                mask_pad_tokens=mask_pad_tokens)
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


def test_quasirandomness():
    # fp16 vae does not run on cpu
    model = stable_diffusion_2(pretrained=False, fsdp=False, encode_latents_in_fp16=False, quasirandomness=True)
    # Generate many quasi-random samples
    fake_latents = torch.randn(2048, 4, 8, 8)
    for i in range(10**3):
        timesteps = model._generate_timesteps(fake_latents)
        assert (timesteps >= 0).all()
        assert (timesteps < 1000).all()
