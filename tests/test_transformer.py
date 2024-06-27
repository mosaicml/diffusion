# Copyright 2022 MosaicML Diffusion authors
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from diffusion.models.models import text_to_image_transformer
from diffusion.models.t2i_transformer import patchify, unpatchify
from diffusion.models.transformer import get_multidimensional_position_embeddings


def test_multidimensional_position_embeddings():
    B = 32
    D = 2
    T = 16
    F = 64
    position_embeddings = torch.randn(D, T, F)
    # Coords should be shape (B, S, D). So for sequence element B, S, one should get D embeddings.
    # These should correspond to the D elements for which T = S in the position embeddings.
    coords = torch.tensor([(i, j) for i in range(3) for j in range(3)])
    coords = coords.unsqueeze(0).expand(B, -1, -1).reshape(B, -1, 2)
    S = coords.shape[1]
    # Get the posistion embeddings from the coords
    sequenced_embeddings = get_multidimensional_position_embeddings(position_embeddings, coords)
    # Test that they are the right shape
    assert sequenced_embeddings.shape == (B, S, F, D)
    # Test that the embeddings are correct
    assert torch.allclose(sequenced_embeddings[0, 0, :, 0], position_embeddings[0, 0, :])
    assert torch.allclose(sequenced_embeddings[1, 2, :, 1], position_embeddings[1, coords[1, 2, 1], :])


@pytest.mark.parametrize('patch_size', [1, 2, 4])
@pytest.mark.parametrize('batch_size', [1, 4])
@pytest.mark.parametrize('C', [3, 4])
@pytest.mark.parametrize('H', [32, 64])
@pytest.mark.parametrize('W', [32, 64])
def test_patch_and_unpatch(patch_size, batch_size, C, H, W):
    # Fake image data
    image = torch.randn(batch_size, C, H, W)
    # Patchify
    image_patches, image_coords = patchify(image, patch_size)
    # Verify patches are the correct size
    assert image_patches.shape == (batch_size, H * W // patch_size**2, C * patch_size**2)
    # Verify coords are the correct size
    assert image_coords.shape == (batch_size, H * W // patch_size**2, 2)
    # Unpatchify
    image_recon = [unpatchify(image_patches[i], image_coords[i], patch_size) for i in range(image_patches.shape[0])]
    # Verify reconstructed image is the correct size
    assert len(image_recon) == batch_size
    assert image_recon[0].shape == (C, H, W)
    # Verify reconstructed image is close to the original
    for i in range(batch_size):
        assert torch.allclose(image_recon[i], image[i], atol=1e-6)


def test_t2i_transformer_forward():
    # fp16 vae does not run on cpu
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = text_to_image_transformer(num_layers=2)
    batch_size = 1
    H = 32
    W = 32
    image = torch.randn(batch_size, 3, H, W, device=device).half()
    caption = torch.randint(low=0, high=128, size=(
        batch_size,
        77,
    ), dtype=torch.long, device=device)
    caption_mask = torch.ones_like(caption, dtype=torch.bool, device=device)
    batch = {'image': image, 'captions': caption, 'attention_mask': caption_mask}
    outputs = model(batch)  # model.forward generates the unet output noise or v_pred target.
    # Desired output shape
    seq_len = H / (8 * 2) * W / (8 * 2)
    output_shape = (1, seq_len, 4 * 2 * 2)
    assert outputs['predictions'].shape == output_shape
    assert outputs['targets'].shape == output_shape


@pytest.mark.parametrize('guidance_scale', [0.0, 3.0])
@pytest.mark.parametrize('negative_prompt', [None, 'so cool'])
def test_t2i_transformer_generate(guidance_scale, negative_prompt):
    model = model = text_to_image_transformer(num_layers=2)
    output = model.generate(
        prompt='a cool doge',
        negative_prompt=negative_prompt,
        num_inference_steps=1,
        num_images_per_prompt=1,
        height=32,
        width=32,
        guidance_scale=guidance_scale,
        progress_bar=False,
    )
    assert output.shape == (1, 3, 32, 32)
