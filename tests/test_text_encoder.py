# Copyright 2022 MosaicML Diffusion authors
# SPDX-License-Identifier: Apache-2.0

import itertools

import pytest
import torch

from diffusion.models.text_encoder import MultiTextEncoder, MultiTokenizer

# Get all permutations of tokenizers to test
tokenizer_to_max_len = {
    'stabilityai/stable-diffusion-xl-base-1.0/tokenizer': 77,
    'stabilityai/stable-diffusion-xl-base-1.0/tokenizer_2': 77,
    'intfloat/e5-large-v2': 512
}
tokenizers = [
    'stabilityai/stable-diffusion-xl-base-1.0/tokenizer', 'stabilityai/stable-diffusion-xl-base-1.0/tokenizer_2',
    'intfloat/e5-large-v2'
]
tokenizer_combins = tokenizers.copy()
for i in range(2, len(tokenizers) + 1):
    tokenizer_combins += list(itertools.combinations(tokenizers, i))


@pytest.mark.parametrize('tokenizer_names', tokenizer_combins)
@pytest.mark.parametrize('prompt', [
    'What is a test? Who are we testing? Why are we testing? WHERE are we testing?!',
    ['What is a test? Who are we testing?', 'Why are we testing? WHERE are we testing?!']
])
def test_multi_tokenizer(prompt, tokenizer_names):
    tokenizer = MultiTokenizer(tokenizer_names)
    tokenized_text = tokenizer(prompt,
                               padding='max_length',
                               max_length=tokenizer.model_max_length,
                               truncation=True,
                               return_tensors='pt')

    # Check tokenized_text shape
    n_prompts = 1 if isinstance(prompt, str) else len(prompt)
    if isinstance(tokenizer_names, str):
        tokenizer_names = (tokenizer_names,)
    n_tokenizers = len(tokenizer_names)
    sequence_length = min([tokenizer_to_max_len[name] for name in tokenizer_names])
    assert tokenized_text['input_ids'].shape == torch.Size([n_prompts, n_tokenizers, sequence_length])
    assert tokenized_text['attention_mask'].shape == torch.Size([n_prompts, n_tokenizers, sequence_length])


text_encoder_dims = {
    'stabilityai/stable-diffusion-xl-base-1.0/text_encoder': 768,
    'stabilityai/stable-diffusion-xl-base-1.0/text_encoder_2': 1280,
    'intfloat/e5-large-v2': 1024
}
text_encoders = [
    'stabilityai/stable-diffusion-xl-base-1.0/text_encoder', 'stabilityai/stable-diffusion-xl-base-1.0/text_encoder_2',
    'intfloat/e5-large-v2'
]
text_encoder_combins = text_encoders.copy()
for i in range(2, len(text_encoders) + 1):
    text_encoder_combins += list(itertools.combinations(text_encoders, i))


@pytest.mark.parametrize('tokenizer_names,text_encoder_names', list(zip(tokenizer_combins, text_encoder_combins)))
def test_multi_text_encoder(tokenizer_names, text_encoder_names, encode_latents_in_fp16=False):
    prompt = 'What is a test? Who are we testing? Why are we testing? WHERE are we testing?!'
    tokenizer = MultiTokenizer(tokenizer_names)
    tokenized_text = tokenizer(prompt,
                               padding='max_length',
                               max_length=tokenizer.model_max_length,
                               truncation=True,
                               return_tensors='pt')
    text_encoder = MultiTextEncoder(text_encoder_names, encode_latents_in_fp16=encode_latents_in_fp16)
    if encode_latents_in_fp16:
        text_encoder = text_encoder.half()
    out = text_encoder(tokenized_text['input_ids'], tokenized_text['attention_mask'])

    n_prompts = 1 if isinstance(prompt, str) else len(prompt)
    if isinstance(tokenizer_names, str):
        tokenizer_names = (tokenizer_names,)
    if isinstance(text_encoder_names, str):
        text_encoder_names = (text_encoder_names,)
    out_dim = sum([text_encoder_dims[name] for name in text_encoder_names])
    sequence_length = min([tokenizer_to_max_len[name] for name in tokenizer_names])
    assert out[0].shape == torch.Size([n_prompts, sequence_length, out_dim])
    assert out_dim == text_encoder.text_encoder_dim
