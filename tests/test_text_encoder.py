# Copyright 2022 MosaicML Diffusion authors
# SPDX-License-Identifier: Apache-2.0

import itertools

import pytest
import torch

from diffusion.models.text_encoder import MultiTextEncoder, MultiTokenizer

# Get all permutations of tokenizers to test
tokenizer_to_max_len = {'stabilityai/stable-diffusion-xl-base-1.0/tokenizer': 77, 'stabilityai/stable-diffusion-xl-base-1.0/tokenizer_2': 77, 'intfloat/e5-large-v2': 512}
tokenizers = ['stabilityai/stable-diffusion-xl-base-1.0/tokenizer', 'stabilityai/stable-diffusion-xl-base-1.0/tokenizer_2', 'intfloat/e5-large-v2']
tokenizer_combinations = tokenizers.copy()
for i in range(2, len(tokenizers)+1):
    tokenizer_combinations += list(itertools.combinations(tokenizers, i))
@pytest.mark.parametrize('tokenizer_names', tokenizer_combinations)
@pytest.mark.parametrize('prompt', ['What is a test? Who are we testing? Why are we testing? WHERE are we testing?!', ['What is a test? Who are we testing?', 'Why are we testing? WHERE are we testing?!']])
def test_multi_tokenizer(prompt, tokenizer_names):
    tokenizer = MultiTokenizer(tokenizer_names)
    tokenized_text =  tokenizer(prompt,
                                padding='max_length',
                                max_length=tokenizer.model_max_length,
                                truncation=True,
                                return_tensors='pt')

    n_prompts = 1 if isinstance(prompt, str) else len(prompt)
    if isinstance(tokenizer_names, str):
        tokenizer_names = (tokenizer_names,)
    n_tokenizers = len(tokenizer_names)
    sequence_length = max([tokenizer_to_max_len[name] for name in tokenizer_names])
    assert tokenized_text['input_ids'].shape == torch.Size([n_prompts, n_tokenizers, sequence_length])
    assert tokenized_text['attention_mask'].shape == torch.Size([n_prompts, n_tokenizers, sequence_length])


def test_multi_text_encoder(text_encoder, names):
    pass