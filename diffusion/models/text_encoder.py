# Copyright 2022 MosaicML Diffusion authors
# SPDX-License-Identifier: Apache-2.0

"""Text encoders and tokenizers used in diffusion models."""

import logging
import math
import textwrap
from typing import List, Optional, Tuple, Union

import torch
from transformers import AutoTokenizer, AutoModel, CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer, PretrainedConfig

class MultiTextEncoder(torch.nn.Module):
    """Wrapper to handle multiple HuggingFace text encoders.

    Wraps any number of HuggingFace text encoders to behave as one model by sharing inputs and concatenating output.
    
    Args:
        model_names (str, list[str]): Name(s) of the text encoder(s) to load. The name format should be 
            "org_name/repo_name/subfolder" where the subfolder is exclused if it is not used in the repo.
            Default: ``'stabilityai/stable-diffusion-xl-base-1.0/text_encoder'``.
        model_dim_keys (optional, str, list[str]): Key(s) that specify the models' output dimension in the config.
            If ``None``, this is set to ['projection_dim', 'd_model', 'hidden_size']. Default: ``None``.
        encode_latents_in_fp16 (bool): Whether to encode text embeddings in fp16. Default: ``True``.
    """

    def __init__(
            self, 
            model_names: Union[str, List[str]] = 'stabilityai/stable-diffusion-xl-base-1.0/text_encoder',
            model_dim_keys: Optional[Union[str, List[str]]] = None,
            encode_latents_in_fp16: bool = True,
        ):
        super().__init__()
        if isinstance(model_names, str):
            model_names = [model_names]
        if model_dim_keys is None:
            model_dim_keys = ['projection_dim', 'd_model', 'hidden_size'] # CLIP, T5, E5
        torch_dtype = torch.float16 if encode_latents_in_fp16 else None

        self.text_encoders = torch.nn.ModuleList()
        self.text_encoder_dim = 0
        self.architectures = []
        for model_name in model_names:
            # Process model_name string and get model config
            name_split = model_name.split('/')
            base_name = '/'.join(name_split[:2]) 
            subfolder = '/'.join(name_split[2:])
            text_encoder_config = PretrainedConfig.get_config_dict(base_name, subfolder=subfolder)[0]

            # Add text_encoder output dim to total dim
            dim_found = False
            for model_dim_key in model_dim_keys:
                if model_dim_key in text_encoder_config:
                    self.text_encoder_dim += text_encoder_config[model_dim_key]
                    dim_found = True
            if not dim_found:
                raise ValueError(
                    textwrap.dedent(f"""\
                                    Did not find any model_dim_keys ({model_dim_keys}) in the config for {model_name}.\
                                    Please specify the appropriate model_dim_keys for the target model config."""))
            
            architectures = text_encoder_config['architectures']
            if architectures == ['CLIPTextModel']:
                self.text_encoders.append(CLIPTextModel.from_pretrained(base_name, subfolder=subfolder, torch_dtype=torch_dtype))
            elif architectures == ['CLIPTextModelWithProjection']:
                self.text_encoders.append(CLIPTextModelWithProjection.from_pretrained(base_name, subfolder=subfolder, torch_dtype=torch_dtype))
            else:
                self.text_encoders.append(AutoModel.from_pretrained(base_name, subfolder=subfolder, torch_dtype=torch_dtype))
            self.architectures += architectures


    @property
    def device(self):
        return self.text_encoders[0].device

    def forward(self, tokenized_texts):
        all_text_embed = []
        all_pooled_text = []
        for i in range(len(self.text_encoders)):
            output_hidden_states = self.architectures[i] in ['CLIPTextModel', 'CLIPTextModelWithProjection']
            out = self.text_encoder[i](tokenized_texts[i], output_hidden_states=output_hidden_states)
            text_embed = out.hidden_states[-2] if output_hidden_states else out[0]
            pooled_text = out[0] if self.architectures[i] == 'CLIPTextModelWithProjection' else None
            
            all_text_embed.append(text_embed)
            if pooled_text is not None:
                all_pooled_text.append(pooled_text)

        text_embed = torch.concat(all_text_embed, dim=-1)
        if all_pooled_text:
            pooled_text = torch.concat(all_pooled_text, dim=-1)
            return text_embed, pooled_text
        return (text_embed,)
