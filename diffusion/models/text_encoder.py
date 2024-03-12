# Copyright 2022 MosaicML Diffusion authors
# SPDX-License-Identifier: Apache-2.0

"""Text encoders and tokenizers used in diffusion models."""

import textwrap
from typing import List, Optional, Tuple, Union

import torch
from transformers import AutoModel, AutoTokenizer, CLIPTextModel, CLIPTextModelWithProjection, PretrainedConfig


class MultiTextEncoder(torch.nn.Module):
    """Wrapper to handle multiple HuggingFace text encoders.

    Wraps any number of HuggingFace text encoders to behave as one model by sharing inputs and concatenating output.

    Args:
        model_names (str, Tuple[str, ...]): Name(s) of the text encoder(s) to load. The name format should be
            "org_name/repo_name/subfolder" where the subfolder is excluded if it is not used in the repo.
        model_dim_keys (optional, str, list[str]): Key(s) that specify the models' output dimension in the config.
            If ``None``, this is set to ['projection_dim', 'd_model', 'hidden_size']. Default: ``None``.
        encode_latents_in_fp16 (bool): Whether to encode text embeddings in fp16. Default: ``True``.
        pretrained_sdxl (bool): Whether this text encoder is for a pretrained SDXL. If true, this will only use
            the projected output from a CLIPTextModelWithProjection. Default: ``False``.
    """

    def __init__(
        self,
        model_names: Union[str, Tuple[str, ...]],
        model_dim_keys: Optional[Union[str, List[str]]] = None,
        encode_latents_in_fp16: bool = True,
        pretrained_sdxl: bool = False,
    ):
        super().__init__()
        self.pretrained_sdxl = pretrained_sdxl

        if isinstance(model_names, str):
            model_names = (model_names,)
        if model_dim_keys is None:
            model_dim_keys = ['d_model', 'hidden_size']  # T5, CLIP, E5
        torch_dtype = torch.float16 if encode_latents_in_fp16 else None

        self.text_encoders = torch.nn.ModuleList()
        self.text_encoder_dim = 0
        self.text_encoder_proj_dim = 0
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
                    # This does not add to proj_dim when pretrained and architecture is CLIPTextModel
                    if not self.pretrained_sdxl or text_encoder_config['architectures'] != ['CLIPTextModel']:
                        self.text_encoder_proj_dim += text_encoder_config[model_dim_key]
                    dim_found = True
            if not dim_found:
                raise ValueError(
                    textwrap.dedent(f"""\
                                    Did not find any model_dim_keys ({model_dim_keys}) in the config for {model_name}.\
                                    Please specify the appropriate model_dim_keys for the target model config."""))

            architectures = text_encoder_config['architectures']
            if architectures == ['CLIPTextModel']:
                self.text_encoders.append(
                    CLIPTextModel.from_pretrained(base_name, subfolder=subfolder, torch_dtype=torch_dtype))
            elif architectures == ['CLIPTextModelWithProjection']:
                self.text_encoders.append(
                    CLIPTextModelWithProjection.from_pretrained(base_name, subfolder=subfolder,
                                                                torch_dtype=torch_dtype))
            else:
                self.text_encoders.append(
                    AutoModel.from_pretrained(base_name, subfolder=subfolder, torch_dtype=torch_dtype))
            self.architectures += architectures

    @property
    def device(self):
        return self.text_encoders[0].device

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        # Check input_ids and attention_mask is shape [batch_size, len(self.text_encoders), max_sequence_length]
        if len(input_ids.shape) == 2:
            input_ids = input_ids.unsqueeze(dim=1)
        if attention_mask is not None and len(attention_mask.shape) == 2:
            attention_mask = attention_mask.unsqueeze(dim=1)
        if input_ids.shape[1] != len(self.text_encoders) or (attention_mask is not None and
                                                             attention_mask.shape[1] != len(self.text_encoders)):
            raise ValueError(
                'input_ids and attention_mask must be of shape [batch_size, len(self.tokenizers), max_seq_len]')

        all_text_embed = []
        all_pooled_text = []
        for i in range(len(self.text_encoders)):
            output_hidden_states = self.architectures[i] in ['CLIPTextModel', 'CLIPTextModelWithProjection']
            out = self.text_encoders[i](input_ids=input_ids[:, i],
                                        attention_mask=attention_mask[:, i] if attention_mask is not None else None,
                                        output_hidden_states=output_hidden_states)
            text_embed = out.hidden_states[-2] if output_hidden_states else out[0]
            all_text_embed.append(text_embed)

            if self.architectures[i] == 'CLIPTextModelWithProjection':
                pooled_text = out[0]
                all_pooled_text.append(pooled_text)
            elif not self.pretrained_sdxl:
                pooled_text = out[1]
                all_pooled_text.append(pooled_text)

        text_embed = torch.concat(all_text_embed, dim=-1)
        pooled_text = torch.concat(all_pooled_text, dim=-1)
        return text_embed, pooled_text


class MultiTokenizer:
    """Wrapper to handle multiple HuggingFace tokenizers.

    Args:
        tokenizer_names_or_paths (str, Tuple[str, ...]): Name(s) of the tokenizer(s) to load. The name format should be
            "org_name/repo_name/subfolder" where the subfolder is excluded if it is not used in the repo.
    """

    def __init__(self, tokenizer_names_or_paths: Union[str, Tuple[str, ...]]):
        if isinstance(tokenizer_names_or_paths, str):
            tokenizer_names_or_paths = (tokenizer_names_or_paths,)

        self.tokenizers = []
        for tokenizer_name_or_path in tokenizer_names_or_paths:
            path_split = tokenizer_name_or_path.split('/')
            base_name = '/'.join(path_split[:2])
            subfolder = '/'.join(path_split[2:])
            self.tokenizers.append(AutoTokenizer.from_pretrained(base_name, subfolder=subfolder))

        self.model_max_length = min([t.model_max_length for t in self.tokenizers])

    def __call__(self, text, padding, max_length, truncation, return_tensors):
        """Function to tokenize text.

        Returns:
            {'input_ids': PyTorch Tensor for tokenized text of shape [n_text, len(self.tokenizers), max_length],
            'attention_mask': PyTorch Tensor containing 0s and 1s of shape [n_text, len(self.tokenizers), max_length]}
        """
        input_ids = []
        attention_masks = []
        for tokenizer in self.tokenizers:
            out = tokenizer(text,
                            padding=padding,
                            max_length=max_length,
                            truncation=truncation,
                            return_tensors=return_tensors)

            input_ids.append(out.input_ids)
            attention_masks.append(out.attention_mask)

        input_ids = torch.stack(input_ids, dim=1)
        attention_mask = torch.stack(attention_masks, dim=1)
        return {'input_ids': input_ids, 'attention_mask': attention_mask}

    def batch_decode(self, sequences, skip_special_tokens: bool = False):
        sequences = sequences[:, 0] if len(sequences.shape) == 3 else sequences
        text = self.tokenizers[0].batch_decode(sequences, skip_special_tokens=skip_special_tokens)
        return text
