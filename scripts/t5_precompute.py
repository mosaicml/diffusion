# Copyright 2022 MosaicML Diffusion authors
# SPDX-License-Identifier: Apache-2.0

"""Script to stream text from a dataset, compute CLIP and T5 latents, and write the latents to streaming dataset."""

import json
import os
from argparse import ArgumentParser

import torch
from streaming import MDSWriter, StreamingDataset
from streaming.base.storage import download_file
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, CLIPTextModel

# TODO: Implement batching? 10% faster (when using t5-only), but a lot more complicated code

arg_parser = ArgumentParser()
arg_parser.add_argument('--remote_src_base',
                        type=str,
                        required=True,
                        help='Remote base to download MDS-formatted shards.')
arg_parser.add_argument('--remote_dst_base', type=str, required=True, help='Remote base to write MDS-formatted shards.')
arg_parser.add_argument('--subdir_paths',
                        nargs='+',
                        type=str,
                        required=True,
                        help='Path to the subdirectory to process.')
arg_parser.add_argument('--caption_keys', nargs='+', type=str, required=True, help='Keys to use as captions.')
args = arg_parser.parse_args()

cache_dir = '/tmp/hf_files'

# Instantiate tokenizers
print('Building tokenizers')
t5_tokenizer = AutoTokenizer.from_pretrained('google/t5-v1_1-xxl', cache_dir=cache_dir, local_files_only=True)
clip_tokenizer = AutoTokenizer.from_pretrained('stabilityai/stable-diffusion-xl-base-1.0',
                                               subfolder='tokenizer',
                                               cache_dir=cache_dir,
                                               local_files_only=True)

print('Building models')
t5_model = AutoModel.from_pretrained('google/t5-v1_1-xxl',
                                     torch_dtype=torch.float16,
                                     cache_dir=cache_dir,
                                     local_files_only=True).encoder.cuda().eval()
clip_model = CLIPTextModel.from_pretrained('stabilityai/stable-diffusion-xl-base-1.0',
                                           subfolder='text_encoder',
                                           torch_dtype=torch.float16,
                                           cache_dir=cache_dir,
                                           local_files_only=True).cuda().eval()

columns = None
for subdir_path in tqdm(args.subdir_paths):
    remote_src = os.path.join(args.remote_src_base, subdir_path)
    remote_dst = os.path.join(args.remote_dst_base, subdir_path)
    # Attempt to download an index.json for the remote source, skip this subdir if it doesn't exist
    try:
        download_file(os.path.join(remote_src, 'index.json'), f'/tmp/index_tries/{subdir_path}/index.json', timeout=60)
    except Exception:
        print(f'Failed to download index.json for {subdir_path}, skipping')
        continue
    # Dataset
    dataset = StreamingDataset(remote=remote_src,
                               local=os.path.join('/tmp', subdir_path),
                               download_timeout=300,
                               shuffle=False)

    # Get columns
    if columns is None:
        with open(os.path.join('/tmp/', subdir_path, 'index.json')) as f:
            index_json = json.load(f)
        columns = dict(zip(index_json['shards'][0]['column_names'], index_json['shards'][0]['column_encodings']))
        for caption_key in args.caption_keys:
            columns[f'{caption_key}_T5_ATTENTION_MASK'] = 'bytes'
            columns[f'{caption_key}_T5_LATENTS'] = 'bytes'
            columns[f'{caption_key}_CLIP_ATTENTION_MASK'] = 'bytes'
            columns[f'{caption_key}_CLIP_LATENTS'] = 'bytes'
            columns[f'{caption_key}_CLIP_POOLED_TEXT'] = 'bytes'
        print(columns)

    # Make writer
    writer = MDSWriter(out=remote_dst, columns=columns, compression='zstd', hashes=[], size_limit='1GB')

    with torch.no_grad():
        for i in range(len(dataset)):
            sample = dataset[i]

            for caption_key in args.caption_keys:
                captions = sample[caption_key]
                # Pre-compute T5
                t5_tokenizer_out = t5_tokenizer(captions,
                                                padding='max_length',
                                                max_length=t5_tokenizer.model_max_length,
                                                truncation=True,
                                                return_tensors='pt')
                tokenized_captions = t5_tokenizer_out['input_ids'].cuda()
                attention_masks = t5_tokenizer_out['attention_mask'].to(torch.bool).cuda()
                sample[f'{caption_key}_T5_ATTENTION_MASK'] = t5_tokenizer_out['attention_mask'].squeeze(0).to(
                    torch.bool).numpy().tobytes()
                t5_out = t5_model(input_ids=tokenized_captions, attention_mask=attention_masks)
                sample[f'{caption_key}_T5_LATENTS'] = t5_out[0].squeeze(0).cpu().numpy().tobytes()
                # Pre-compute CLIP
                clip_tokenizer_out = clip_tokenizer(captions,
                                                    padding='max_length',
                                                    max_length=clip_tokenizer.model_max_length,
                                                    truncation=True,
                                                    return_tensors='pt')
                tokenized_captions = clip_tokenizer_out['input_ids'].cuda()
                attention_masks = clip_tokenizer_out['attention_mask'].cuda()
                sample[f'{caption_key}_CLIP_ATTENTION_MASK'] = clip_tokenizer_out['attention_mask'].squeeze(0).to(
                    torch.bool).numpy().tobytes()
                clip_out = clip_model(input_ids=tokenized_captions,
                                      attention_mask=attention_masks,
                                      output_hidden_states=True)
                sample[f'{caption_key}_CLIP_LATENTS'] = clip_out.hidden_states[-2].squeeze(0).cpu().numpy().tobytes()
                sample[f'{caption_key}_CLIP_POOLED_TEXT'] = clip_out[1].squeeze(0).cpu().numpy().tobytes()

            writer.write(sample)
    writer.finish()
