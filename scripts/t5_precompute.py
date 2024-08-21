# Copyright 2022 MosaicML Diffusion authors
# SPDX-License-Identifier: Apache-2.0

"""Script to stream text from a dataset, compute CLIP and T5 latents, and write the latents to streaming dataset."""

import json
import os
import threading
from argparse import ArgumentParser

import torch
from composer.utils import dist
from streaming import MDSWriter, StreamingDataset
from streaming.base.storage import download_file
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, CLIPTextModel


def parse_args():
    """Parse command-line arguments.

    Returns:
        Namespace: Command-line arguments.
    """
    parser = ArgumentParser()
    parser.add_argument('--remote_src_base',
                        type=str,
                        required=True,
                        help='Remote base to download MDS-formatted shards.')
    parser.add_argument('--remote_dst_base', type=str, required=True, help='Remote base to write MDS-formatted shards.')
    parser.add_argument('--subdir_paths',
                        nargs='+',
                        type=str,
                        required=True,
                        help='Path to the subdirectory to process.')
    parser.add_argument('--caption_keys', nargs='+', type=str, required=True, help='Keys to use as captions.')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for processing.')
    parser.add_argument('--start', type=int, default=0, help='Start index for the dataset.')
    parser.add_argument('--end', type=int, default=None, help='Optional end index for the dataset.')
    return parser.parse_args()


def load_models_and_tokenizers(cache_dir, device=None):
    """Load models and tokenizers.

    Args:
        cache_dir (str): Directory with cached weights.
        device (Optional[torch.device]): Device to load models onto.
    """
    device = torch.device('cuda') if device is None else device

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
                                         local_files_only=True).encoder.eval().to(device)
    clip_model = CLIPTextModel.from_pretrained('stabilityai/stable-diffusion-xl-base-1.0',
                                               subfolder='text_encoder',
                                               torch_dtype=torch.float16,
                                               cache_dir=cache_dir,
                                               local_files_only=True).eval().to(device)

    return t5_tokenizer, clip_tokenizer, t5_model, clip_model


def prefetch_samples(dataset, start_idx, end_idx):
    """Walk through the dataset to prefetch samples."""
    for i in range(start_idx, end_idx):
        _ = dataset[i]


def main():
    """Precompute T5-XXL and CLIP captions and write a new dataset."""
    args = parse_args()
    cache_dir = '/tmp/hf_files'
    device = torch.device(f'cuda:{dist.get_local_rank()}' if torch.cuda.is_available() else 'cpu')

    t5_tokenizer, clip_tokenizer, t5_model, clip_model = load_models_and_tokenizers(cache_dir, device)

    columns = None
    for subdir_path in tqdm(args.subdir_paths):
        remote_src = os.path.join(args.remote_src_base, subdir_path)
        remote_dst = os.path.join(args.remote_dst_base, subdir_path)

        # Attempt to download an index.json for the remote source, skip this subdir if it doesn't exist
        try:
            download_file(os.path.join(remote_src, 'index.json'),
                          f'/tmp/index_tries/{subdir_path}/index.json',
                          timeout=300)
        except Exception:
            print(f'Failed to download index.json for {subdir_path}, skipping')
            continue

        # Dataset
        dataset = StreamingDataset(remote=remote_src, local=os.path.join('/tmp', subdir_path), shuffle=False)

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

        # Splitting logic
        dataset_len = dataset.num_samples
        end = args.end if args.end is not None else dataset_len
        samples_per_rank, remainder = divmod(end - args.start, dist.get_world_size())
        start_idx = args.start + dist.get_local_rank() * samples_per_rank + min(remainder, dist.get_local_rank())
        end_idx = start_idx + samples_per_rank
        if dist.get_local_rank() < remainder:
            end_idx += 1

        # Start prefetching samples
        prefetch_thread = threading.Thread(target=prefetch_samples, args=(dataset, start_idx, end_idx))
        prefetch_thread.start()

        # Make writer - each rank needs it's own output
        output_dir = os.path.join(remote_dst, str(dist.get_global_rank()))
        writer = MDSWriter(out=output_dir,
                           columns=columns,
                           compression='zstd',
                           hashes=[],
                           size_limit='1GB',
                           exist_ok=True)

        with torch.no_grad():
            for sample_id in tqdm(range(start_idx, end_idx, args.batch_size)):
                batch_end_idx = min(sample_id + args.batch_size, end_idx)
                samples = [dataset[i] for i in range(sample_id, batch_end_idx)]

                for caption_key in args.caption_keys:
                    caption_batch = [sample[caption_key] for sample in samples]

                    # Pre-compute T5
                    t5_tokenizer_out = t5_tokenizer(caption_batch,
                                                    padding='max_length',
                                                    max_length=t5_tokenizer.model_max_length,
                                                    truncation=True,
                                                    return_tensors='pt')
                    tokenized_captions = t5_tokenizer_out['input_ids'].to(device)
                    attention_masks = t5_tokenizer_out['attention_mask'].to(torch.bool).to(device)
                    t5_out = t5_model(input_ids=tokenized_captions, attention_mask=attention_masks)

                    # Pre-compute CLIP
                    clip_tokenizer_out = clip_tokenizer(caption_batch,
                                                        padding='max_length',
                                                        max_length=clip_tokenizer.model_max_length,
                                                        truncation=True,
                                                        return_tensors='pt')
                    tokenized_captions = clip_tokenizer_out['input_ids'].to(device)
                    attention_masks = clip_tokenizer_out['attention_mask'].to(device)
                    clip_out = clip_model(input_ids=tokenized_captions,
                                          attention_mask=attention_masks,
                                          output_hidden_states=True)

                    # Add caption_key latents to sample
                    for i in range(len(samples)):
                        samples[i][f'{caption_key}_T5_ATTENTION_MASK'] = t5_tokenizer_out['attention_mask'][i].to(
                            torch.bool).numpy().tobytes()
                        samples[i][f'{caption_key}_T5_LATENTS'] = t5_out[0][i].cpu().numpy().tobytes()
                        samples[i][f'{caption_key}_CLIP_ATTENTION_MASK'] = clip_tokenizer_out['attention_mask'][i].to(
                            torch.bool).numpy().tobytes()
                        samples[i][f'{caption_key}_CLIP_LATENTS'] = clip_out.hidden_states[-2][i].cpu().numpy().tobytes(
                        )
                        samples[i][f'{caption_key}_CLIP_POOLED_TEXT'] = clip_out[1][i].cpu().numpy().tobytes()

                for sample in samples:
                    writer.write(sample)
        writer.finish()


if __name__ == '__main__':
    main()
