# Copyright 2022 MosaicML Diffusion authors
# SPDX-License-Identifier: Apache-2.0

"""Tag LAION with latents."""

import os
from argparse import ArgumentParser, Namespace
from typing import List, Optional, Sequence, Union

import torch
import wandb
from composer.devices import DeviceGPU
from composer.utils import dist
from streaming import MDSWriter, Stream, StreamingDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, T5Tokenizer



class StreamingLAIONDataset(StreamingDataset):
    """Implementation of the LAION dataset as a streaming dataset. Only returns sample and caption.

    Args:
        streams (Sequence[Stream], optional): One or more Streams to stream/cache samples from. StreamingLAIONDataset
            uses either ``streams`` or ``remote``/``local``. Default:``None``.
        remote (str, optional): Remote directory (S3 or local filesystem) where dataset is stored. Default: ``None``.
        local (str, optional): Local filesystem directory where dataset is cached during operation. Default: ``None``.
        split (str, optional): The dataset split to use. Currently, only ``None`` is supported. Default: ``None``.
        shuffle (bool): Whether to shuffle the samples in this dataset. Default: ``False``.
        tokenizer_name_or_path (str): The name or path of the tokenizer to use. Default: ``'t5-v1_1-xxl'``.
        predownload (Optional[int]): The number of samples to prefetch. Default: ``100_000``.
        download_retry (Optional[int]): The number of times to retry a download. Default: ``2``.
        download_timeout (Optional[float]): The timeout for a download. Default: ``120``.
        batch_size (Optional[int]): Hint batch_size that will be used on each device's DataLoader. Default: ``None``.
    """

    def __init__(
        self,
        streams: Optional[Sequence[Stream]] = None,
        remote: Optional[str] = None,
        local: Optional[str] = None,
        split: Optional[str] = None,
        shuffle: Optional[bool] = False,
        tokenizer_name_or_path: Optional[str] = 't5-v1_1-xxl',
        predownload: Optional[int] = 100_000,
        download_retry: Optional[int] = 2,
        download_timeout: Optional[float] = 120,
        batch_size: Optional[int] = None,
    ) -> None:

        super().__init__(
            streams=streams,
            remote=remote,
            local=local,
            split=split,
            shuffle=shuffle,
            predownload=predownload,
            keep_zip=False,
            download_retry=download_retry,
            download_timeout=download_timeout,
            validate_hash=None,
            batch_size=batch_size,
        )

        self.tokenizer = T5Tokenizer.from_pretrained(tokenizer_name_or_path)

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        caption = sample['caption']
        tokenized_caption = self.tokenizer(
            caption,
            padding='max_length',
            max_length=self.tokenizer.model_max_length,
            truncation=True,
        )['input_ids']
        tokenized_caption = torch.tensor(tokenized_caption)

        return {'captions': tokenized_caption, 'sample': sample}


def build_streaming_laion_dataloader(
    remote: Union[str, List],
    local: Union[str, List],
    batch_size: int,
    tokenizer_name_or_path: str = 't5-v1_1-xxl',
    num_samples: Optional[int] = None,
    predownload: Optional[int] = 100_000,
    download_retry: Optional[int] = 2,
    download_timeout: Optional[float] = 120,
    drop_last: bool = True,
    shuffle: bool = True,
    **dataloader_kwargs,
):
    """Builds a streaming LAION dataloader returning just captions.

    Args:
        remote (str, Sequence[str]): One or more remote directories (S3 or local filesystem) where dataset is stored.
        local (str, Sequence[str]): One or more local filesystem directories where dataset is cached during operation.
        batch_size (int): The batch size to use.
        tokenizer_name_or_path (str): The name or path of the tokenizer to use. Default: ``'t5-v1_1-xxl'``.
        num_samples (Optional[int]): The number of samples to use. Default: ``None`` uses all available samples.
        predownload (Optional[int]): The number of samples to prefetch. Default: ``100_000``.
        download_retry (Optional[int]): The number of times to retry a download. Default: ``2``.
        download_timeout (Optional[float]): The timeout for a download. Default: ``120``.
        drop_last (bool): Whether to drop the last batch if it is incomplete. Default: ``True``.
        shuffle (bool): Whether to shuffle the samples in this dataset. Default: ``True``.
        **dataloader_kwargs: Additional arguments to pass to the dataloader.
    """
    if isinstance(remote, Sequence) or isinstance(local, Sequence):
        assert isinstance(remote, Sequence) and isinstance(
            local, Sequence), 'If either remote or local is a sequence, both must be sequences'
        assert len(remote) == len(
            local), f'remote and local must be lists of the same length, got lengths {len(remote)} and {len(local)}'
    else:
        # Hacky... make remote and local lists to simplify downstream code
        remote, local = [remote], [local]

    # Create a Stream for each (remote, local) pair
    streams = []
    for r, l in zip(remote, local):
        streams.append(Stream(remote=r, local=l, download_retry=download_retry, download_timeout=download_timeout))

    dataset = StreamingLAIONDataset(
        streams=streams,
        split=None,
        shuffle=shuffle,
        tokenizer_name_or_path=tokenizer_name_or_path,
        predownload=predownload,
        download_retry=download_retry,
        download_timeout=download_timeout,
        batch_size=batch_size,
    )
    # Create a subset of the dataset
    if num_samples is not None:
        dataset = torch.utils.data.Subset(dataset, range(num_samples))  # type: ignore

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        sampler=None,
        drop_last=drop_last,
        **dataloader_kwargs,
    )

    return dataloader


def parse_args() -> Namespace:
    """Parse command-line arguments.

    Returns:
        Namespace: Command-line arguments.
    """
    args = ArgumentParser()
    args.add_argument('--local', type=str, required=True, help='Local directory to store shards.')
    args.add_argument('--remote_download',
                      type=str,
                      default='',
                      help='Remote path to download MDS-formatted shards to.')
    args.add_argument('--remote_upload', type=str, default='', help='Remote path to upload MDS-formatted shards to.')
    args.add_argument('--bucket', type=int, help='Bucket index under remote path.')
    args.add_argument('--model_name',
                      type=str,
                      default='google/t5-v1_1-xxl',
                      help='Name of model to use for encoding.')
    args.add_argument('--batch-size', type=int, default=8, help='Batch size to use for encoding.')
    # Add wandb arguments
    args.add_argument('--wandb_disabled', action='store_true')
    args.add_argument('--wandb_name', type=str, default='baseline')
    args.add_argument('--wandb_project', type=str, default='laion-latents')
    args.add_argument('--wandb_entity', type=str, default='mosaic-ml')
    return args.parse_args()


def main(args: Namespace) -> None:
    """Add latents to LAION dataset.

    Args:
        args (Namespace): Command-line arguments.
    """
    if not args.wandb_disabled and dist.get_local_rank() == 0:
        wandb.init(name=args.wandb_name, project=args.wandb_project, entity=args.wandb_entity)

    dataloader = build_streaming_laion_dataloader(
        remote=[os.path.join(args.remote_download, str(args.bucket))],
        local=[os.path.join(args.local, str(args.bucket))],
        batch_size=args.batch_size,
        tokenizer_name_or_path=args.model_name,
        predownload=20_000,
        drop_last=False,
        shuffle=False,
        prefetch_factor=2,
        num_workers=8,
        persistent_workers=True,
        pin_memory=True,
        download_timeout=300,
    )

    device = DeviceGPU()
    text_encoder = AutoModelForSeq2SeqLM.from_pretrained(args.model_name, subfolder='text_encoder', torch_dtype=torch.bfloat16).eval()
    text_encoder = device.module_to_device(text_encoder)

    columns = {
        'punsafe': 'float64',
        'pwatermark': 'float64',
        'similarity': 'float64',
        'caption': 'str',
        'url': 'str',
        'key': 'str',
        'status': 'str',
        'error_message': 'str',
        'width': 'int32',
        'height': 'int32',
        'original_width': 'int32',
        'original_height': 'int32',
        'exif': 'str',
        'jpg': 'bytes',
        'hash': 'int64',
        'aesthetic_score': 'float64',
        'caption_t5xxl_latents': 'bytes',
    }

    # We split each bucket into 8 copies for each GPU per node
    remote_upload = os.path.join(args.remote_upload, str((args.bucket - 1) * 8 + dist.get_local_rank()))
    writer = MDSWriter(
        out=remote_upload,
        columns=columns,
        compression=None,
        hash=[],
        size_limit=256 * (2**20),
        max_workers=64,
    )

    max_sample_idx = 0
    for batch_idx, batch in enumerate(tqdm(dataloader)):
        captions = device.batch_to_device(batch['captions'])

        with torch.no_grad():
            # Encode the text. Assume that the text is already tokenized
            conditioning = text_encoder(captions.view(-1, captions.shape[-1]))[0]  # Should be (batch_size, 77, 768)

        # Move the latents to CPU and convert to numpy / bytes
        conditioning = conditioning.cpu().numpy()

        sample = batch['sample']
        for i in range(conditioning.shape[0]):
            mds_sample = {
                'punsafe': sample['punsafe'][i],
                'pwatermark': sample['pwatermark'][i],
                'similarity': sample['similarity'][i],
                'caption': sample['caption'][i],
                'url': sample['url'][i],
                'key': sample['key'][i],
                'status': sample['status'][i],
                'error_message': sample['error_message'][i],
                'width': sample['width'][i],
                'height': sample['height'][i],
                'original_width': sample['original_width'][i],
                'original_height': sample['original_height'][i],
                'exif': sample['exif'][i],
                'jpg': sample['jpg'][i],
                'hash': sample['hash'][i],
                'aesthetic_score': sample['aesthetic_score'][i],
                'caption_t5xxl_latents': conditioning[i].tobytes(),
            }
            writer.write(mds_sample)
        if not args.wandb_disabled and dist.get_local_rank() == 0:
            wandb.log({'batch': batch_idx, 'progress': batch_idx / len(dataloader)})

        dist.barrier()
        max_sample_idx += args.batch_size * dist.get_world_size()
        # Remove completed shards
        if batch_idx % 10 == 0 and dist.get_local_rank() == 0:
            shard_sample_offset = 0
            for shard_id, samples_this_shard in enumerate(dataloader.dataset.samples_per_shard):  # type: ignore
                shard_sample_offset += samples_this_shard
                if max_sample_idx < shard_sample_offset:
                    break
                stream_id = dataloader.dataset.stream_per_shard[shard_id]  # type: ignore
                stream = dataloader.dataset.streams[stream_id]  # type: ignore
                for raw_info, zip_info in dataloader.dataset.shards[shard_id].file_pairs:  # type: ignore
                    if raw_info:
                        path = os.path.join(stream.local, raw_info.basename)
                        if os.path.exists(path):
                            os.remove(path)
                    if zip_info:
                        path = os.path.join(stream.local, zip_info.basename)
                        if os.path.exists(path):
                            os.remove(path)

    writer.finish()


if __name__ == '__main__':
    main(parse_args())
