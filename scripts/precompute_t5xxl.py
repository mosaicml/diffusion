# Copyright 2022 MosaicML Diffusion authors
# SPDX-License-Identifier: Apache-2.0

"""Tag LAION with latents."""

import os
import urllib.parse as ul
from bs4 import BeautifulSoup
import re
from argparse import ArgumentParser, Namespace
from typing import List, Optional, Sequence, Union

import torch
import wandb
from composer.devices import DeviceGPU
from composer.utils import dist
from streaming import MDSWriter, Stream, StreamingDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


def clean_caption(caption):
    """Clean caption from bad symbols.

    Copied from: https://github.com/deep-floyd/IF/blob/develop/deepfloyd_if/modules/t5.py

    Copyright (c) 2023 DeepFloyd, StabilityAI

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    1. The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    2. All persons obtaining a copy or substantial portion of the Software,
    a modified version of the Software (or substantial portion thereof), or
    a derivative work based upon this Software (or substantial portion thereof)
    must not delete, remove, disable, diminish, or circumvent any inference filters or
    inference filter mechanisms in the Software, or any portion of the Software that
    implements any such filters or filter mechanisms.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
    """
    caption = str(caption)
    caption = ul.unquote_plus(caption)
    caption = caption.strip().lower()
    caption = re.sub('<person>', 'person', caption)
    # urls:
    caption = re.sub(
        r'\b((?:https?:(?:\/{1,3}|[a-zA-Z0-9%])|[a-zA-Z0-9.\-]+[.](?:com|co|ru|net|org|edu|gov|it)[\w/-]*\b\/?(?!@)))',  # noqa
        '', caption)  # regex for urls
    caption = re.sub(
        r'\b((?:www:(?:\/{1,3}|[a-zA-Z0-9%])|[a-zA-Z0-9.\-]+[.](?:com|co|ru|net|org|edu|gov|it)[\w/-]*\b\/?(?!@)))',  # noqa
        '', caption)  # regex for urls
    # html:
    caption = BeautifulSoup(caption, features='html.parser').text

    # @<nickname>
    caption = re.sub(r'@[\w\d]+\b', '', caption)

    # 31C0—31EF CJK Strokes
    # 31F0—31FF Katakana Phonetic Extensions
    # 3200—32FF Enclosed CJK Letters and Months
    # 3300—33FF CJK Compatibility
    # 3400—4DBF CJK Unified Ideographs Extension A
    # 4DC0—4DFF Yijing Hexagram Symbols
    # 4E00—9FFF CJK Unified Ideographs
    caption = re.sub(r'[\u31c0-\u31ef]+', '', caption)
    caption = re.sub(r'[\u31f0-\u31ff]+', '', caption)
    caption = re.sub(r'[\u3200-\u32ff]+', '', caption)
    caption = re.sub(r'[\u3300-\u33ff]+', '', caption)
    caption = re.sub(r'[\u3400-\u4dbf]+', '', caption)
    caption = re.sub(r'[\u4dc0-\u4dff]+', '', caption)
    caption = re.sub(r'[\u4e00-\u9fff]+', '', caption)
    #######################################################

    # все виды тире / all types of dash --> "-"
    caption = re.sub(
        r'[\u002D\u058A\u05BE\u1400\u1806\u2010-\u2015\u2E17\u2E1A\u2E3A\u2E3B\u2E40\u301C\u3030\u30A0\uFE31\uFE32\uFE58\uFE63\uFF0D]+',  # noqa
        '-', caption)

    # кавычки к одному стандарту
    caption = re.sub(r'[`´«»“”¨]', '"', caption)
    caption = re.sub(r'[‘’]', "'", caption)

    # &quot;
    caption = re.sub(r'&quot;?', '', caption)
    # &amp
    caption = re.sub(r'&amp', '', caption)

    # ip adresses:
    caption = re.sub(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', ' ', caption)

    # article ids:
    caption = re.sub(r'\d:\d\d\s+$', '', caption)

    # \n
    caption = re.sub(r'\\n', ' ', caption)

    # "#123"
    caption = re.sub(r'#\d{1,3}\b', '', caption)
    # "#12345.."
    caption = re.sub(r'#\d{5,}\b', '', caption)
    # "123456.."
    caption = re.sub(r'\b\d{6,}\b', '', caption)
    # filenames:
    caption = re.sub(r'[\S]+\.(?:png|jpg|jpeg|bmp|webp|eps|pdf|apk|mp4)', '', caption)

    #
    caption = re.sub(r'[\"\']{2,}', r'"', caption)  # """AUSVERKAUFT"""
    caption = re.sub(r'[\.]{2,}', r' ', caption)  # """AUSVERKAUFT"""

    caption = re.sub(self.bad_punct_regex, r' ', caption)  # ***AUSVERKAUFT***, #AUSVERKAUFT
    caption = re.sub(r'\s+\.\s+', r' ', caption)  # " . "

    # this-is-my-cute-cat / this_is_my_cute_cat
    regex2 = re.compile(r'(?:\-|\_)')
    if len(re.findall(regex2, caption)) > 3:
        caption = re.sub(regex2, ' ', caption)

    caption = self.basic_clean(caption)

    caption = re.sub(r'\b[a-zA-Z]{1,3}\d{3,15}\b', '', caption)  # jc6640
    caption = re.sub(r'\b[a-zA-Z]+\d+[a-zA-Z]+\b', '', caption)  # jc6640vc
    caption = re.sub(r'\b\d+[a-zA-Z]+\d+\b', '', caption)  # 6640vc231

    caption = re.sub(r'(worldwide\s+)?(free\s+)?shipping', '', caption)
    caption = re.sub(r'(free\s)?download(\sfree)?', '', caption)
    caption = re.sub(r'\bclick\b\s(?:for|on)\s\w+', '', caption)
    caption = re.sub(r'\b(?:png|jpg|jpeg|bmp|webp|eps|pdf|apk|mp4)(\simage[s]?)?', '', caption)
    caption = re.sub(r'\bpage\s+\d+\b', '', caption)

    caption = re.sub(r'\b\d*[a-zA-Z]+\d+[a-zA-Z]+\d+[a-zA-Z\d]*\b', r' ', caption)  # j2d1a2a...

    caption = re.sub(r'\b\d+\.?\d*[xх×]\d+\.?\d*\b', '', caption)

    caption = re.sub(r'\b\s+\:\s+', r': ', caption)
    caption = re.sub(r'(\D[,\./])\b', r'\1 ', caption)
    caption = re.sub(r'\s+', ' ', caption)

    caption.strip()

    caption = re.sub(r'^[\"\']([\w\W]+)[\"\']$', r'\1', caption)
    caption = re.sub(r'^[\'\_,\-\:;]', r'', caption)
    caption = re.sub(r'[\'\_,\-\:\-\+]$', r'', caption)
    caption = re.sub(r'^\.\S+$', '', caption)

    return caption.strip()


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

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        caption = clean_caption(sample['caption'])
        tokenized_caption = self.tokenizer(
            caption,
            max_length=77,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

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
    args.add_argument('--batch-size', type=int, default=128, help='Batch size to use for encoding.')
    # Add wandb arguments
    args.add_argument('--wandb_disabled', action='store_true')
    args.add_argument('--wandb_name', type=str, default='baseline')
    args.add_argument('--wandb_project', type=str, default='precompute-latents-t5xxl')
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
    dist.initialize_dist(device=device, timeout=2700)

    text_encoder = AutoModelForSeq2SeqLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16, cache_dir='/tmp/text-encoder').encoder.eval()
    # Download on local rank 0 first and cache
    # if dist.get_local_rank() == 0:
    #     text_encoder = AutoModelForSeq2SeqLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16, cache_dir='/tmp/text-encoder').eval()
    # dist.barrier()
    # if dist.get_local_rank() > 0:
    #     text_encoder = AutoModelForSeq2SeqLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16, cache_dir='/tmp/text-encoder').eval()
    # dist.barrier()

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
        input_ids = device.batch_to_device(batch['captions']['input_ids'])
        attention_mask = device.batch_to_device(batch['captions']['attention_mask'])
        input_ids = input_ids.reshape(-1, input_ids.shape[-1])
        attention_mask = attention_mask.reshape(-1, attention_mask.shape[-1])

        with torch.no_grad():
            # Encode the text. Assume that the text is already tokenized
            conditioning = text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )['last_hidden_state'].detach()  # Should be (batch_size, 77, 4096)

        # Cast latents to fp32, move to CPU, and convert to numpy / bytes
        conditioning = conditioning.float().cpu().numpy()

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