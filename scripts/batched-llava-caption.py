# Copyright 2022 MosaicML Diffusion authors
# SPDX-License-Identifier: Apache-2.0

"""Script to LLaVA caption an image dataset."""

import os
import threading
import time
from argparse import ArgumentParser, Namespace
from typing import Optional

import torch
import wandb
from composer.utils import dist
from huggingface_hub import snapshot_download
from torchvision import transforms
from tqdm.auto import tqdm

try:
    from llava.constants import DEFAULT_IMAGE_TOKEN  # type: ignore
    from llava.constants import DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, IMAGE_TOKEN_INDEX  # type: ignore
    from llava.conversation import conv_templates  # type: ignore
    from llava.mm_utils import get_model_name_from_path, tokenizer_image_token  # type: ignore
    from llava.model.builder import load_pretrained_model  # type: ignore
    from llava.utils import disable_torch_init  # type: ignore
except ImportError as e:
    raise ImportError(
        'LLaVA is not installed. Please install it with `pip install llava@git+https://github.com/haotian-liu/LLaVA.git`'
    ) from e

from PIL import Image, ImageOps
from streaming import Stream
from streaming.base import MDSWriter

from diffusion.datasets.image import StreamingImageDataset


class LLaVACaptioner:
    """LLaVA captioner class."""

    def __init__(self,
                 model_name: str = 'liuhaotian/llava-v1.6-vicuna-13b',
                 max_tokens: int = 1024,
                 compile: bool = False,
                 quantize: bool = False,
                 multi_gpu: bool = False,
                 device: Optional[torch.device] = None):
        self.model_name = model_name
        self.conv_mode = 'llava_v1'
        self.max_tokens = max_tokens
        self.device = torch.device('cuda') if device is None else device

        self.tokenizer, self.model, self.image_processor, self.context_len = self.load_llava(quantize=quantize,
                                                                                             multi_gpu=multi_gpu)
        self.generate = self.model.generate
        if compile:
            self.model = torch.compile(self.model)
            self.generate = torch.compile(self.generate)

        self.input_ids: Optional[torch.Tensor] = None

    def load_llava(self, quantize: bool = False, multi_gpu: bool = False):
        """Loads the llava model."""
        # Download the llava model if it isn't there already.
        snapshot_download(
            repo_id=self.model_name,
            local_dir='/tmp/llava',
            local_dir_use_symlinks=False,
        )
        disable_torch_init()
        model_path = os.path.expanduser('/tmp/llava')
        model_name = get_model_name_from_path(model_path)

        device_map = 'auto' if multi_gpu else self.device
        if quantize:
            return load_pretrained_model(model_path,
                                         None,
                                         model_name,
                                         device_map=device_map,
                                         load_in_4bit=True,
                                         bnb_4bit_use_double_quant=True,
                                         bnb_4bit_compute_dtype=torch.float16)
        else:
            return load_pretrained_model(model_path, None, model_name, device_map=device_map)

    def add_image_tokens(self, prompt: str) -> str:
        if self.model.config.mm_use_im_start_end:
            prompt = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + prompt
        else:
            prompt = DEFAULT_IMAGE_TOKEN + '\n' + prompt
        return prompt

    def tokenize(self, prompt: str) -> torch.Tensor:
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
        return input_ids.unsqueeze(0)

    def format_prompt(self, prompt: str, batch_size: int = 1) -> None:
        # Format the prompt
        prompt = self.add_image_tokens(prompt)
        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        input_ids = self.tokenize(prompt)
        # repeat the prompt along the batch dimension for each image
        input_ids = input_ids.repeat(batch_size, 1)
        input_ids = input_ids.to(self.device)
        self.input_ids = input_ids

    def get_outputs(self, image_batch: torch.Tensor, prompt: str) -> list:
        """Get the output from llava."""
        if self.input_ids is None or self.input_ids.shape[0] != image_batch.shape[0]:
            self.format_prompt(prompt, batch_size=image_batch.shape[0])
        # Prep the image inputs
        image_tensor = self.image_processor.preprocess(image_batch, do_rescale=False,
                                                       return_tensors='pt')['pixel_values'].half().to(self.device)
        # Forward through the model
        with torch.no_grad():
            output_ids = self.generate(self.input_ids,
                                       images=image_tensor,
                                       do_sample=True,
                                       temperature=0.2,
                                       top_p=None,
                                       num_beams=1,
                                       max_new_tokens=self.max_tokens,
                                       use_cache=True)
        # Postprocess outputs
        decoded_output = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        outputs = [o.strip() for o in decoded_output]
        return outputs


class ResizeAndPad:
    """Resize and pad an image to a target size.

    Args:
    - width (int): The target width.
    - height (int): The target height.
    """

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height

    def resize_and_pad(self, image: Image.Image) -> Image.Image:
        """Resize and pad an image to the target size while maintaining aspect ratio.

        Args:
        - image (PIL Image): The image to be resized and padded.

        Returns:
        - PIL Image: The resized and padded image.
        """
        # Calculate the aspect ratio and find the smaller dimension.
        original_width, original_height = image.size
        aspect_ratio = original_width / original_height

        # Resize such that the larger dimension fits the corresponding target dimension.
        if original_width > original_height:  # Width is larger, match width
            resize_width = self.width
            resize_height = round(resize_width / aspect_ratio)
        elif original_width <= original_height:  # Height is larger or equal, match height
            resize_height = self.height
            resize_width = round(resize_height * aspect_ratio)
        else:
            raise ValueError('Invalid image dimensions')
        resized_image = image.resize((resize_width, resize_height), Image.Resampling.LANCZOS)

        # Calculate padding
        pad_width_left = (self.width - resize_width) // 2
        pad_width_right = self.width - resize_width - pad_width_left

        pad_height_top = (self.height - resize_height) // 2
        pad_height_bottom = self.height - resize_height - pad_height_top

        # Apply asymmetric padding if necessary
        padded_image = ImageOps.expand(resized_image,
                                       border=(pad_width_left, pad_height_top, pad_width_right, pad_height_bottom),
                                       fill=0)

        return padded_image

    def __call__(self, image: Image.Image) -> Image.Image:
        return self.resize_and_pad(image)


def parse_args():
    """Parse command-line arguments.

    Returns:
        Namespace: Command-line arguments.
    """
    parser = ArgumentParser()
    parser.add_argument('--remote', type=str, help='Remote to use for the dataset.')
    parser.add_argument('--local', type=str, help='Local directory to use for the dataset.')
    parser.add_argument('--output', help='Output path for the filtered dataset.')
    parser.add_argument('--output_caption_key', type=str, default='llava_caption', help='Dataset output caption key.')
    parser.add_argument('--image_key', type=str, default='image', help='Dataset image key.')
    parser.add_argument('--image_output_key', type=str, default='image_output', help='Dataset image output key.')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for the LLaVA model.')
    parser.add_argument('--height', type=int, default=336, help='Height of the image.')
    parser.add_argument('--width', type=int, default=336, help='Width of the image.')
    parser.add_argument('--model_name',
                        type=str,
                        default='liuhaotian/llava-v1.6-vicuna-13b',
                        help='LLaVA model to use.')
    parser.add_argument('--llava_prompt',
                        type=str,
                        default='Describe this image and its style in a very detailed manner.',
                        help='Prompt to use for LLaVA.')
    parser.add_argument('--max_tokens', type=int, default=1024, help='Maximum tokens to generate.')
    parser.add_argument('--compile', action='store_true', help='Compile the model.')
    parser.add_argument('--quantize', action='store_true', help='Quantize the model.')
    parser.add_argument('--multi_gpu', action='store_true', help='Use multi-gpu.')
    parser.add_argument('--start', type=int, default=0, help='Start index for the dataset.')
    parser.add_argument('--end', type=int, default=None, help='Optional end index for the dataset.')
    # Add wandb arguments
    parser.add_argument('--wandb_disabled', action='store_true')
    parser.add_argument('--wandb_name', type=str, default='llava-captions')
    parser.add_argument('--wandb_project', type=str, default='llava-captions')
    parser.add_argument('--wandb_entity', type=str, default='mosaic-ml')
    return parser.parse_args()


def make_dataset(remote: str,
                 local: str,
                 image_key: str = 'image',
                 image_output_key: str = 'image_output',
                 height: int = 336,
                 width: int = 336):
    """Make a streaming image dataset."""
    streams = []
    for r, l in zip([remote], [local]):
        streams.append(Stream(remote=r, local=l))

    transform = transforms.Compose([ResizeAndPad(width, height), transforms.ToTensor()])
    dataset = StreamingImageDataset(
        streams=streams,
        image_key=image_key,
        image_output_key=image_output_key,
        transform=transform,
        shuffle=False,
        return_all_fields=True,
    )
    return dataset


def prefetch_samples(dataset, start_idx, end_idx):
    """Walk through the dataset to prefetch samples."""
    for i in range(start_idx, end_idx):
        _ = dataset[i]


def main(args: Namespace) -> None:
    """Add LLaVA generated captions to the dataset.

    Args:
        args (Namespace): Command-line arguments.
    """
    if not args.wandb_disabled:
        wandb.init(name=args.wandb_name + f'-rank-{dist.get_global_rank()}',
                   project=args.wandb_project,
                   entity=args.wandb_entity)

    dataset = make_dataset(args.remote,
                           args.local,
                           image_key=args.image_key,
                           image_output_key=args.image_output_key,
                           height=args.height,
                           width=args.width)
    dataset_len = dataset.num_samples
    # Need to grab the column names and types from the first shard in the dataset.
    # Assumes all shards have the same columns and types.
    reader = dataset.shards[0]
    columns = dict(zip(reader.column_names, reader.column_encodings))
    columns[args.output_caption_key] = 'str'
    # Construct the start and end indices for this rank. We want each rank to process a subset of the dataset.
    end = args.end if args.end is not None else dataset_len
    samples_per_rank, remainder = divmod(end - args.start, dist.get_world_size())
    # Need to distribute the remainder across the ranks. Give each rank up to remainder one extra sample.
    start_idx = args.start + dist.get_local_rank() * samples_per_rank + min(remainder, dist.get_local_rank())
    end_idx = start_idx + samples_per_rank
    if dist.get_local_rank() < remainder:
        end_idx += 1
    if not args.wandb_disabled:
        wandb.log({'start_idx': start_idx, 'end_idx': end_idx, 'dataset_len': dataset_len})
    # Start prefetching samples
    prefetch_thread = threading.Thread(target=prefetch_samples, args=(dataset, start_idx, end_idx))
    prefetch_thread.start()

    # Device should be first gpu if available, else cpu
    device = torch.device(f'cuda:{dist.get_local_rank()}' if torch.cuda.is_available() else 'cpu')
    captioner = LLaVACaptioner(model_name=args.model_name,
                               max_tokens=args.max_tokens,
                               compile=args.compile,
                               quantize=args.quantize,
                               multi_gpu=args.multi_gpu,
                               device=device)

    # Each rank needs it's own output
    output_dir = os.path.join(args.output, str(dist.get_global_rank()))
    # Process each subset
    start_time = time.time()
    sample_time = time.time()
    with MDSWriter(out=output_dir, columns=columns) as out:
        for sample_id in tqdm(range(start_idx, end_idx, args.batch_size)):
            batch_end_idx = min(sample_id + args.batch_size, end_idx)
            images = [dataset[i][args.image_output_key] for i in range(sample_id, batch_end_idx)]
            image_batch = torch.stack(images)  # type: ignore
            sample_time = time.time()
            outputs = captioner.get_outputs(image_batch, args.llava_prompt)
            sample_time = time.time() - sample_time
            for output_id, output in enumerate(outputs):
                new_sample = dataset[sample_id + output_id]
                new_sample[args.output_caption_key] = output
                out.write(new_sample)
            if not args.wandb_disabled:
                if sample_id == start_idx:
                    # On the first batch, log sample images and captions for verification.
                    columns = ['id', 'image', 'caption']
                    verification_samples = [[i, wandb.Image(images[i]), outputs[i]] for i in range(len(images))]
                    wandb.log({'sample outputs': wandb.Table(data=verification_samples, columns=columns)})
                completed = sample_id + len(images) - start_idx
                progress = completed / (end_idx - start_idx)
                elapsed_time = time.time() - start_time
                time_per_sample = elapsed_time / completed
                est_time_remaining = (end_idx - start_idx - completed) * time_per_sample
                wandb.log({
                    'samples': completed,
                    'progress': progress,
                    'elapsed time (s)': elapsed_time,
                    'current time per sample (s)': sample_time / len(images),
                    'avg. time per sample (s)': time_per_sample,
                    'est. time remaining (s)': est_time_remaining
                })


if __name__ == '__main__':
    main(parse_args())
