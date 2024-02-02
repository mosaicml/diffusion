# Copyright 2022 MosaicML Diffusion authors
# SPDX-License-Identifier: Apache-2.0

"""Script to LLaVA caption an image dataset."""

import argparse
import os
import time

import torch
from huggingface_hub import snapshot_download
from torchvision import transforms

try:
    from llava.constants import DEFAULT_IMAGE_TOKEN  # type: ignore
    from llava.constants import DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, IMAGE_TOKEN_INDEX  # type: ignore
    from llava.conversation import SeparatorStyle, conv_templates  # type: ignore
    from llava.mm_utils import get_model_name_from_path, tokenizer_image_token  # type: ignore
    from llava.model.builder import load_pretrained_model  # type: ignore
    from llava.utils import disable_torch_init  # type: ignore
except ImportError:
    raise ImportError(
        'LLaVA is not installed. Please install it with `pip install llava@git+https://github.com/haotian-liu/LLaVA.git`'
    )

from PIL import Image, ImageOps
from streaming import Stream
from streaming.base import MDSWriter

from diffusion.datasets.image import StreamingImageDataset

parser = argparse.ArgumentParser()
parser.add_argument('--remote', type=str, help='Remote to use for the dataset.')
parser.add_argument('--local', type=str, help='Local directory to use for the dataset.')
parser.add_argument('--output', help='Output path for the filtered dataset.')
parser.add_argument('--output_caption_key', type=str, default='llava_caption', help='Dataset output caption key.')
parser.add_argument('--image_key', type=str, default='image', help='Dataset image key.')
parser.add_argument('--image_output_key', type=str, default='image_output', help='Dataset image output key.')
parser.add_argument('--batch_size', type=int, default=1, help='Batch size for the LLaVA model.')
parser.add_argument('--height', type=int, default=512, help='Height of the image.')
parser.add_argument('--width', type=int, default=512, help='Width of the image.')
parser.add_argument('--model_name', type=str, default='liuhaotian/llava-v1.6-vicuna-13b', help='LLaVA model to use.')
parser.add_argument('--llava_prompt',
                    type=str,
                    default='Describe this image and its style in a very detailed manner.',
                    help='Prompt to use for LLaVA.')
parser.add_argument('--max_tokens', type=int, default=1024, help='Maximum tokens to generate.')
parser.add_argument('--compile', action='store_true', help='Compile the model.')
args = parser.parse_args()


def make_dataset(remote: str, local: str, image_key: str = 'image', image_output_key: str = 'image_output'):
    """Make a streaming image dataset."""
    streams = []
    for r, l in zip([remote], [local]):
        streams.append(Stream(remote=r, local=l))

    transform = transforms.Compose([])
    dataset = StreamingImageDataset(
        streams=streams,
        image_key=image_key,
        image_output_key=image_output_key,
        transform=transform,
        shuffle=False,
        return_all_fields=True,
    )
    return dataset


class LLaVACaptioner:
    """LLaVA captioner class."""

    def __init__(self, model_name: str = 'liuhaotian/llava-v1.5-13b', max_tokens: int = 512, compile: bool = False):
        self.model_name = model_name
        self.tokenizer, self.model, self.image_processor, self.context_len = self.load_llava()
        if compile:
            self.model = torch.compile(self.model)
        self.conv_mode = 'llava_v1'
        self.max_tokens = max_tokens
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def to(self, device: torch.device):
        self.device = device
        self.model.to(device)
        return self

    def load_llava(self):
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
        return load_pretrained_model(model_path, None, model_name)

    def add_image_tokens(self, prompt: str) -> str:
        if self.model.config.mm_use_im_start_end:
            prompt = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + prompt
        else:
            prompt = DEFAULT_IMAGE_TOKEN + '\n' + prompt
        return prompt

    def tokenize(self, prompt: str) -> torch.Tensor:
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
        return input_ids.unsqueeze(0).to(self.device)

    def get_outputs(self, image_batch: torch.Tensor, prompt: str) -> list:
        """Get the output from llava."""
        # Format the prompt
        prompt = self.add_image_tokens(prompt)
        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        input_ids = self.tokenize(prompt)
        # Prep the image
        image_batch = image_batch.to(self.device)  # In range (0, 1)
        # repeat the prompt along the batch dimension for each image
        input_ids = input_ids.repeat(image_batch.shape[0], 1)
        # Prep the image inputs
        image_tensor = self.image_processor.preprocess(image_batch, do_rescale=False,
                                                       return_tensors='pt')['pixel_values']
        # Forward through the model
        with torch.inference_mode():
            output_ids = self.model.generate(input_ids,
                                             images=image_tensor.half().to(device),
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


def resize_and_pad(image, target_width, target_height):
    """Resize and pad an image to the target size while maintaining aspect ratio.

    Args:
    - image (PIL Image): The image to be resized and padded.
    - target_width (int): The target width.
    - target_height (int): The target height.

    Returns:
    - PIL Image: The resized and padded image.
    """
    # Calculate the aspect ratio and find the smaller dimension.
    original_width, original_height = image.size
    aspect_ratio = original_width / original_height

    # Resize such that the smaller dimension fits the corresponding target dimension.
    if original_width < original_height:  # Width is smaller, match width
        resize_width = target_width
        resize_height = round(resize_width / aspect_ratio)
    elif original_width > original_height:  # Height is smaller, match height
        resize_height = target_height
        resize_width = round(resize_height * aspect_ratio)
    else:  # Image is square, match either dimension
        resize_width = target_width
        resize_height = target_height

    resized_image = image.resize((resize_width, resize_height), Image.ANTIALIAS)

    # Calculate padding
    pad_width_left = (target_width - resize_width) // 2
    pad_width_right = target_width - resize_width - pad_width_left

    pad_height_top = (target_height - resize_height) // 2
    pad_height_bottom = target_height - resize_height - pad_height_top

    # Apply asymmetric padding if necessary
    padded_image = ImageOps.expand(resized_image,
                                   border=(pad_width_left, pad_height_top, pad_width_right, pad_height_bottom),
                                   fill=0)

    return padded_image


if __name__ == '__main__':
    dataset = make_dataset(args.remote, args.local, image_key=args.image_key, image_output_key=args.image_output_key)
    dataset_len = len(dataset)
    to_tensor = transforms.ToTensor()

    # Device should be first gpu if available, else cpu
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    captioner = LLaVACaptioner(model_name=args.model_name, max_tokens=args.max_tokens, compile=args.compile).to(device)

    reader = dataset.shards[0]
    columns = dict(zip(reader.column_names, reader.column_encodings))
    columns[args.output_caption_key] = 'str'

    start = 0
    with MDSWriter(out=args.output, columns=columns) as out:
        for sample_id in range(0, len(dataset), args.batch_size):
            images = [dataset[i][args.image_output_key] for i in range(sample_id, sample_id + args.batch_size)]
            image_batch = [resize_and_pad(image, args.width, args.height) for image in images]
            image_batch = [to_tensor(image) for image in image_batch]
            image_batch = torch.stack(image_batch)
            if sample_id == args.batch_size:
                start = time.time()
            outputs = captioner.get_outputs(image_batch, args.llava_prompt)
            for output_id, output in enumerate(outputs):
                new_sample = dataset[sample_id + output_id]
                new_sample[args.output_caption_key] = output
                out.write(new_sample)
                print('*' * 120)
                print(output)
                print('*' * 120)
            if sample_id >= args.batch_size:
                current_time = time.time()
                print('-' * 120)
                print('Sample ID:', sample_id, "of", dataset_len)
                print('Time per image:', (current_time - start) / (sample_id))
                print('-' * 120)
