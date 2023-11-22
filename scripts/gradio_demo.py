# Copyright 2022 MosaicML Diffusion authors
# SPDX-License-Identifier: Apache-2.0

"""Local Gradio demo script."""

import argparse
import base64
from io import BytesIO

import gradio as gr
from PIL import Image

from diffusion.inference import StableDiffusionInference, StableDiffusionXLInference

parser = argparse.ArgumentParser()
parser.add_argument('--load_path', default=None, type=str, help='Path to load checkpoint from')
parser.add_argument('--sdxl', default=False, action='store_true', help='Use SDXL model')
parser.add_argument('--resolution', default=512, type=int, help='Resolution to generate images at.')
parser.add_argument('--share', default=False, action='store_true', help='Share the demo publicly.')
parser.add_argument('--progress_bar', default=False, action='store_true', help='Show progress bar.')
args = parser.parse_args()


class ImageGenerator:
    """Wrapper class to interface model deployment with Gradio."""

    def __init__(self, model_deployment, resolution, progress_bar):
        self.model_deployment = model_deployment
        self.resolution = resolution
        self.progress_bar = progress_bar

    def get_images(self, text, seed, guidance_scale):
        input_dict = {
            'input': {
                'prompt': text
            },
            'parameters': {
                'height': self.resolution,
                'width': self.resolution,
                'seed': seed,
                'guidance_scale': guidance_scale,
                'num_images_per_prompt': 4,
                'progress_bar': self.progress_bar
            }
        }

        img = self.model_deployment.predict([input_dict])
        img_data = [base64.b64decode(i) for i in img]
        imgs = [Image.open(BytesIO(i)) for i in img_data]
        return imgs


if __name__ == '__main__':
    if args.sdxl:
        model_deployment = StableDiffusionXLInference(local_checkpoint_path=args.load_path)
    else:
        model_deployment = StableDiffusionInference(local_checkpoint_path=args.load_path)

    image_generator = ImageGenerator(model_deployment, args.resolution, args.progress_bar)

    with gr.Blocks() as demo:
        with gr.Column():
            with gr.Row():
                text = gr.Textbox(lines=1, label='Text prompt')
            with gr.Row():
                guidance_scale = gr.Slider(minimum=1, maximum=15, value=5, step=0.1, label='Guidance scale')
                seed = gr.Slider(minimum=0, maximum=1e6, randomize=True, step=1, label='Random seed')
            with gr.Row():
                generate_button = gr.Button('Generate')
            with gr.Row():
                img0 = gr.Image(type='pil', label='Image 1', height=args.resolution, width=args.resolution)
                img1 = gr.Image(type='pil', label='Image 2', height=args.resolution, width=args.resolution)
            with gr.Row():
                img2 = gr.Image(type='pil', label='Image 3', height=args.resolution, width=args.resolution)
                img3 = gr.Image(type='pil', label='Image 4', height=args.resolution, width=args.resolution)
        generate_button.click(image_generator.get_images,
                              inputs=[text, seed, guidance_scale],
                              outputs=[img0, img1, img2, img3])

    demo.launch(share=args.share)
