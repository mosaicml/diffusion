# Copyright 2022 MosaicML Diffusion authors
# SPDX-License-Identifier: Apache-2.0

"""Script to compute FID and CLIP score on a trained model with the COCO dataset.

Currently, one needs to install composer via

`pip install git+https://github.com/nik-mosaic/composer.git@0384dcb56183ae2a6b61cf4fa07b45358d335652`

for the CLIPScore metric to work.
"""

import argparse

from composer import Trainer
from composer.loggers import WandBLogger
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.multimodal import CLIPScore

from diffusion.callbacks import LogDiffusionImages
from diffusion.datasets import build_streaming_cocoval_dataloader
from diffusion.models import stable_diffusion_2

parser = argparse.ArgumentParser()
parser.add_argument('--remote', type=str, help='path to coco streaming dataset')
parser.add_argument('--load_path', default=None, type=str, help='path to load model from')
parser.add_argument('--guidance_scale', default=1.0, type=float, help='guidance scale to evaluate at')
parser.add_argument('--size', default=512, type=int, help='image size to evaluate at')
parser.add_argument('--no_crop', action='store_false', help='use resize instead of crop on COCO images.')
parser.add_argument('--batch_size', default=16, type=int, help='eval batch size to use')
parser.add_argument('--seed', default=17, type=int)
parser.add_argument('--wandb', action='store_true', help='log to wandb')
parser.add_argument('--project', default='diffusion-eval', type=str, help='wandb project to use')
parser.add_argument('--name', default='fid-clip-evaluation', type=str, help='wandb name to use')

args = parser.parse_args()

# Create the eval dataloader
coco_val_dataloader = build_streaming_cocoval_dataloader(
    remote=args.remote,
    local='/tmp/mds-cache/mds-coco-2014-val-fid-clip/',
    resize_size=args.size,
    use_crop=args.no_crop,
    batch_size=args.batch_size,
    prefetch_factor=2,
    num_workers=8,
    persistent_workers=True,
    pin_memory=True,
)

fid = FrechetInceptionDistance(normalize=True)
# CLIP score here will use the default model, which is openai/clip-vit-large-patch14
clip = CLIPScore()

val_guidance_scales = [args.guidance_scale]

# If a checkpoint is specified, evaluate it. Otherwise evaluate the pretrained SD2.0 model.
if args.load_path is not None:
    name = args.name
    model, _ = stable_diffusion_2(
        model_name='stabilityai/stable-diffusion-2-base',
        val_metrics=[fid, clip],
        val_guidance_scales=val_guidance_scales,
        val_seed=args.seed,
        pretrained=False,
        encode_latents_in_fp16=False,
        fsdp=False,
    )
else:
    name = args.name + '-pretrained'
    model, _ = stable_diffusion_2(
        model_name='stabilityai/stable-diffusion-2-base',
        val_metrics=[fid, clip],
        val_guidance_scales=val_guidance_scales,
        val_seed=args.seed,
        pretrained=True,
        encode_latents_in_fp16=False,
        fsdp=False,
    )

# Set up wandb if desired
loggers = []
if args.wandb:
    wandb_logger = WandBLogger(project=args.project, name=name)
    loggers.append(wandb_logger)

# Image logging callback
prompts = [
    'a couple waiting to cross the street underneath an umbrella.', 'three men walking in the rain with umbrellas.',
    'a man is riding a red motor cycle, with baskets.', 'a clock that has animal pictures instead of numbers.',
    'a brightly decorated bus sits on the road.',
    'a horse bucking with a rider on it, completely vertical, with another horse and onlookers.',
    'a white and blue bus is on a city street at night.', 'a large clock tower on a building by a river',
    'beans and other food is sitting on a plate.', 'a group of people that are standing up on a tennis court'
]

log_images = LogDiffusionImages(guidance_scale=args.guidance_scale, prompts=prompts, size=args.size, seed=args.seed)
callbacks = [log_images]

# Run the evaluation
trainer = Trainer(
    model=model,
    load_path=args.load_path,
    load_weights_only=True,
    eval_dataloader=coco_val_dataloader,
    loggers=loggers,
    callbacks=callbacks,
)
trainer.eval()
