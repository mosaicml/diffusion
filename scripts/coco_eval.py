# Copyright 2022 MosaicML Diffusion authors
# SPDX-License-Identifier: Apache-2.0

"""Script to evaluate a trained sd-2 model on coco-captions."""

import argparse

from composer import Trainer
from composer.loggers import WandBLogger
from composer.utils import reproducibility
from torchmetrics.image.fid import FrechetInceptionDistance

from diffusion.datasets.coco.coco_captions import build_streaming_cocoval_dataloader
from diffusion.models import stable_diffusion_2

parser = argparse.ArgumentParser()
parser.add_argument('--seed', default=17, type=int)
parser.add_argument('--remote', type=str, help='path to coco streaming dataset')
args = parser.parse_args()

reproducibility.seed_all(args.seed)
coco_val_dataloader = build_streaming_cocoval_dataloader(
    remote=args.remote,
    local='/tmp/mds-cache/mds-coco-2014-val-10k/',
    resize_size=512,
    batch_size=4,
    prefetch_factor=2,
    num_workers=8,
    persistent_workers=True,
    pin_memory=True,
)

fid = FrechetInceptionDistance(normalize=True)
model = stable_diffusion_2(model_name='stabilityai/stable-diffusion-2-base', val_metrics=[fid])

logger = WandBLogger(name='coco-val2014-10k-fid')

#setup wandb
trainer = Trainer(model=model, eval_dataloader=coco_val_dataloader, loggers=logger)
trainer.eval()
