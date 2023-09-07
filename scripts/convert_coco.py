# Copyright 2022 MosaicML Diffusion authors
# SPDX-License-Identifier: Apache-2.0

"""Script to convert mscoco captions to a streaming dataset."""

import json
from argparse import ArgumentParser

import numpy as np
from PIL import Image
from streaming.base import MDSWriter
from tqdm import tqdm

args = ArgumentParser()
args.add_argument('--data_path',
                  type=str,
                  required=True,
                  help='Path to local coco2014 validation image directory.\
                            ex: data/val2014/')
args.add_argument('--annotation_path',
                  type=str,
                  required=True,
                  help='Path to local coco2014 val annotation json.\
                            ex: COCO2014-val/annotations/captions_val2014.json')
args.add_argument('--remote', type=str, default='', help='Remote path to upload MDS-formatted shards to.')


def main(args):
    """Converts coco captions to MDS."""
    data_path = args.data_path
    captions_path = args.captions_path
    data = json.loads(captions_path)

    # create {image_id: list[captions]} dictionary
    coco_captions = {}
    for sample in data['annotations']:
        image_id = sample['image_id']
        caption = sample['caption']
        if image_id in coco_captions:
            coco_captions[image_id]['captions'].append(caption.replace('\n', ''))
        else:
            coco_captions[image_id] = {'captions': [caption]}

        if 'image_file' not in coco_captions[image_id]:
            image_file = f'{data_path}/val2014/COCO_val2014_{image_id:012d}.jpg'
            coco_captions[image_id]['image_file'] = image_file

    print('creating random subset')
    np.random.seed(0)
    subset_10k_1_ids = np.random.choice(list(coco_captions.keys()), size=10000, replace=False)
    coco_captions_subset_10k_1 = {}
    for idx in subset_10k_1_ids:
        coco_captions_subset_10k_1[idx] = coco_captions[idx]

    fields = {'image': 'jpeg', 'captions': 'json'}
    with MDSWriter(out=args.remote, columns=fields) as out:
        for sample in tqdm(coco_captions_subset_10k_1):
            image = Image.open(coco_captions_subset_10k_1[sample]['image_file'])
            captions = coco_captions_subset_10k_1[sample]['captions']
            mds_sample = {'image': image, 'captions': captions}
            out.write(mds_sample)


if __name__ == '__main__':
    main(args.parse_args())
