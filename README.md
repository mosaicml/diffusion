<h2><p align="center">Stable Diffusion Training with MosaicML</p></h2>

<p align="center">
    <a href="https://mosaicml.me/slack">
        <img alt="Chat @ Slack" src="https://img.shields.io/badge/slack-chat-2eb67d.svg?logo=slack">
    </a>
    <a href="https://github.com/mosaicml/examples/blob/main/LICENSE">
        <img alt="License" src="https://img.shields.io/badge/License-Apache%202.0-green.svg">
    </a>
</p>
<br />


This repo contains code used to train your own Stable Diffusion model on your own data.

<p align="center">
  <picture>
    <img alt="training curve" src="./assets/training-curve.png" width="75%">
  </picture>
</p>

# Results
Results from our Mosaic Diffusion model after training for 550k iterations at 256x256 resolution, then for 850k iterations at 512x512:

<p align="center">
  <picture>
    <img alt="training curve" src="./assets/collage.svg" width="100%">
  </picture>
</p>

<p align="center">
  <picture>
    <img alt="model outputs of SD-2-1" src="./assets/mushroom-collage.svg">
  </picture>
</p>

# Prerequisites

Here are the system settings we recommend to start training your own diffusion models:

- Use a Docker image with PyTorch 1.13+, e.g. [MosaicML's PyTorch base image](https://hub.docker.com/r/mosaicml/pytorch/tags)
  - Recommended tag: `mosaicml/pytorch_vision:1.13.1_cu117-python3.10-ubuntu20.04`
  - This image comes pre-configured with the following dependencies:
    - PyTorch Version: 1.13.1
    - CUDA Version: 11.7
    - Python Version: 3.10
    - Ubuntu Version: 20.04
- Use a system with NVIDIA GPUs

- For running on NVIDIA H100s, use a docker image with PyTorch 2.0+ e.g. [MosaicML's PyTorch base image](https://hub.docker.com/r/mosaicml/pytorch/tags)
  - Recommended tag: `mosaicml/pytorch_vision:2.0.1_cu118-python3.10-ubuntu20.04`
  - This image comes pre-configured with the following dependencies:
    - PyTorch Version: 2.0.1
    - CUDA Version: 11.8
    - Python Version: 3.10
    - Ubuntu Version: 20.04
  - Depending on the training config, an additional install of `xformers` may be needed:
    ```
    pip install -U ninja
    pip install -U git+https://github.com/facebookresearch/xformers
    ```

# How many GPUs do I need?

We benchmarked the U-Net training throughput as we scale the number of A100 GPUs from 8 to 128. Our time estimates are based on training Stable Diffusion 2.0 base on 1,126,400,000 images at 256x256 resolution and 1,740,800,000 images at 512x512 resolution. Our cost estimates are based on $2 / A100-hour. Since the time and cost estimates are for the U-Net only, these only hold if the VAE and CLIP latents are computed before training. It took 3,784 A100-hours (cost of $7,600) to pre-compute the VAE and CLIP latents offline. If you are computing VAE and CLIP latents while training, expect a 1.4x increase in time and cost.

| Number of A100s | Throughput for UNet @ 256x256 (images / second) | Throughput for UNet @ 512x512 (images / second) | Days to Train on MosaicML Cloud | Approx. Cost on MosaicML Cloud |
|:---------------:|:-----------------------------------------------:|:-----------------------------------------------:|:-------------------------------:|:------------------------------:|
|        8        |                       1100                      |                       290                       |              81.33              |             $31,230            |
|        16       |                       2180                      |                       585                       |              40.42              |             $31,043            |
|        32       |                       4080                      |                       1195                      |              20.06              |             $30,805            |
|        64       |                       8530                      |                       2340                      |              10.14              |             $31,146            |
|       128       |                      11600                      |                       4590                      |               5.51              |             $33,874            |

# Clone the repo and install requirements

```
git clone https://github.com/mosaicml/diffusion.git
cd diffusion
pip install -e .
```

# Data Prep

If you are interested in training on LAION-5B or evaluating on COCO Captions, we provide [scripts](https://github.com/mosaicml/diffusion/tree/main/scripts) to download and process these datasets into Streaming datasets.

Alternatively, you can use your own image-caption dataset(s) as long as samples are returned as a dictionary from a PyTorch Dataset class. To use a custom dataset with our configurations, define a function that returns a PyTorch DataLoader for the custom dataset (for an example, see [`build_streaming_laion_dataloader()`](https://github.com/mosaicml/diffusion/blob/34e95ef50836581fab1bec3effaed8fa9d0ae464/diffusion/datasets/laion/laion.py#L115)). The best way to add custom code is to fork this repo, then add the python scripts to `diffusion/datasets`.

# Adjust config

The configurations for the two phases of training are specified at [`SD-2-base-256.yaml`](https://github.com/mosaicml/diffusion/blob/main/yamls/hydra-yamls/SD-2-base-256.yaml) and [`SD-2-base-512.yaml`](https://github.com/mosaicml/diffusion/blob/main/yamls/hydra-yamls/SD-2-base-512.yaml). A few fields are left blank that need to be filled in to start training. The `dataset` field is the primary field to change. If you downloaded and converted the LAION-5B dataset into your own Streaming dataset, change the `remote` field under `train_dataset` to the bucket containing your streaming LAION-5B. Similarly for COCO validation, change the `remote` field under `eval_dataset` to the bucket containing your streaming COCO.

If you opted to use your own datasets, change the `_target_` field under both `train_dataset` and `eval_dataset` to contain the absolute path to the function that returns the PyTorch DataLoader for your dataset. Replace the fields after `_target_` with the arguments for your function.

If you have not pre-computed the VAE and CLIP latents for your dataset, set `precomputed_latents` field to `false`.

## Train the model

Once the configurations have been updated, start training at 256x256 resolution by running:
```
composer run.py --config-path yamls/hydra-yamls --config-name SD-2-base-256.yaml
```
Next, start training at 512x512 resolution by running:
```
composer run.py --config-path yamls/hydra-yamls --config-name SD-2-base-512.yaml
```
You can also log generated images to Weights and Biases throughout training to qualitatively measure model performance. This is done by specifying the `LogDiffusionImages` callback class under `callbacks` in a configuration file like so:
```
  image_monitor:
    _target_: diffusion.callbacks.log_diffusion_images.LogDiffusionImages
    prompts: # add any prompts you would like to visualize
    - A dog wearing a spacesuit
    - An astronaut riding a horse
    size: 256 # generated image resolution
    guidance_scale: 3
```

# Offline Eval
We also provide an offline evaluation script to compute common metrics on a saved checkpoint and an image+prompt dataset. To use this, run:
```
composer composer run_eval.py --config-path yaml/dir --config-name eval-clean-fid
```
This will compute FID, KID, CLIP-FID, and CLIP score at configurable guidance scales using the image+prompts pairs. See the yaml template [here](https://github.com/mosaicml/diffusion/blob/main/yamls/hydra-yamls/eval-clean-fid.yaml) for more configuration options, including logging to Weights and Biases.

# Training an SDXL model
We support training SDXL architectures and provide sample yamls for each stage of training. Once the sample configurations have been updated for your own data and use case, start training at 256x256 resolution by running:
```
composer run.py --config-path yamls/hydra-yamls --config-name SDXL-base-256.yaml
```
Next, start training at 512x512 resolution by running:
```
composer run.py --config-path yamls/hydra-yamls --config-name SDXL-base-512.yaml
```
Finally, train with aspect ratio bucketing at 1024x1024 by running the following. The sample config assumes data is split into separate directories corresponding to different aspect ratio buckets.
```
composer run.py --config-path yamls/hydra-yamls --config-name SDXL-base-1024.yaml
```

# Contact Us
If you run into any problems with the code, please file Github issues directly to this repo.
If you want to work with us directly, please reach out to us at demo@mosaicml.com!
