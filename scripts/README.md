This directory contains code to convert LAION and COCO captions into [Streaming](https://github.com/mosaicml/streaming) datasets. Once the datasets are converted into Streaming datasets, the data can be streamed to any desired endpoint.


# LAION Dataset

We provide some helper scripts to download and process the LAION dataset into a [streaming dataset](https://docs.mosaicml.com/projects/streaming/en/stable/).
`laion_download_all.sh` and `laion_download_aesthetic.sh` use [img2dataset](https://github.com/rom1504/img2dataset) to download the images and metadata for the full dataset and aesthetics only dataset, respectively.
`laion_cloudwriter.py` runs in parallel, converting images to the streaming MDS format and uploading them to a cloud storage bucket. Additionally,
it buckets different images by resolution. `mcloud/laion2b-en-interactive.yaml` is an example config file for this dataset preparation. Note that this script
does not require a GPU.

`precompute_latents.py` takes a streaming LAION dataset and attaches the precomputed VAE and CLIP latents to each sample. As we often train latent diffusion
models for multiple epochs, this lets us avoid recomputing the latents for each epoch. `mcloud/precompute-latents.yaml` is an example config file for this. Note that
this script requires a GPU.

# COCO Dataset

We used the images and captions from the COCO 2014 validation set to measure the FID score of our model. The data can be downloaded from the [COCO website](https://cocodataset.org/#download) by clicking on the links "2014 Val images [41K/6GB]" and "2014 Train/Val annotations [241MB]" for images and annotations, respectively. The `convert_coco.py` script takes an object store location and the paths to the COCO images and annotations, then converts the data to a Streaming dataset and uploads the Streaming dataset to the specified object store location. Please see the [Streaming](https://github.com/mosaicml/streaming) repository for more information on how to configure your object storage.
