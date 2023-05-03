# Inference with Diffusion Models on the MosaicML platform

In this folder, we provide an example of how to create and deploy a diffusion model on the MosaicML platform.

You'll find in this folder:

- `inference_model.py` - a custom model class that implements forward logic for a diffusion model
- `mosaic_inference.yaml` - a configuration file that specifies information about the deployment, such as what Docker image to use

## Prerequisites

First, you'll need access to MosaicML's inference service. You can request access [here](https://forms.mosaicml.com/demo).

Once you have access, you just need to install the MosaicML client and command line interface:
```bash
pip install mosaicml-cli
```

# Deploying the Model

Deploying the model is as simple as running `mcli deploy -f yamls/instructor_model.yaml`.

The logs for a successful deployment should look something like:

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="./assets/instructor_model_logs.png">
  <img alt="Logs from server startup." src="./assets/instructor_model_logs.png">
</picture>

# Running Inference

Once the model has successfully been deployed, we can run inference by running `mcli predict <deployment_name> --inputs <input_dictionary>`.