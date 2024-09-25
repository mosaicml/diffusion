# Copyright 2022 MosaicML Diffusion authors
# SPDX-License-Identifier: Apache-2.0

ARG BASE_IMAGE
FROM $BASE_IMAGE

ARG BRANCH_NAME
ARG DEP_GROUPS

ENV TORCH_CUDA_ARCH_LIST="8.0 8.6 8.7 8.9 9.0"

# Check for changes in setup.py.
# If there are changes, the docker cache is invalidated and a fresh pip installation is triggered.
ADD https://raw.githubusercontent.com/mosaicml/diffusion/$BRANCH_NAME/setup.py setup.py
RUN rm setup.py

# Install and uninstall diffusion to cache diffusion requirements
RUN git clone -b $BRANCH_NAME https://github.com/mosaicml/diffusion.git
RUN pip install --no-cache-dir "./diffusion${DEP_GROUPS}"
RUN pip uninstall -y diffusion
RUN rm -rf diffusion
