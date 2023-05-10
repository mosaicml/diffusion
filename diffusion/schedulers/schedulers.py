# Copyright 2022 MosaicML Diffusion authors
# SPDX-License-Identifier: Apache-2.0

# scheduler should implement .step(model_output, t, images, generator=rng_generator).prev_sample which returns the previous sample

import numpy as np
import torch


class ContinuousTimeScheduler:

    def __init__(self, t_max=3.14159 / 2, num_inference_timesteps=50, prediction_type='epsilon'):
        self.t_max = t_max
        self.num_inference_timesteps = num_inference_timesteps
        self.prediction_type = prediction_type

        self.timesteps = np.linspace((1 - 0.1) * self.t_max, 0, num=num_inference_timesteps, endpoint=False)
        self.init_noise_sigma = 1.0

    def __len__(self):
        return self.num_inference_timesteps

    def set_timesteps(self, num_inference_timesteps):
        self.num_inference_timesteps = num_inference_timesteps
        self.timesteps = np.linspace((1 - 0.1) * self.t_max, 0, num=num_inference_timesteps, endpoint=False)

    def add_noise(self, inputs, noise, timesteps):
        # expand timesteps to the right number of dimensions
        while len(timesteps.shape) < len(inputs.shape):
            timesteps = timesteps.unsqueeze(-1)
        # compute sin, cos of the timesteps
        sin_t = torch.sin(timesteps)
        cos_t = torch.cos(timesteps)
        # combine the signal with the noise
        return cos_t * inputs + sin_t * noise

    def get_velocity(self, inputs, noise, timesteps):
        # v is defined by -sin(t) * inputs + cos(t) * noise
        # expand timesteps to the right number of dimensions
        while len(timesteps.shape) < len(inputs.shape):
            timesteps = timesteps.unsqueeze(-1)
        sin_t = torch.sin(timesteps)
        cos_t = torch.cos(timesteps)
        return -sin_t * inputs + cos_t * noise

    def scale_model_input(self, model_input, t):
        return model_input

    def step(self, model_output, t, model_input, generator=None):
        if t == 0:
            return {'prev_sample': model_input}
        # compute sin, cos, tan of the timesteps
        sin_t = np.sin(t)
        cos_t = np.cos(t)
        tan_t = np.tan(t)
        beta_t = 2 * tan_t
        # Compute the previous sample x_t -> x_t-1
        dt = self.t_max / self.timesteps.shape[0]
        # Get the predicted clean input from each of the prediction types
        if self.prediction_type == 'sample':
            x_0 = model_output
        elif self.prediction_type == 'epsilon':
            x_0 = (model_input - sin_t * model_output) / cos_t
        elif self.prediction_type == 'v_prediction':
            x_0 = cos_t * model_input - sin_t * model_output
        # Compute the score function
        score = -(model_input - cos_t * x_0) / np.square(sin_t)
        # Compute the previous sample
        x_prev = model_input + 0.5 * beta_t * model_input * dt + beta_t * score * dt
        # Add the noise term
        x_prev += np.sqrt(beta_t * dt) * torch.randn_like(model_input)
        print(t, x_prev.mean().item(), x_prev.std().item(), x_prev.max().item(), x_prev.min().item())
        return {'prev_sample': x_prev}
