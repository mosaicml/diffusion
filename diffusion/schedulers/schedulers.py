# Copyright 2022 MosaicML Diffusion authors
# SPDX-License-Identifier: Apache-2.0

"""Custom schedulers for diffusion models."""

import numpy as np
import torch


def tangent_schedule(times):
    """Computes beta(t) sin(phi(t)) and cos(phi(t)).

    Implements a schedule where angle = time, hence beta = 2 * tan(t).
    """
    if isinstance(times, torch.Tensor):
        beta_t = 2 * torch.tan(times)
        cos_phi_t = torch.cos(times)
        sin_phi_t = torch.sin(times)
    else:
        beta_t = 2 * np.tan(times)
        cos_phi_t = np.cos(times)
        sin_phi_t = np.sin(times)
    return beta_t, sin_phi_t, cos_phi_t


class ContinuousTimeScheduler:
    """Scheduler for continuous time (variance preserving) diffusion models.

    Currently, the noise schedule is hardcoded to angle = time, hence beta = 2 * tan(t). This results in a maximum
    time of t_max = pi/2. For stability, t_max should be less than pi/2 during generation, as otherwise a divide by
    zero can occur.

    Args:
        t_max (float): The maximum timestep in the diffusion process. Default: `1.57`.
        num_inference_timesteps (int): The number of timesteps to use during inference. Default `50`.
        prediction_type (str): The type of prediction to use during inference. Must be one of 'sample', 'epsilon', or
            'v_prediction'. Default: `epsilon`.
        use_ode (bool): Whether to use Euler's method to integrate the probability flow ODE. Default: `False`.
        schedule_function (callable): A function that takes in a tensor of times and returns beta(t), sin(phi(t)),
            cos(phi(t)). Default: `tangent_schedule`.
    """

    def __init__(self,
                 t_max: float = 1.57,
                 num_inference_timesteps: int = 50,
                 prediction_type: str = 'epsilon',
                 use_ode: bool = False,
                 schedule_function=tangent_schedule):
        self.t_max = t_max
        self.num_inference_timesteps = num_inference_timesteps
        self.prediction_type = prediction_type
        self.use_ode = use_ode
        self.schedule_function = schedule_function

        self.timesteps = np.linspace(self.t_max, 0, num=num_inference_timesteps, endpoint=False)
        self.init_noise_sigma = 1.0  # Needed to work with our generate function that uses huggingface schedulers

    def __len__(self):
        return self.num_inference_timesteps

    def set_timesteps(self, num_inference_timesteps):
        self.num_inference_timesteps = num_inference_timesteps
        self.timesteps = np.linspace(self.t_max, 0, num=num_inference_timesteps, endpoint=False)

    def add_noise(self, inputs, noise, timesteps):
        # expand timesteps to the right number of dimensions
        while len(timesteps.shape) < len(inputs.shape):
            timesteps = timesteps.unsqueeze(-1)
        # compute sin, cos of the angle
        _, sin_phi, cos_phi = self.schedule_function(timesteps)
        # combine the signal with the noise
        return cos_phi * inputs + sin_phi * noise

    def get_velocity(self, inputs, noise, timesteps):
        # v is defined by -sin(t) * inputs + cos(t) * noise
        # expand timesteps to the right number of dimensions
        while len(timesteps.shape) < len(inputs.shape):
            timesteps = timesteps.unsqueeze(-1)
        # compute sin, cos of the angle
        _, sin_phi, cos_phi = self.schedule_function(timesteps)
        return -sin_phi * inputs + cos_phi * noise

    def scale_model_input(self, model_input, t):
        # Needed to work with our generate function that uses huggingface schedulers
        return model_input

    def step(self, model_output, t, model_input, generator=None):
        if t == 0:
            return {'prev_sample': model_input}
        # compute beta, sin, cos
        beta_t, sin_phi_t, cos_phi_t = self.schedule_function(t)
        # Compute the time deltas
        # A more general implementation would allow for unequally spaced timesteps.
        dt = self.t_max / self.timesteps.shape[0]
        # Get the predicted clean input from each of the prediction types
        if self.prediction_type == 'sample':
            x_0 = model_output
        elif self.prediction_type == 'epsilon':
            x_0 = (model_input - sin_phi_t * model_output) / cos_phi_t
        elif self.prediction_type == 'v_prediction':
            x_0 = cos_phi_t * model_input - sin_phi_t * model_output
        else:
            raise ValueError(
                f'prediction type must be one of sample, epsilon, or v_prediction. Got {self.prediction_type}')
        # Compute the score function
        score = -(model_input - cos_phi_t * x_0) / np.square(sin_phi_t)
        # Compute the previous sample x_t -> x_t-1
        if self.use_ode:
            # Use Euler's method to integrate the probability flow ODE
            x_prev = model_input + 0.5 * (model_input + score) * beta_t * dt
        else:
            # Use Euler-Maruyama to integrate the reverse SDE
            x_prev = model_input + (0.5 * model_input + score) * beta_t * dt
            # Add the noise term
            x_prev += np.sqrt(beta_t * dt) * torch.randn_like(model_input)
        return {'prev_sample': x_prev}
