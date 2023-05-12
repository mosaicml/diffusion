# Copyright 2022 MosaicML Diffusion authors
# SPDX-License-Identifier: Apache-2.0

"""Custom schedulers for diffusion models."""

import numpy as np
import torch


class ContinuousTimeScheduler:
    """Scheduler for continuous time (variance preserving) diffusion models.

    Currently, the noise schedule is hardcoded to angle = time, hence beta = 2 * tan(t). This results in a maximum
    time of t_max = pi/2. For stability, t_max should be less than pi/2 during generation, as otherwise a divide by
    zero can occur.

    Args:
        t_max (float): The maximum timestep in the diffusion process. Defaults to 1.57.
        num_inference_timesteps (int): The number of timesteps to use during inference. Defaults to 50.
        prediction_type (str): The type of prediction to use during inference. Must be one of 'sample', 'epsilon', or
            'v_prediction'. Defaults to 'epsilon'.
        use_ode (bool): Whether to use Euler's method to integrate the probability flow ODE. Defaults to False.
    """

    def __init__(self,
                 t_max: float = 1.57,
                 num_inference_timesteps: int = 50,
                 prediction_type: str = 'epsilon',
                 use_ode: bool = False):
        self.t_max = t_max
        self.num_inference_timesteps = num_inference_timesteps
        self.prediction_type = prediction_type
        self.use_ode = use_ode
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
        # A more general implementation would first compute the angle from the timesteps
        sin_t = torch.sin(timesteps)
        cos_t = torch.cos(timesteps)
        # combine the signal with the noise
        return cos_t * inputs + sin_t * noise

    def get_velocity(self, inputs, noise, timesteps):
        # v is defined by -sin(t) * inputs + cos(t) * noise
        # expand timesteps to the right number of dimensions
        while len(timesteps.shape) < len(inputs.shape):
            timesteps = timesteps.unsqueeze(-1)
        # A more general implementation would first compute the angle from the timesteps
        sin_t = torch.sin(timesteps)
        cos_t = torch.cos(timesteps)
        return -sin_t * inputs + cos_t * noise

    def scale_model_input(self, model_input, t):
        # Needed to work with our generate function that uses huggingface schedulers
        return model_input

    def step(self, model_output, t, model_input, generator=None):
        if t == 0:
            return {'prev_sample': model_input}
        # Compute beta at the current timestep
        tan_t = np.tan(t)
        beta_t = 2 * tan_t
        # compute sin, cos, tan of the angle
        # A more general implementation would first calculate an angle from beta_t and the timestep
        # Then compute sin, cos.
        sin_t = np.sin(t)
        cos_t = np.cos(t)
        # Compute the time deltas
        # A more general implementation would allow for unequally spaced timesteps.
        dt = self.t_max / self.timesteps.shape[0]
        # Get the predicted clean input from each of the prediction types
        if self.prediction_type == 'sample':
            x_0 = model_output
        elif self.prediction_type == 'epsilon':
            x_0 = (model_input - sin_t * model_output) / cos_t
        elif self.prediction_type == 'v_prediction':
            x_0 = cos_t * model_input - sin_t * model_output
        else:
            raise ValueError(
                f'prediction type must be one of sample, epsilon, or v_prediction. Got {self.prediction_type}')
        # Compute the score function
        score = -(model_input - cos_t * x_0) / np.square(sin_t)
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
