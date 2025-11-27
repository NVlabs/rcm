import torch


class FlowEulerSampler:

    def __init__(
        self,
        num_train_timesteps=1000,
        sigma_max=1.0,
        sigma_min=0.0,
    ):
        self.num_train_timesteps = num_train_timesteps
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min

    def set_timesteps(self, num_inference_steps=100, shift=3.0, device="cuda"):
        self.sigmas = torch.linspace(self.sigma_max, self.sigma_min, num_inference_steps + 1)[:-1]
        self.sigmas = shift * self.sigmas / (1 + (shift - 1) * self.sigmas)
        self.timesteps = self.sigmas * self.num_train_timesteps
        self.sigmas = self.sigmas.to(device)
        self.timesteps = self.timesteps.to(device)

    def step(self, model_output, timestep, sample):
        timestep_id = torch.argmin((self.timesteps - timestep).abs(), dim=0)
        sigma = self.sigmas[timestep_id]
        if timestep_id + 1 >= len(self.timesteps):
            sigma_ = 0
        else:
            sigma_ = self.sigmas[timestep_id + 1]
        prev_sample = sample + model_output * (sigma_ - sigma)
        return prev_sample
