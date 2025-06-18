from typing import *

import torch
from diffusers import DDIMScheduler


class MyDDIMScheduler(DDIMScheduler):
    def inv_step(
        self,
        model_output: torch.FloatTensor,
        timestep: Union[float, torch.FloatTensor],
        sample: torch.FloatTensor,
        generator: Optional[torch.Generator] = None,
        return_dict: bool = True,
    ) -> torch.Tensor:
        """
        DDIM Inversion step.
        """

        # I copy the DDIM Scheduler step code.
        # but I omit many config options for simplicity

        # 1. get next step for inversion (note same "frame" as regular step)
        prev_timestep = max(
            timestep - self.config.num_train_timesteps // self.num_inference_steps,
            torch.tensor(0),
        )

        # print(prev_timestep.item(), "->", timestep.item())

        # 2. compute alphas, betas
        # observe I use prev, (current) instead of (current), next
        # this is because I want the notation to be compatible
        # with most DDIM inversion formulations
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep]

        def phi(alpha):
            return (1 / alpha - 1) ** 0.5

        sample_inv = (
            torch.sqrt(alpha_prod_t / alpha_prod_t_prev) * sample
            + torch.sqrt(alpha_prod_t)
            * (phi(alpha_prod_t) - phi(alpha_prod_t_prev))
            * model_output
        )

        assert sample_inv.shape == sample.shape

        return (sample_inv,)  # tuple for compat