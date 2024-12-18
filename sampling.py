import os

import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt

from diffusion_utilities import plot_sample
from networks import ContextUnet


def sample(config, device, model=None, diffusion_params=None):
    if model is None:
        model = ContextUnet(config.input_channels, config.base_channels, config.classes, config.image_size)
        # load in model weights and set to eval mode
        checkpoints_path = os.path.join(config.log_folder, config.run_string, "checkpoints")
        model.load_state_dict(torch.load(f"{checkpoints_path}/context_model_{config.epochs - 1}.pth",
                                         map_location=device))
        print("Loaded in Context Model")

    model.eval()

    # sample from the model to visualize a few samples
    plt.clf()
    ctx = F.one_hot(torch.randint(0, 5, (32,)), 5).to(device=device).float()
    samples, intermediate = sample_ddim_context(config, device, model, 32, ctx, diffusion_params)
    preview_path = os.path.join(config.log_folder, config.run_string, "previews")
    animation_ddim_context = plot_sample(intermediate, 32, 4, preview_path, "ani_run",
                                         None, save=False)
    # HTML(animation_ddim_context.to_jshtml())


# fast sampling algorithm with context
@torch.no_grad()
def sample_ddim_context(config, device, model, number_of_samples, context, diffusion_params, n=20):
    time_steps = config.time_steps
    # x_T ~ N(0, 1), sample initial noise
    samples = torch.randn(number_of_samples, 4, config.image_size, config.image_size).to(device)

    # array to keep track of generated steps for plotting
    intermediate = []
    step_size = time_steps // n
    for i in range(time_steps, 0, -step_size):
        print(f'sampling timestep {i:3d}', end='\r')

        # reshape time tensor
        t = torch.tensor([i / time_steps])[:, None, None, None].to(device)

        eps = model(samples, t, c=context)  # predict noise e_(x_t,t)
        samples = denoise_ddim(samples, i, i - step_size, eps, diffusion_params)
        intermediate.append(samples.detach().cpu().numpy())

    intermediate = np.stack(intermediate)
    return samples, intermediate


# define sampling function for DDIM
# removes the noise using ddim
def denoise_ddim(x, t, t_prev, pred_noise, diffusion_params):
    ab_t = diffusion_params["ab_t"]
    ab = ab_t[t]
    ab_prev = ab_t[t_prev]

    x0_pred = ab_prev.sqrt() / ab.sqrt() * (x - (1 - ab).sqrt() * pred_noise)
    dir_xt = (1 - ab_prev).sqrt() * pred_noise

    return x0_pred + dir_xt
