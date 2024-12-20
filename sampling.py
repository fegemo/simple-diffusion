import os
import re
import sys
import logging

import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt

from diffusion_utilities import plot_grid, plot_sample, create_diffusion_params
from networks import ContextUnet
from configuration import OptionParser

def sample(config, device, model=None, epoch=None):
    loaded_model_checkpoint = False
    if model is None:
        # load in model weights and set to eval mode
        checkpoint_path = config.ckpt
        model = ContextUnet(config.input_channels, config.base_channels, config.classes, config.image_size).to(device)
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        loaded_model_checkpoint = True
        logging.info(f"Loaded in Context Model: {checkpoint_path}")
    
        extracted_epoch = re.findall(r"\d+", checkpoint_path.split("/")[3])[0]
        epoch = int(extracted_epoch)

    sampler = sample_ddim_context if config.sampler == "ddim" else sample_ddpm_context

    model.eval()

    # sample from the model to visualize a few samples
    # plt.clf()
    number_of_classes = config.classes
    # ctx = F.one_hot(torch.randint(0, 5, (32,)), 5).to(device=device).float()
    number_of_samples = 6 * number_of_classes
    sample_class_indices = torch.linspace(0, number_of_samples - 1, steps=number_of_samples) // (number_of_samples/number_of_classes)
    ctx = F.one_hot(sample_class_indices.to(device=device).long()).float()
    samples, intermediate = sampler(config, device, model, number_of_samples, ctx, create_diffusion_params(config, device))
    
    execution_run_string = config.run_string
    if loaded_model_checkpoint:
        # we're sampling without training... 
        # so we extract the run_string from the --ckpt option
        extracted_run_string = config.ckpt.split("/")[1]
        preview_path = os.path.join(config.log_folder, extracted_run_string, "previews")
    else:
        # we've just finished training, so we use the same run_string of the training
        preview_path = os.path.join(config.log_folder, config.run_string, "previews")
    os.makedirs(preview_path, exist_ok=True)

    # creates images of the final steps
    for ts in [-1, -4, -10]:
        ts_for_file = -1 * ts - 1
        final_image_path = os.path.join(preview_path, f"{config.sampler}_at_epoch_{epoch}_{ts_for_file}ts_generated_at_{execution_run_string}.png")
        plot_grid(torch.from_numpy(intermediate[ts]), number_of_samples, number_of_classes, final_image_path)
    
    # creates the animation
    animation_context = plot_sample(intermediate, number_of_samples, number_of_classes, preview_path, "ani_run",
                                         None, save=False)
    animation_context.save(os.path.join(preview_path, f"{config.sampler}_at_epoch_{epoch}_generated_at_{execution_run_string}.gif"))



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
        print(f"Sampling timestep {i:3d}", end="\r")

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


# sample with context using standard algorithm
@torch.no_grad()
def sample_ddpm_context(config, device, model, number_of_samples, context, diffusion_params, save_rate=20):
    time_steps = config.time_steps
    # x_T ~ N(0, 1), sample initial noise
    samples = torch.randn(number_of_samples, 4, config.image_size, config.image_size).to(device)  

    # array to keep track of generated steps for plotting
    intermediate = [] 
    for i in range(time_steps, 0, -1):
        print(f"Sampling timestep {i:3d}", end="\r")

        # reshape time tensor
        t = torch.tensor([i / time_steps])[:, None, None, None].to(device)

        # sample some random noise to inject back in. For i = 1, don't add back in noise
        z = torch.randn_like(samples) if i > 1 else 0

        eps = model(samples, t, c=context)    # predict noise e_(x_t,t, ctx)
        samples = denoise_ddpm(samples, i, eps, diffusion_params, z)
        if i % save_rate==0 or i==time_steps or i<8:
            intermediate.append(samples.detach().cpu().numpy())

    intermediate = np.stack(intermediate)
    return samples, intermediate

# helper function; removes the predicted noise (but adds some noise back in to avoid collapse)
def denoise_ddpm(x, t, pred_noise, diffusion_params, z=None):
    if z is None:
        z = torch.randn_like(x)
    b_t = diffusion_params["b_t"]
    a_t = diffusion_params["a_t"]
    ab_t = diffusion_params["ab_t"]
    noise = b_t.sqrt()[t] * z
    mean = (x - pred_noise * ((1 - a_t[t]) / (1 - ab_t[t]).sqrt())) / a_t[t].sqrt()
    return mean + noise


if __name__ == "__main__":
    # logging configuration
    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(message)s",
        level=logging.DEBUG,
        datefmt="%Y-%m-%d %H:%M:%S")
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)

    # getting the options and basic setup
    config, parser = OptionParser().parse(sys.argv[1:], True)
    torch.manual_seed(config.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else torch.device("cpu"))

    sample(config, device)