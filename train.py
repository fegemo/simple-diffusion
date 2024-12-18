import logging
import time

from tqdm import tqdm
import torch.nn.functional as F

from io_utils import seconds_to_human_readable
from networks import ContextUnet

from diffusion_utilities import *


def train(config, device, dataloader):
    # sets up the model
    channels = config.input_channels
    base_channels = config.base_channels
    number_of_classes = config.classes
    image_size = config.image_size

    model = ContextUnet(channels, base_channels, number_of_classes, image_size).to(device)
    if config.verbose:
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logging.debug(f"Constructed model with trainable parameters: {total_params:,}")

    # sets up training configuration
    # construct DDPM noise schedule
    beta1 = config.beta_1
    beta2 = config.beta_2
    time_steps = config.time_steps

    b_t = (beta2 - beta1) * torch.linspace(0, 1, time_steps + 1, device=device) + beta1
    a_t = 1 - b_t
    ab_t = torch.cumsum(a_t.log(), dim=0).exp()
    ab_t[0] = 1

    optim = torch.optim.Adam(model.parameters(), lr=config.lr)

    # helper function: perturbs an image to a specified noise level
    def perturb_input(x, t, noise):
        return ab_t.sqrt()[t, None, None, None] * x + (1 - ab_t[t, None, None, None]) * noise

    # starts training
    checkpoints_path = os.path.join(config.log_folder, config.run_string, "checkpoints")
    epochs = config.epochs

    model.train()

    logging.info("Started training.")
    initial_time = time.time()
    for ep in (pbar_epochs := tqdm(range(epochs))):
        pbar_epochs.set_description(f"Training Epochs")

        # linearly decay learning rate
        optim.param_groups[0]["lr"] = config.lr * (1 - ep / epochs)

        for x, c in (pbar_batches := tqdm(dataloader, mininterval=2, leave=False)):
            pbar_batches.set_description(f"Minibatches")
            optim.zero_grad()
            x = x.to(device)
            c = c.to(x)

            # randomly mask out c
            context_mask = torch.bernoulli(torch.zeros(c.shape[0]) + 0.9).to(device)
            c = c * context_mask.unsqueeze(-1)

            # perturb data
            noise = torch.randn_like(x)
            t = torch.randint(1, time_steps + 1, (x.shape[0],)).to(device)
            x_pert = perturb_input(x, t, noise)

            # use network to recover noise
            pred_noise = model(x_pert, t / time_steps, c=c)

            # loss is mean squared error between the predicted and true noise
            loss = F.mse_loss(pred_noise, noise)
            loss.backward()

            optim.step()

        # save model periodically
        is_every_nth_epoch = ep % 4 == 0
        is_last_epoch = ep == int(epochs - 1)
        if is_every_nth_epoch or is_last_epoch:
            os.makedirs(checkpoints_path, exist_ok=True)
            torch.save(model.state_dict(), checkpoints_path + f"context_model_{ep}.pth")
            logging.info(f"Saved model at {checkpoints_path} context_model_{ep}.pth")

    elapsed_time = time.time() - initial_time
    logging.info(f"Finished training. Took {seconds_to_human_readable(elapsed_time)}.")

    diffusion_params = {
        b_t: b_t,
        a_t: a_t,
        ab_t: ab_t
    }
    return model, diffusion_params

#
# def __init__(config, device, dataloader):
#     return train(config, device, dataloader)
