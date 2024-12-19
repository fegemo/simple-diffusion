import logging
import sys
import os

import torch
from torch.utils.data import DataLoader

from configuration import OptionParser
from diffusion_utilities import CustomDataset, transform
from train import train
from sampling import sample


def train_and_sample():
    # logging configuration
    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(message)s",
        level=logging.DEBUG,
        datefmt="%Y-%m-%d %H:%M:%S")
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)

    # getting the options and basic setup
    config, parser = OptionParser().parse(sys.argv[1:], True)
    logging.info(f"Running with options: {parser.get_description(', ', ':')}")
    parser.save_configuration(os.path.join(config.log_folder, config.run_string), sys.argv)

    if config.verbose:
        logging.debug(f"Torch version: {torch.__version__}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else torch.device("cpu"))
    if torch.cuda.is_available():
        logging.info("Default GPU: {}".format(torch.cuda.get_device_name()))

    torch.manual_seed(config.seed)
    if config.verbose:
        logging.debug("SEED set to: ", config.seed)

    # loads the dataset
    dataset = CustomDataset(config.images, config.labels, transform, null_context=False)
    dataloader = DataLoader(dataset, batch_size=config.batch, shuffle=True, num_workers=1)

    # starts training
    model, diffusion_params = train(config, device, dataloader)

    # generates images with the trained model
    sample(config, device, model, diffusion_params)


if __name__ == '__main__':
    train_and_sample()
