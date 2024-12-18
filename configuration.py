import argparse
import os
import datetime
import sys
from math import ceil

SEED = 42

IMG_SIZE = 64
INPUT_CHANNELS = 4
OUTPUT_CHANNELS = 4

EPOCHS = 150
BATCH_SIZE = 100
LR = 1e-3
TIME_STEPS = 500
BETA_1 = 1e-4
BETA_2 = 0.02
BASE_CHANNELS = 64
NUMBER_OF_CLASSES = 5
LOG_FOLDER = "output"
DATASET_IMAGES_PATH = "./data/train-images-FRONT.npy"
DATASET_LABELS_PATH = "./data/train-classes-FRONT.npy"


class SingletonMeta(type):
    """
    The Singleton class can be implemented in different ways in Python. Some
    possible methods include: base class, decorator, metaclass. We will use the
    metaclass because it is best suited for this purpose.
    """

    _instances = {}

    def __call__(cls, *args, **kwargs):
        """
        Possible changes to the value of the `__init__` argument do not affect
        the returned instance.
        """
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]


class OptionParser(metaclass=SingletonMeta):
    def __init__(self) -> None:
        self.parser = argparse.ArgumentParser()
        self.initialized = False
        self.values = {}

    def initialize(self):
        self.parser.add_argument("--image-size", help="size of squared images", default=IMG_SIZE, type=int)
        self.parser.add_argument("--output-channels", help="size of squared images", default=OUTPUT_CHANNELS, type=int)
        self.parser.add_argument("--input-channels", help="size of squared images", default=INPUT_CHANNELS, type=int)
        self.parser.add_argument("--verbose", help="outputs verbosity information",
                                 default=False, action="store_true")

        self.parser.add_argument("--images", help="path to the dataset images",
                                 default=DATASET_IMAGES_PATH)
        self.parser.add_argument("--labels", help="path to the dataset labels",
                                 default=DATASET_LABELS_PATH)
        self.parser.add_argument("--base-channels", help="number of base channels for the model",
                                 default=BASE_CHANNELS, type=int)
        self.parser.add_argument("--classes", help="number of classes for the model",
                                 default=NUMBER_OF_CLASSES, type=int)

        self.parser.add_argument("--batch", type=int, help="the batch size", default=BATCH_SIZE)
        self.parser.add_argument("--lr", type=float, help="(initial) learning rate", default=LR)
        self.parser.add_argument("--epochs", type=int, help="number of epochs to train", default=EPOCHS)
        self.parser.add_argument("--evaluate-steps", type=int, help="number of generator update steps "
                                                                    "to wait until an evaluation is done", default=1000)
        self.parser.add_argument("--beta-1", type=float, help="initial beta value for the diffusion process",
                                 default=BETA_1)
        self.parser.add_argument("--beta-2", type=float, help="final beta value for the diffusion process",
                                 default=BETA_2)
        self.parser.add_argument("--time-steps", type=int, help="number of time steps for the "
                                                                "diffusion process", default=TIME_STEPS)

        self.parser.add_argument("--callback-evaluate-fid",
                                 help="every few update steps, evaluate with the FID metric the performance "
                                      "on the train and test sets",
                                 default=False, action="store_true")
        self.parser.add_argument("--callback-evaluate-l1",
                                 help="every few update steps, evaluate with the L1 metric the performance "
                                      "on the train and test sets",
                                 default=False, action="store_true")
        self.parser.add_argument("--save-model", help="saves the model at the end", default=False,
                                 action="store_true")
        self.parser.add_argument(
            "--log-folder", help="the folder in which the training procedure saves the logs", default=LOG_FOLDER)
        self.initialized = True

    def parse(self, args=None, return_parser=False):
        if args is None:
            args = sys.argv[1:]
        if not self.initialized:
            self.initialize()
        self.values = self.parser.parse_args(args)

        setattr(self.values, "seed", SEED)
        setattr(self.values, "run_string", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        setattr(self.values, "inner_channels", min(self.values.input_channels, self.values.output_channels))

        if return_parser:
            return self.values, self
        else:
            return self.values

    def get_description(self, param_separator=",", key_value_separator="-"):
        sorted_args = sorted(vars(self.values).items())
        description = param_separator.join(map(lambda p: f"{p[0]}{key_value_separator}{p[1]}", sorted_args))
        return description

    def save_configuration(self, folder_path, argv):
        os.makedirs(folder_path, exist_ok=True)
        with open(os.sep.join([folder_path, "configuration.txt"]), "w") as file:
            file.write(" ".join(argv) + "\n\n")
            file.write(self.get_description("\n", ": ") + "\n")


def in_notebook():
    try:
        from IPython import get_ipython
        if "IPKernelApp" not in get_ipython().config:  # pragma: no cover
            return False
    except ImportError:
        return False
    except AttributeError:
        return False
    return True
