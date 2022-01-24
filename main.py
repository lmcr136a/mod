import sys
import argparse
import warnings
warnings.filterwarnings('ignore')

from utils.utils import configuration, show_test_acc
from dataset import get_data_loader
from models.model import get_network
from run import run


def main(args):
    cfg, writer = configuration(args.config)

    dataloader, n_class = get_data_loader(cfg["data"])

    network = get_network(cfg["network"], n_class)

    result = run(dataloader, network, cfg["run"], writer, n_class)

    show_test_acc(result)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="config")
    parser.add_argument(
        "--config",
        "-c",
        nargs="?",
        type=str,
        default="configs/lasso.yml",
        help="Configuration file to use, .yml format",
    )
    args = parser.parse_args()
    main(args)