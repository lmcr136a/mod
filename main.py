import sys
import argparse
import warnings
warnings.filterwarnings('ignore')

from utils.utils import configuration, show_test_acc
from run import run


def main(args):
    cfg, writer = configuration(args.config)

    run(cfg, writer)



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