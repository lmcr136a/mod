import sys
import argparse
import warnings
warnings.filterwarnings('ignore')

from utils.utils import configuration, show_test_acc
from run import run


def main(cfg_dir, force_gpu=None):
    cfg, writer = configuration(cfg_dir, force_gpu)

    run(cfg, writer)
    
    sys.stdout.close()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="config")
    parser.add_argument(
        "--config",
        "-c",
        nargs="?",
        type=str,
        help="Configuration file to use, .yml format",
    )
    parser.add_argument(
        "--configs",
        "-cs",
        nargs='+',
        type=str,
        help="Configuration file to use, .yml format",
    )
    parser.add_argument(
        "--force_gpu",
        "-g",
        nargs="?",
        type=int,
        help="gpu",
    )

    args = parser.parse_args()
    if args.configs:
        for i, cfg in enumerate(args.configs):
            print(f"({i}/{len(args.configs)})   {cfg}, \nLEFT: {args.configs[i:]}")
            main(cfg, args.force_gpu)
    else:
        main(args.config)
    