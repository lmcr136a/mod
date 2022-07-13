import sys
import argparse
import warnings
warnings.filterwarnings('ignore')

from utils.utils import configuration, show_test_acc, write_result
from run import run
import time
import torch

def main(cfg_dir, force_gpu=None):
    cfg, writer = configuration(cfg_dir, force_gpu)

    run(cfg, writer)
    
    sys.stdout = sys.__stdout__



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

    try:
        torch.multiprocessing.set_start_method('spawn')
    except RuntimeError:
        print("Runtime ERROR")
        pass

    args = parser.parse_args()
    if args.configs:
        for i, cfg in enumerate(args.configs):
            start = time.time()
            print(f"\n({i}/{len(args.configs)})   {cfg}, \nLEFT: {args.configs[i:]}")
            main(cfg, args.force_gpu)
            print(f"Total running time: {round((time.time() - start)/3600, 3)} hour")
    else:
        main(args.config)
    