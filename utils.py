import time
import datetime
import os
import sys
import shutil
import yaml
from torch.utils.tensorboard import SummaryWriter

def configuration(config):
    """
    input: path of .yml file
    output: dictionary of configurations
    """
    print("\n\n\n                               START LOGGING\n\n")
    tb_dir = f'tensorboards/{time.strftime("%m%d_%H%M%S", time.localtime())}_{config.split("/")[1].split(".")[0]}'
    writer = SummaryWriter(tb_dir)
    f = open(tb_dir+"/log.txt", 'w')
    sys.stdout = f
    with open(config, encoding="utf-8") as fp:
        cfg = yaml.load(fp, Loader=yaml.FullLoader)
    cfg["network"].update({"logdir": tb_dir})
    
    with open(tb_dir+"/"+config.split("/")[1], "w") as f:
        yaml.dump(cfg, f)
    return cfg, writer

def show_test_acc(result):
    print('\n==========================================================')
    print('#                                                          #')
    print('#                  Test acc : {:.4f}                  #'.format(result))
    print('#                                                          #')
    print('==========================================================\n')