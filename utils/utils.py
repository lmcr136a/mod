import time
import datetime
import os
import sys
import torch
import shutil
import yaml
from torch.utils.tensorboard import SummaryWriter
from thop import profile

def configuration(config):
    """
    input: path of .yml file
    output: dictionary of configurations
    """
    print("\n\n\n                               START LOGGING\n\n")
    tb_dir = f'experiments/{time.strftime("%m%d_%H%M%S", time.localtime())}_{config.split("/")[1].split(".")[0]}'
    writer = SummaryWriter(tb_dir)
    f = open(tb_dir+"/log.txt", 'w')
    sys.stdout = f
    with open(config, encoding="utf-8") as fp:
        cfg = yaml.load(fp, Loader=yaml.FullLoader)
        
    with open(tb_dir+"/"+config.split("/")[1], "w") as f:
        yaml.dump(cfg, f)

    return cfg, writer

def show_test_acc(result):
    print('\n==========================================================')
    print('#                                                          #')
    print('#                  Test acc : {:.4f}                  #'.format(result))
    print('#                                                          #')
    print('==========================================================\n')


def show_profile(network):
    input_image = torch.randn(1, 3, 32, 32).cuda()
    flops, params = profile(network, inputs=(input_image,))
    print('Params: %.2f' % (params))
    print('Flops: %.2f' % (flops))