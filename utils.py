import datetime
import os
import shutil
import yaml

def configuration(config):
    """
    input: path of .yml file
    output: dictionary of configurations
    """

    with open(config, encoding="utf-8") as fp:
        cfg = yaml.load(fp, Loader=yaml.FullLoader)

    return cfg

def show_test_acc(result):
    print('\n==========================================================')
    print('#                                                          #')
    print('#                  Test acc : {:.4f}                  #'.format(result))
    print('#                                                          #')
    print('==========================================================\n')