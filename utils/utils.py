import time

import os
import sys
import torch
import shutil
import yaml
from torch.utils.tensorboard import SummaryWriter
from thop import profile
import datetime
from pytz import timezone, utc


def num_(number):
    if len(str(number)) == 1:
        return f"0{number}"
    else:
        return f"{number}"


def configuration(config, force_gpu, jupyter=False):
    """
    input: path of .yml file
    output: dictionary of configurations
    """

    with open("configs/"+config, encoding="utf-8") as fp:
        cfg = yaml.load(fp, Loader=yaml.FullLoader)        

    if force_gpu is not None:
        cfg['run']['gpu_device']=force_gpu # 하지만 안쓰임
        os.environ["CUDA_VISIBLE_DEVICES"]= str(force_gpu)
    print(f"\n{'*'*10}START LOGGING, GPU: {cfg['run']['gpu_device']} {'*'*10}\n")
    KST = timezone('Asia/Seoul')
    now = datetime.datetime.utcnow()
    t = utc.localize(now).astimezone(KST)
    # tb_dir = f'experiments/___{config.split(".")[0]}_{cfg["network"]["model"]}_{num_(t.month)}{num_(t.day)}_{num_(t.hour)}{num_(t.minute)}{num_(t.second)}'
    tb_dir = f'experiments/{config.split(".")[0]}' #_{num_(t.month)}{num_(t.day)}_{num_(t.hour)}{num_(t.minute)}{num_(t.second)}'
    writer = SummaryWriter(tb_dir)
    if not jupyter:
        f = open(tb_dir+"/log.txt", 'w')
        sys.stdout = f

        with open(tb_dir+"/"+config, "w") as f:
            yaml.dump(cfg, f)

    return cfg, writer

def write_result(log_dir):
    with open(log_dir+"/log.txt", encoding="utf-8") as fp:
        result_log = fp.read()
    KST = timezone('Asia/Seoul')
    now = datetime.datetime.utcnow()
    t = utc.localize(now).astimezone(KST)
    time_info = f'{num_(t.month)}{num_(t.day)}_{num_(t.hour)}{num_(t.minute)}{num_(t.second)}'

    result_log = result_log[-160:-140]
    result_log = log_dir.split("/")[-1] + f"   {time_info}:     " + result_log + "\n\n"

    with open("result_summary.txt", 'a') as rf:
        rf.write(result_log)
        


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

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f



def progress_bar(current, total, msg=None):
    
    _, term_width = os.popen('stty size', 'r').read().split()
    term_width = int(term_width)

    TOTAL_BAR_LENGTH = 65.
    last_time = time.time()
    begin_time = last_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()