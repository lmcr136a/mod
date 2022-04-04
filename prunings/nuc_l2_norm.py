import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import time

from thop import profile
from collections import OrderedDict
from models.model import get_network
import chip.utils as utils

conv_index = torch.tensor(1)

def cal_fm(model, model_name, train_loader, log_dir, device="cuda", repeat=5):

    def get_feature_hook(self, input, output):
        global conv_index

        if not os.path.isdir(log_dir +'/conv_feature_map'):
            os.makedirs(log_dir +'/conv_feature_map')
        np.save(log_dir +'/conv_feature_map' + '/conv_feature_map_'+ str(conv_index) + '.npy',
                output.cpu().numpy())
        conv_index += 1

    def inference():
        model.eval()
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                #use 5 batches to get feature maps.
                if batch_idx >= repeat:
                    break

                inputs, targets = inputs.to(device), targets.to(device)

                model(inputs)

    if model_name=='vgg_16_bn':

        # if len(gpu) > 1:
        #     relucfg = model.module.relucfg
        # else:
        relucfg = model.relucfg
        start = time.time()
        for i, cov_id in enumerate(relucfg):
            cov_layer = model.features[cov_id]
            handler = cov_layer.register_forward_hook(get_feature_hook)
            inference()
            handler.remove()

    elif model_name=='resnet56':

        cov_layer = eval('model.relu')
        handler = cov_layer.register_forward_hook(get_feature_hook)
        inference()
        handler.remove()

        # ResNet56 per block
        cnt=1
        for i in range(3):
            block = eval('model.layer%d' % (i + 1))
            for j in range(9):
                cov_layer = block[j].relu1
                handler = cov_layer.register_forward_hook(get_feature_hook)
                inference()
                handler.remove()
                cnt+=1

                cov_layer = block[j].relu2
                handler = cov_layer.register_forward_hook(get_feature_hook)
                inference()
                handler.remove()
                cnt += 1

    elif model_name=='resnet110':

        cov_layer = eval('model.relu')
        handler = cov_layer.register_forward_hook(get_feature_hook)
        inference()
        handler.remove()

        cnt = 1
        # ResNet110 per block
        for i in range(3):
            block = eval('model.layer%d' % (i + 1))
            for j in range(18):
                cov_layer = block[j].relu1
                handler = cov_layer.register_forward_hook(get_feature_hook)
                inference()
                handler.remove()
                cnt += 1
                feature_result = torch.tensor(0.)
                total = torch.tensor(0.)

                cov_layer = block[j].relu2
                handler = cov_layer.register_forward_hook(get_feature_hook)
                inference()
                handler.remove()
                cnt += 1

    elif model_name in ['resnet50', "resnet34"]:
        # cov_layer = eval('model.maxpool')
        # handler = cov_layer.register_forward_hook(get_feature_hook)
        # inference()
        # handler.remove()

        # ResNet50 per bottleneck
        for i in range(4):
            block = eval('model.layer%d' % (i + 1))
            for j in range(model.num_blocks[i]):
                cov_layer = block[j].relu1
                handler = cov_layer.register_forward_hook(get_feature_hook)
                inference()
                handler.remove()

                cov_layer = block[j].relu2
                handler = cov_layer.register_forward_hook(get_feature_hook)
                inference()
                handler.remove()

                if model_name == "resnet50":
                    cov_layer = block[j].relu3
                    handler = cov_layer.register_forward_hook(get_feature_hook)
                    inference()
                    handler.remove()

                    if j==0:
                        cov_layer = block[j].relu3
                        handler = cov_layer.register_forward_hook(get_feature_hook)
                        inference()
                        handler.remove()


def calculate_nucl2(model_name, log_dir, norm_p, fm_path=None, repeat=5):
    def reduced_1_row_norm(inputs, row_index, data_index):
        inputs[data_index, row_index, :] = torch.zeros(inputs.shape[-1])
        u, s, v = torch.svd(inputs[data_index, :, :])
        m = torch.norm(s, norm_p).item()
        # m = torch.norm(inputs[data_index, :, :], p = 'nuc').item()
        return m

    def ci_score(path_conv):
        conv_output = torch.tensor(np.round(np.load(path_conv), 4))
        conv_reshape = conv_output.reshape(conv_output.shape[0], conv_output.shape[1], -1)

        r1_norm = torch.zeros([conv_reshape.shape[0], conv_reshape.shape[1]])
        for i in range(conv_reshape.shape[0]):
            for j in range(conv_reshape.shape[1]):
                r1_norm[i, j] = reduced_1_row_norm(conv_reshape.clone(), j, data_index = i)

        ci = np.zeros_like(r1_norm)

        for i in range(r1_norm.shape[0]):
            original_norm = torch.norm(torch.tensor(conv_reshape[i, :, :]), p='nuc').item()
            ci[i] = original_norm - r1_norm[i]

        # return shape: [batch_size, filter_number]
        return ci

    def mean_repeat_ci(repeat, num_layers):
        layer_ci_mean_total = []
        for j in range(num_layers):
            repeat_ci_mean = []
            for i in range(repeat):
                index = j * repeat + i + 1
                # add
                if fm_path:
                    path_conv = fm_path+"/conv_feature_map/conv_feature_map_tensor({1}).npy".format(model_name, str(index))
                else:
                    path_conv = log_dir+"/conv_feature_map/conv_feature_map_tensor({1}).npy".format(model_name, str(index))
                batch_ci = ci_score(path_conv)
                single_repeat_ci_mean = np.mean(batch_ci, axis=0)
                repeat_ci_mean.append(single_repeat_ci_mean)

            layer_ci_mean = np.mean(repeat_ci_mean, axis=0)
            layer_ci_mean_total.append(layer_ci_mean)

        return np.array(layer_ci_mean_total)

    if model_name == "resnet56":
        num_layers = 55
    elif model_name == "resnet50":
        num_layers = 53
    elif model_name == "resnet34":
        num_layers = 32

    save_path = log_dir + '/CI_' + model_name

    ci = mean_repeat_ci(repeat, num_layers)

    if model_name == 'resnet50':
        num_layers = 53
    elif model_name == "resnet34":
        num_layers = 32
    for i in range(num_layers):
        print(i)
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        np.save(save_path + "/ci_conv{0}.npy".format(str(i + 1)), ci[i])
    return save_path


def nucl2_prune(cfg_network, pretrain_dir, ci_dir, n_class):
    name_base = ""
    def load_vgg_model(model, oristate_dict):
        state_dict = model.state_dict()
        last_select_index = None #Conv index selected in the previous layer

        cnt=0
        prefix = ci_dir+'/ci_conv'
        subfix = ".npy"
        for name, module in model.named_modules():
            name = name.replace('module.', '')

            if isinstance(module, nn.Conv2d):

                cnt+=1
                oriweight = oristate_dict[name + '.weight']
                curweight = state_dict[name_base+name + '.weight']
                orifilter_num = oriweight.size(0)
                currentfilter_num = curweight.size(0)

                if orifilter_num != currentfilter_num:

                    cov_id = cnt
                    print('loading ci from: ' + prefix + str(cov_id) + subfix)
                    ci = np.load(prefix + str(cov_id) + subfix)
                    select_index = np.argsort(ci)[orifilter_num-currentfilter_num:]  # preserved filter id
                    select_index.sort()

                    if last_select_index is not None:
                        for index_i, i in enumerate(select_index):
                            for index_j, j in enumerate(last_select_index):
                                state_dict[name_base+name + '.weight'][index_i][index_j] = \
                                    oristate_dict[name + '.weight'][i][j]
                    else:
                        for index_i, i in enumerate(select_index):
                            state_dict[name_base+name + '.weight'][index_i] = \
                                    oristate_dict[name + '.weight'][i]

                    last_select_index = select_index

                elif last_select_index is not None:
                    for i in range(orifilter_num):
                        for index_j, j in enumerate(last_select_index):
                            state_dict[name_base+name + '.weight'][i][index_j] = \
                                oristate_dict[name + '.weight'][i][j]
                else:
                    state_dict[name_base+name + '.weight'] = oriweight
                    last_select_index = None

        model.load_state_dict(state_dict)

    def load_resnet_model(model, oristate_dict, layer):
        cfg = {
            34: [3, 3, 3, 3],
            50: [9, 9, 9],
            56: [9, 9, 9],
            110: [18, 18, 18],
        }


        state_dict = model.state_dict()

        current_cfg = cfg[layer]
        last_select_index = None

        all_conv_weight = []

        prefix = ci_dir+'/ci_conv'
        subfix = ".npy"

        cnt=1
        for layer, num in enumerate(current_cfg):
            layer_name = 'layer' + str(layer + 1) + '.'
            for k in range(num):
                for l in range(2):

                    cnt+=1
                    cov_id=cnt

                    conv_name = layer_name + str(k) + '.conv' + str(l + 1)
                    conv_weight_name = conv_name + '.weight'
                    all_conv_weight.append(conv_weight_name)
                    oriweight = oristate_dict[conv_weight_name]
                    curweight =state_dict[name_base+conv_weight_name]
                    orifilter_num = oriweight.size(0)
                    currentfilter_num = curweight.size(0)

                    if orifilter_num != currentfilter_num:
                        print('loading ci from: ' + prefix + str(cov_id) + subfix)
                        ci = np.load(prefix + str(cov_id) + subfix)
                        select_index = np.argsort(ci)[orifilter_num - currentfilter_num:]  # preserved filter id
                        select_index.sort()

                        if last_select_index is not None:
                            for index_i, i in enumerate(select_index):
                                for index_j, j in enumerate(last_select_index):
                                    state_dict[name_base+conv_weight_name][index_i][index_j] = \
                                        oristate_dict[conv_weight_name][i][j]
                        else:
                            for index_i, i in enumerate(select_index):
                                state_dict[name_base+conv_weight_name][index_i] = \
                                    oristate_dict[conv_weight_name][i]

                        last_select_index = select_index

                    elif last_select_index is not None:
                        for index_i in range(orifilter_num):
                            for index_j, j in enumerate(last_select_index):
                                state_dict[name_base+conv_weight_name][index_i][index_j] = \
                                    oristate_dict[conv_weight_name][index_i][j]
                        last_select_index = None

                    else:
                        state_dict[name_base+conv_weight_name] = oriweight
                        last_select_index = None

        for name, module in model.named_modules():
            name = name.replace('module.', '')

            if isinstance(module, nn.Conv2d):
                conv_name = name + '.weight'
                if 'shortcut' in name:
                    continue
                if conv_name not in all_conv_weight:
                    state_dict[name_base+conv_name] = oristate_dict[conv_name]

            elif isinstance(module, nn.Linear):
                state_dict[name_base+name + '.weight'] = oristate_dict[name + '.weight']
                state_dict[name_base+name + '.bias'] = oristate_dict[name + '.bias']

        model.load_state_dict(state_dict)

    import re
    cprate_str = "[0.]+[0.4]*2+[0.4]*9+[0.5]*9+[0.5]*9"
    cprate_str_list = cprate_str.split('+')
    pat_cprate = re.compile(r'\d+\.\d*')
    pat_num = re.compile(r'\*\d+')
    cprate = []
    for x in cprate_str_list:
        num = 1
        find_num = re.findall(pat_num, x)
        if find_num:
            assert len(find_num) == 1
            num = int(find_num[0].replace('*', ''))
        find_cprate = re.findall(pat_cprate, x)
        assert len(find_cprate) == 1
        cprate += [float(find_cprate[0])] * num

    sparsity = cprate

    # load model
    print('sparsity:' + str(sparsity))
    print('==> Building model..')
    model = get_network(cfg_network, n_class, sparsity=sparsity).cuda()

    #calculate model size
    input_image_size=32
    input_image = torch.randn(1, 3, input_image_size, input_image_size).cuda()
    flops, params = profile(model, inputs=(input_image,))
    print('Params: %.2f' % (params))
    print('Flops: %.2f' % (flops))


    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    criterion_smooth = utils.CrossEntropyLabelSmooth(n_class, 0)
    criterion_smooth = criterion_smooth.cuda()


    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.005)
    lr_decay_step = list(map(int, "50,100".split(',')))
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_decay_step, gamma=0.1)

    start_epoch = 0
    best_top1_acc= 0

    # load the checkpoint if it exists

    print('resuming from pretrain model')
    
    origin_model = get_network(cfg_network, n_class, sparsity=[0.] * 100).cuda()
    ckpt = torch.load(pretrain_dir, map_location='cuda:0')

    if cfg_network["model"] == 'resnet_110':
        new_state_dict = OrderedDict()
        for k, v in ckpt.items():
            new_state_dict[k.replace('module.', '')] = v
        origin_model.load_state_dict(new_state_dict)
    else:
        origin_model.load_state_dict(ckpt)

    oristate_dict = origin_model.state_dict()

    if cfg_network["model"] == 'vgg16_bn':
        load_vgg_model(model, oristate_dict)
    elif cfg_network["model"] == 'resnet56':
        load_resnet_model(model, oristate_dict, 56)
    elif cfg_network["model"] == 'resnet34':
        load_resnet_model(model, oristate_dict, 34)
    elif cfg_network["model"] == 'resnet110':
        load_resnet_model(model, oristate_dict, 110)
    else:
        print(cfg_network["model"])
        raise
    
    return model, optimizer, criterion


