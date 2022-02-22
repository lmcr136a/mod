from gettext import npgettext
from operator import mod
import os
from pyexpat import model
import torch
import numpy as np
from utils.utils import progress_bar

from models.model import get_network


def rank_generation(model_name, network, device, test_loader, criterion):
    
    def get_feature_hook(self, input, output):
        global feature_result
        global entropy
        global total
        a = output.shape[0]
        b = output.shape[1]
        c = torch.tensor([torch.matrix_rank(output[i,j,:,:]).item() for i in range(a) for j in range(b)])

        c = c.view(a, -1).float()
        c = c.sum(0)
        feature_result = feature_result * total + c
        total = total + a
        feature_result = feature_result / total


    def test():
        global best_acc
        network.eval()
        test_loss = 0
        correct = 0
        total = 0
        limit = args.limit

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(test_loader):
                if batch_idx >= limit:  # use the first 6 batches to estimate the rank.
                    break
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = network(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                progress_bar(batch_idx, limit, 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                    % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))#'''



    if model_name=='vgg_16_bn':

        relucfg = network.relucfg

        for i, cov_id in enumerate(relucfg):
            cov_layer = network.features[cov_id]
            handler = cov_layer.register_forward_hook(get_feature_hook)
            test()
            handler.remove()

            if not os.path.isdir('rank_conv/'+model_name+'_limit%d'%(args.limit)):
                os.mkdir('rank_conv/'+model_name+'_limit%d'%(args.limit))
            np.save('rank_conv/'+model_name+'_limit%d'%(args.limit)+'/rank_conv' + str(i + 1) + '.npy', feature_result.numpy())

            feature_result = torch.tensor(0.)
            total = torch.tensor(0.)

    elif model_name=='resnet_56':

        cov_layer = eval('network.relu')
        handler = cov_layer.register_forward_hook(get_feature_hook)
        test()
        handler.remove()

        if not os.path.isdir('rank_conv/' + model_name+'_limit%d'%(args.limit)):
            os.mkdir('rank_conv/' + model_name+'_limit%d'%(args.limit))
        np.save('rank_conv/' + model_name+'_limit%d'%(args.limit)+ '/rank_conv%d' % (1) + '.npy', feature_result.numpy())
        feature_result = torch.tensor(0.)
        total = torch.tensor(0.)

        # ResNet56 per block
        cnt=1
        for i in range(3):
            block = eval('network.layer%d' % (i + 1))
            for j in range(9):
                cov_layer = block[j].relu1
                handler = cov_layer.register_forward_hook(get_feature_hook)
                test()
                handler.remove()
                np.save('rank_conv/' + model_name +'_limit%d'%(args.limit)+ '/rank_conv%d'%(cnt + 1)+'.npy', feature_result.numpy())
                cnt+=1
                feature_result = torch.tensor(0.)
                total = torch.tensor(0.)

                cov_layer = block[j].relu2
                handler = cov_layer.register_forward_hook(get_feature_hook)
                test()
                handler.remove()
                np.save('rank_conv/' + model_name +'_limit%d'%(args.limit)+ '/rank_conv%d'%(cnt + 1)+'.npy', feature_result.numpy())
                cnt += 1
                feature_result = torch.tensor(0.)
                total = torch.tensor(0.)


    elif model_name=='resnet_110':

        cov_layer = eval('network.relu')
        handler = cov_layer.register_forward_hook(get_feature_hook)
        test()
        handler.remove()

        if not os.path.isdir('rank_conv/'+model_name+'_limit%d'%(args.limit)):
            os.mkdir('rank_conv/'+model_name+'_limit%d'%(args.limit))
        np.save('rank_conv/'+model_name+'_limit%d'%(args.limit)+'/rank_conv' + str(i + 1) + '.npy', feature_result.numpy())
        feature_result = torch.tensor(0.)
        total = torch.tensor(0.)

        cnt = 1
        # ResNet110 per block
        for i in range(3):
            block = eval('network.layer%d' % (i + 1))
            for j in range(18):
                cov_layer = block[j].relu1
                handler = cov_layer.register_forward_hook(get_feature_hook)
                test()
                handler.remove()
                np.save('rank_conv/' + model_name  + '_limit%d' % (args.limit) + '/rank_conv%d' % (
                cnt + 1) + '.npy', feature_result.numpy())
                cnt += 1
                feature_result = torch.tensor(0.)
                total = torch.tensor(0.)

                cov_layer = block[j].relu2
                handler = cov_layer.register_forward_hook(get_feature_hook)
                test()
                handler.remove()
                np.save('rank_conv/' + model_name  + '_limit%d' % (args.limit) + '/rank_conv%d' % (
                    cnt + 1) + '.npy', feature_result.numpy())
                cnt += 1
                feature_result = torch.tensor(0.)
                total = torch.tensor(0.)

    elif model_name=='resnet_50':

        cov_layer = eval('network.maxpool')
        handler = cov_layer.register_forward_hook(get_feature_hook)
        test()
        handler.remove()

        if not os.path.isdir('rank_conv/'+model_name+'_limit%d'%(args.limit)):
            os.mkdir('rank_conv/'+model_name+'_limit%d'%(args.limit))
        np.save('rank_conv/'+model_name+'_limit%d'%(args.limit)+'/rank_conv' + str(i + 1) + '.npy', feature_result.numpy())
        feature_result = torch.tensor(0.)
        total = torch.tensor(0.)

        # ResNet50 per bottleneck
        cnt=1
        for i in range(4):
            block = eval('network.layer%d' % (i + 1))
            for j in range(network.num_blocks[i]):
                cov_layer = block[j].relu1
                handler = cov_layer.register_forward_hook(get_feature_hook)
                test()
                handler.remove()
                np.save('rank_conv/' + model_name+'_limit%d'%(args.limit) + '/rank_conv%d'%(cnt+1)+'.npy', feature_result.numpy())
                cnt+=1
                feature_result = torch.tensor(0.)
                total = torch.tensor(0.)

                cov_layer = block[j].relu2
                handler = cov_layer.register_forward_hook(get_feature_hook)
                test()
                handler.remove()
                np.save('rank_conv/' + model_name + '_limit%d' % (args.limit) + '/rank_conv%d' % (cnt + 1) + '.npy',
                        feature_result.numpy())
                cnt += 1
                feature_result = torch.tensor(0.)
                total = torch.tensor(0.)

                cov_layer = block[j].relu3
                handler = cov_layer.register_forward_hook(get_feature_hook)
                test()
                handler.remove()
                if j==0:
                    np.save('rank_conv/' + model_name + '_limit%d' % (args.limit) + '/rank_conv%d' % (cnt + 1) + '.npy',
                            feature_result.numpy())#shortcut conv
                    cnt += 1
                np.save('rank_conv/' + model_name + '_limit%d' % (args.limit) + '/rank_conv%d' % (cnt + 1) + '.npy',
                        feature_result.numpy())#conv3
                cnt += 1
                feature_result = torch.tensor(0.)
                total = torch.tensor(0.)


def hrank_main(cfg_network, n_class):
    model_name = cfg_network["model"]
    if model_name == "resnet56":
        compress_rate = "[0.1]+[0.60]*35+[0.0]*2+[0.6]*6+[0.4]*3+[0.1]+[0.4]+[0.1]+[0.4]+[0.1]+[0.4]+[0.1]+[0.4]"
    elif model_name == "resnet110":
        compress_rate = "[0.1]+[0.40]*36+[0.40]*36+[0.4]*36"
    elif model_name == "resnet50":
        compress_rate = "[0.2]+[0.8]*10+[0.8]*13+[0.55]*19+[0.45]*10"
    elif model_name == "vgg16_bn":
        compress_rate = "[0.95]+[0.5]*6+[0.9]*4+[0.8]*2"
    else:
        assert 1==0

    import re
    cprate_str=compress_rate
    cprate_str_list=cprate_str.split('+')
    pat_cprate = re.compile(r'\d+\.\d*')
    pat_num = re.compile(r'\*\d+')
    cprate=[]
    for x in cprate_str_list:
        num=1
        find_num=re.findall(pat_num, x)
        if find_num:
            assert len(find_num) == 1
            num=int(find_num[0].replace('*',''))
        find_cprate = re.findall(pat_cprate, x)
        assert len(find_cprate)==1
        cprate+=[float(find_cprate[0])]*num

    compress_rate=cprate

    print('==> Building model..')
    model = get_network(cfg_network, n_class, sparsity=compress_rate).cuda()
    return model

