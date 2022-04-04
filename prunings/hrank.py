import os
import pickle
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim

from utils.utils import progress_bar
from models.model import get_network
from utils.metrics import AverageMeter, accuracy

from utils.utils import show_test_acc, show_profile

feature_result = torch.tensor(0.)
total = torch.tensor(0.)
def rank_generation(model_name, network, device, dataloader, criterion, logdir, limit=5):
    global feature_result
    
    def get_feature_hook(self, input, output):
        global feature_result
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
        network.eval()
        test_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(dataloader.test_loader):
                if batch_idx >= limit:  # use the first 6 batches to estimate the rank.
                    break
                inputs, targets = inputs.to(device), targets.to(device).to(torch.long)
                outputs = network(inputs)
                targets = targets.to(torch.long)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                progress_bar(batch_idx, limit, 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                    % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))#'''


    dir_name = '/rank_conv'

    if model_name=='vgg_16_bn':

        relucfg = network.relucfg

        for i, cov_id in enumerate(relucfg):
            cov_layer = network.features[cov_id]
            handler = cov_layer.register_forward_hook(get_feature_hook)
            test()
            handler.remove()

            if not os.path.isdir(logdir+dir_name):
                os.mkdir(logdir+dir_name)
            np.save(logdir+dir_name+'/rank_conv' + str(i + 1) + '.npy', feature_result.numpy())

            feature_result = torch.tensor(0.)
            total = torch.tensor(0.)

    elif model_name=='resnet56':

        cov_layer = eval('network.relu')
        handler = cov_layer.register_forward_hook(get_feature_hook)
        test()
        handler.remove()

        if not os.path.isdir(logdir+dir_name):
            os.mkdir(logdir+dir_name)
        np.save(logdir+dir_name+ '/rank_conv1.npy', feature_result.numpy())
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
                np.save(logdir+dir_name + '/rank_conv%d'%(cnt + 1)+'.npy', feature_result.numpy())
                cnt+=1
                feature_result = torch.tensor(0.)
                total = torch.tensor(0.)

                cov_layer = block[j].relu2
                handler = cov_layer.register_forward_hook(get_feature_hook)
                test()
                handler.remove()
                np.save(logdir+dir_name + '/rank_conv%d'%(cnt + 1)+'.npy', feature_result.numpy())
                cnt += 1
                feature_result = torch.tensor(0.)
                total = torch.tensor(0.)


    elif model_name=='resnet110':

        cov_layer = eval('network.relu')
        handler = cov_layer.register_forward_hook(get_feature_hook)
        test()
        handler.remove()

        if not os.path.isdir(logdir+dir_name):
            os.mkdir(logdir+dir_name)
        np.save(logdir+dir_name+ '/rank_conv%d' % (1) + '.npy', feature_result.numpy())
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
                np.save(logdir+dir_name + '/rank_conv%d' % (
                cnt + 1) + '.npy', feature_result.numpy())
                cnt += 1
                feature_result = torch.tensor(0.)
                total = torch.tensor(0.)

                cov_layer = block[j].relu2
                handler = cov_layer.register_forward_hook(get_feature_hook)
                test()
                handler.remove()
                np.save(logdir+dir_name + '/rank_conv%d' % (
                    cnt + 1) + '.npy', feature_result.numpy())
                cnt += 1
                feature_result = torch.tensor(0.)
                total = torch.tensor(0.)

    elif model_name in ['resnet50', 'resnet34']:

        cov_layer = eval('network.maxpool')
        handler = cov_layer.register_forward_hook(get_feature_hook)
        test()
        handler.remove()

        if not os.path.isdir(logdir+dir_name):
            os.mkdir(logdir+dir_name)
        np.save(logdir+dir_name+ '/rank_conv%d' % (1) + '.npy', feature_result.numpy())
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
                np.save(logdir+dir_name + '/rank_conv%d'%(cnt+1)+'.npy', feature_result.numpy())
                cnt+=1
                feature_result = torch.tensor(0.)
                total = torch.tensor(0.)

                cov_layer = block[j].relu2
                handler = cov_layer.register_forward_hook(get_feature_hook)
                test()
                handler.remove()
                np.save(logdir+dir_name + '/rank_conv%d' % (cnt + 1) + '.npy',
                        feature_result.numpy())
                cnt += 1
                feature_result = torch.tensor(0.)
                total = torch.tensor(0.)

                if model_name == "resnet50":
                    cov_layer = block[j].relu3
                    handler = cov_layer.register_forward_hook(get_feature_hook)
                    test()
                    handler.remove()
                if j==0:
                    np.save(logdir+dir_name + '/rank_conv%d' % (cnt + 1) + '.npy',
                            feature_result.numpy())#shortcut conv
                    cnt += 1
                np.save(logdir+dir_name + '/rank_conv%d' % (cnt + 1) + '.npy',
                        feature_result.numpy())#conv3
                cnt += 1
                feature_result = torch.tensor(0.)
                total = torch.tensor(0.)
    return logdir+dir_name


def show_nonzero_params_num(model):
    num_param = 0
    for p in model.parameters():
        for channel in p.data:
            if p.requires_grad and channel.shape and not torch.all(channel == 0):
                num_param += channel.numel()
    return num_param

def hrank_main(cfg_network, n_class, log_dir, device, dataloader):
    resume_path = cfg_network["load_state"]
    rank_conv_path = cfg_network["pruning"]["hrank"]["rank_conv_path"]
    model_name = cfg_network["model"]

    if model_name == "resnet56":
        compress_rate = "[0.1]+[0.60]*35+[0.0]*2+[0.6]*6+[0.4]*3+[0.1]+[0.4]+[0.1]+[0.4]+[0.1]+[0.4]+[0.1]+[0.4]"
    elif model_name == "resnet110":
        compress_rate = "[0.1]+[0.40]*36+[0.40]*36+[0.4]*36"
    elif model_name == "resnet50":
        compress_rate = "[0.2]+[0.8]*10+[0.8]*13+[0.55]*19+[0.45]*10"
    elif model_name == "resnet34":
        compress_rate = "[0.2]+[0.8]*10+[0.8]*13+[0.55]*19+[0.45]*10"
    elif model_name == "vgg_16_bn":
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

    m = eval('mask_' + model_name)(model=model, compress_rate=model.compress_rate, job_dir=log_dir, device=device)

    convcfg = model.covcfg
    param_per_cov_dic={
        'vgg_16_bn': 4,
        'resnet50':3,
        'resnet34':3,
        'resnet56':3,
        'resnet110':3
    }

    
    num_params_before_pruning = show_nonzero_params_num(model)
    print("NUM OF PARAMS BEFORE PRUNING: ", num_params_before_pruning)

    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones="5,10", gamma=0.1)
    criterion = nn.CrossEntropyLoss()
    # Training
    def train(epoch, cov_id, optimizer, scheduler, pruning=True):
        model.train()

        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(dataloader.train_loader):
            with torch.cuda.device(device):
                inputs = inputs.to(device)
                targets = targets.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                targets = targets.to(torch.long)
                loss = criterion(outputs, targets)
                loss.backward()

                optimizer.step()

                if pruning:
                    m.grad_mask(cov_id)

                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                # if batch_idx % 50 == 0:
                #     print(batch_idx, ')   Cov: %d | Loss: %.3f | Acc: %.3f%% (%d/%d)'
                #                 % (cov_id, train_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    def test(epoch, cov_id, optimizer, scheduler):
        top1 = AverageMeter()
        top5 = AverageMeter()

        best_acc=0.
        model.eval()
        num_iterations = len(dataloader.test_loader)
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(dataloader.test_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                targets = targets.to(torch.long)
                loss = criterion(outputs, targets)

                prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
                top1.update(prec1.item(), inputs.size(0))
                top5.update(prec5.item(), inputs.size(0))

            print(
                'Epoch[{0}]({1}/{2}): '
                'Prec@1(1,5) {top1.avg:.2f}, {top5.avg:.2f}'.format(
                    epoch, batch_idx, num_iterations, top1=top1, top5=top5))

        if top1.avg > best_acc:
            print('Saving to '+model_name+'_cov'+str(cov_id)+'.pt')
            state = {
                'state_dict': model.state_dict(),
                'best_prec1': top1.avg,
                'epoch': epoch,
                'scheduler':scheduler.state_dict(),
                'optimizer': optimizer.state_dict() 
            }
            if not os.path.isdir(log_dir+'/pruned_checkpoint'):
                os.mkdir(log_dir+'/pruned_checkpoint')
            best_acc = top1.avg
            # print("----------------------------------------------------save ", cov_id)
            torch.save(state, log_dir+'/pruned_checkpoint/'+model_name+'_cov'+str(cov_id)+'.pt')

        print("=>Best accuracy {:.3f}".format(best_acc))

    test(0, 0, optimizer, scheduler)   ## CIFAR
    
    for cov_id in range(len(convcfg)):
        # Load pruned_checkpoint
        print("cov-id: %d ====> Resuming from pruned_checkpoint..." % (cov_id), "NUM OF PARAMS: ", show_nonzero_params_num(model))

        # print("----------------------------------------------------now ", cov_id)
        m.layer_mask(cov_id + 1,  saved_path=rank_conv_path, resume=None, param_per_cov=param_per_cov_dic[model_name], arch=model_name)

        if cov_id == 0:

            pruned_checkpoint = torch.load(resume_path, map_location='cuda:0')
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            if model_name == 'resnet_50':
                tmp_ckpt = pruned_checkpoint
            else:
                tmp_ckpt = pruned_checkpoint

            for k, v in tmp_ckpt.items():
                new_state_dict[k.replace('module.', '')] = v

            model.load_state_dict(new_state_dict) #'''
        else:
            if model_name in ['resnet50', 'resnet34']:
                if model_name == 'resnet50':
                    skip_list=[1,5,8,11,15,18,21,24,28,31,34,37,40,43,47,50,53]
                elif model_name == 'resnet34':
                    skip_list=[1,4,6,8,11,13,15,17,20,22,24,26,28,30,33,35,37]
                    
                if cov_id+1 not in skip_list:
                    continue
                else:
                    pruned_checkpoint = torch.load(
                        log_dir + "/pruned_checkpoint/" + model_name + "_cov" + str(skip_list[skip_list.index(cov_id+1)-1]) + '.pt', map_location='cuda:0')
                    print(model.fc.weight.data.shape)
                    model.load_state_dict(pruned_checkpoint['state_dict'])
            else:
                # print("----------------------------------------------------load ",cov_id)
                pruned_checkpoint = torch.load(log_dir + "/pruned_checkpoint/" + model_name + "_cov" + str(cov_id) + '.pt', map_location='cuda:0')
                model.load_state_dict(pruned_checkpoint['state_dict'])

        for epoch in range(0, 3):
            train(epoch, cov_id + 1, optimizer, scheduler)
            scheduler.step()
            test(epoch, cov_id + 1, optimizer, scheduler)

    num_params_after_pruning = show_nonzero_params_num(model)
    print("NUM OF PARAMS AFTER PRUNING: ", num_params_after_pruning, f"{(num_params_before_pruning - num_params_after_pruning)*100/num_params_before_pruning}% pruned")
    return model


class mask_vgg_16_bn:
    def __init__(self, model=None, compress_rate=[0.50], job_dir='',device=None):
        self.model = model
        self.compress_rate = compress_rate
        self.mask = {}
        self.job_dir=job_dir
        self.device = device

    def layer_mask(self, cov_id, saved_path, resume=None, param_per_cov=4,  arch="vgg_16_bn"):
        params = self.model.parameters()
        prefix = saved_path+"/rank_conv"
        subfix = ".npy"

        if resume:
            with open(resume, 'rb') as f:
                self.mask = pickle.load(f)
        else:
            resume=self.job_dir+'/mask'

        self.param_per_cov=param_per_cov

        for index, item in enumerate(params):

            if index == cov_id * param_per_cov:
                break
            if index == (cov_id - 1) * param_per_cov:
                f, c, w, h = item.size()
                rank = np.load(prefix + str(cov_id) + subfix)
                pruned_num = int(self.compress_rate[cov_id - 1] * f)
                ind = np.argsort(rank)[pruned_num:]  # preserved filter id

                zeros = torch.zeros(f, 1, 1, 1).to(self.device)
                for i in range(len(ind)):
                    zeros[ind[i], 0, 0, 0] = 1.
                self.mask[index] = zeros  # covolutional weight
                item.data = item.data * self.mask[index]

            if index > (cov_id - 1) * param_per_cov and index <= (cov_id - 1) * param_per_cov + param_per_cov-1:
                self.mask[index] = torch.squeeze(zeros)
                item.data = item.data * self.mask[index]

        with open(resume, "wb") as f:
            pickle.dump(self.mask, f)

    def grad_mask(self, cov_id):
        params = self.model.parameters()
        for index, item in enumerate(params):
            if index == cov_id * self.param_per_cov:
                break
            item.data = item.data * self.mask[index]#prune certain weight


class mask_resnet56:
    def __init__(self, model=None, compress_rate=[0.50], job_dir='',device=None):
        self.model = model
        self.compress_rate = compress_rate
        self.mask = {}
        self.job_dir=job_dir
        self.device = device

    def layer_mask(self, cov_id, saved_path, resume=None, param_per_cov=3,  arch="resnet_56"):
        params = self.model.parameters()
        prefix = saved_path+"/rank_conv"
        subfix = ".npy"

        if resume:
            with open(resume, 'rb') as f:
                self.mask = pickle.load(f)
        else:
            resume=self.job_dir+'/mask'

        self.param_per_cov=param_per_cov

        for index, item in enumerate(params):

            if index == cov_id*param_per_cov:
                break

            if index == (cov_id - 1) * param_per_cov:
                f, c, w, h = item.size()
                rank = np.load(prefix + str(cov_id) + subfix)
                pruned_num = int(self.compress_rate[cov_id - 1] * f)
                ind = np.argsort(rank)[pruned_num:]  # preserved filter id

                zeros = torch.zeros(f, 1, 1, 1).to(self.device)
                for i in range(len(ind)):
                    zeros[ind[i], 0, 0, 0] = 1.
                self.mask[index] = zeros  # covolutional weight
                item.data = item.data * self.mask[index]

            elif index > (cov_id-1)*param_per_cov and index < cov_id*param_per_cov:
                self.mask[index] = torch.squeeze(zeros)
                item.data = item.data * self.mask[index].to(self.device)

        with open(resume, "wb") as f:
            pickle.dump(self.mask, f)

    def grad_mask(self, cov_id):
        params = self.model.parameters()
        for index, item in enumerate(params):
            if index == cov_id*self.param_per_cov:
                break
            item.data = item.data * self.mask[index].to(self.device)#prune certain weight

class mask_resnet34:
    def __init__(self, model=None, compress_rate=[0.50], job_dir='',device=None):
        self.model = model
        self.compress_rate = compress_rate
        self.mask = {}
        self.job_dir=job_dir
        self.device = device

    def layer_mask(self, cov_id, saved_path, resume=None, param_per_cov=3,  arch="resnet34"):
        params = self.model.parameters()
        prefix = saved_path+"/rank_conv"
        subfix = ".npy"

        if resume:
            with open(resume, 'rb') as f:
                self.mask = pickle.load(f)
        else:
            resume=self.job_dir+'/mask'

        self.param_per_cov=param_per_cov

        for index, item in enumerate(params):

            if index == cov_id*param_per_cov:
                break

            if index == (cov_id - 1) * param_per_cov:
                # f, c, w, h = item.size()
                
                f = item.size()[0]
                rank = np.load(prefix + str(cov_id) + subfix)
                pruned_num = int(self.compress_rate[cov_id - 1] * f)
                ind = np.argsort(rank)[pruned_num:]  # preserved filter id

                zeros = torch.zeros(f, 1, 1, 1).to(self.device)
                for i in range(len(ind)):
                    zeros[ind[i], 0, 0, 0] = 1.
                self.mask[index] = zeros  # covolutional weight
                item.data = item.data * self.mask[index]

            elif index > (cov_id-1)*param_per_cov and index < cov_id*param_per_cov:
                self.mask[index] = torch.squeeze(zeros)
                item.data = item.data * self.mask[index].to(self.device)

        with open(resume, "wb") as f:
            pickle.dump(self.mask, f)

    def grad_mask(self, cov_id):
        params = self.model.parameters()
        for index, item in enumerate(params):
            if index == cov_id*self.param_per_cov:
                break
            item.data = item.data * self.mask[index].to(self.device)#prune certain weight


class mask_resnet110:
    def __init__(self, model=None, compress_rate=[0.50], job_dir='',device=None):
        self.model = model
        self.compress_rate = compress_rate
        self.mask = {}
        self.job_dir=job_dir
        self.device = device

    def layer_mask(self, cov_id, saved_path, resume=None, param_per_cov=3,  arch="resnet_110_convwise"):
        params = self.model.parameters()
        prefix = saved_path+"/rank_conv"
        subfix = ".npy"

        if resume:
            with open(resume, 'rb') as f:
                self.mask = pickle.load(f)
        else:
            resume=self.job_dir+'/mask'

        self.param_per_cov=param_per_cov

        for index, item in enumerate(params):

            if index == cov_id*param_per_cov:
                break

            if index == (cov_id - 1) * param_per_cov:
                f, c, w, h = item.size()
                rank = np.load(prefix + str(cov_id) + subfix)
                pruned_num = int(self.compress_rate[cov_id - 1] * f)
                ind = np.argsort(rank)[pruned_num:]  # preserved filter id

                zeros = torch.zeros(f, 1, 1, 1).to(self.device)
                for i in range(len(ind)):
                    zeros[ind[i], 0, 0, 0] = 1.

                self.mask[index] = zeros  # covolutional weight
                item.data = item.data * self.mask[index]

            elif index > (cov_id-1)*param_per_cov and index < cov_id*param_per_cov:
                self.mask[index] = torch.squeeze(zeros)
                item.data = item.data * self.mask[index]

        with open(resume, "wb") as f:
            pickle.dump(self.mask, f)

    def grad_mask(self, cov_id):
        params = self.model.parameters()
        for index, item in enumerate(params):
            if index == cov_id*self.param_per_cov:
                break
            item.data = item.data * self.mask[index].to(self.device)#prune certain weight


class mask_resnet50:
    def __init__(self, model=None, compress_rate=[0.50], job_dir='',device=None):
        self.model = model
        self.compress_rate = compress_rate
        self.mask = {}
        self.job_dir=job_dir
        self.device = device

    def layer_mask(self, cov_id, saved_path, resume=None, param_per_cov=3,  arch="resnet_50_convwise"):
        params = self.model.parameters()
        prefix = saved_path+"/rank_conv"
        subfix = ".npy"

        if resume:
            with open(resume, 'rb') as f:
                self.mask = pickle.load(f)
        else:
            resume=self.job_dir+'/mask'

        self.param_per_cov=param_per_cov

        for index, item in enumerate(params):

            if index == cov_id * param_per_cov:
                break

            if index == (cov_id-1) * param_per_cov:
                f, c, w, h = item.size()
                rank = np.load(prefix + str(cov_id) + subfix)
                pruned_num = int(self.compress_rate[cov_id - 1] * f)
                ind = np.argsort(rank)[pruned_num:]  # preserved filter id
                zeros = torch.zeros(f, 1, 1, 1).to(self.device)#.cuda(self.device[0])#.to(self.device)
                for i in range(len(ind)):
                    zeros[ind[i], 0, 0, 0] = 1.
                self.mask[index] = zeros  # covolutional weight
                item.data = item.data * self.mask[index]

            elif index > (cov_id-1) * param_per_cov and index < cov_id * param_per_cov:
                self.mask[index] = torch.squeeze(zeros)
                item.data = item.data * self.mask[index]

        with open(resume, "wb") as f:
            pickle.dump(self.mask, f)

    def grad_mask(self, cov_id):
        params = self.model.parameters()
        for index, item in enumerate(params):
            if index == cov_id * self.param_per_cov:
                break
            item.data = item.data * self.mask[index]#prune certain weight
