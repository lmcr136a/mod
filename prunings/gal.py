import os
import re
import time
import numpy as np
import collections

import datetime
import torch
import shutil
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from models.resnet_gal import *
from models.vgg_gal import *
    
from pathlib import Path
from torch.optim import Optimizer
from torch.optim.lr_scheduler import StepLR
from utils.metrics import AverageMeter, accuracy
    
num_params_before_pruning = 0

def get_gal_model(cfg_network, device, n_class):
    model_name = cfg_network["model"]
    if not cfg_network["pruning"]["gal"].get("saved_path", None):
        print("GAL 폴더로 이동하여 bash run.sh 또는 bash run_r56.sh 등을 실행하세요")
        exit()
    
    refine = cfg_network["pruning"]["gal"]["saved_path"] # experiments/r56/resnet_pruned_8.pt
    checkpoint = torch.load(refine, map_location=device)

    state_dict = checkpoint['state_dict_s']
    if 'vgg' in model_name:
        cfg = checkpoint['cfg']
        model = vgg_16_bn_sparse(num_classes=n_class, cfg = cfg).to(device)

    elif 'resnet' in model_name:
        mask = checkpoint['mask']
        print(mask)
        model = resnet56_sparse(has_mask = mask, n_classes=n_class).to(device)

    if num_params_before_pruning:
        num_params_after_pruning = show_nonzero_params_num(model)
        print("NUM OF PARAMS AFTER PRUNING: ", num_params_after_pruning, f"{(num_params_before_pruning-num_params_after_pruning)*100/num_params_before_pruning}% pruned")
    else:
        print("PARAMS AFFTER PRUNING: ", show_nonzero_params_num(model))
    model.load_state_dict(state_dict, strict=False)
    return model



class FISTA(Optimizer):
    def __init__(self, params, lr=1e-2, gamma=0.1):
        defaults = dict(lr=lr, gamma=gamma)
        super(FISTA, self).__init__(params, defaults)

    def step(self, decay=1, closure=None):
        loss = None

        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]

                if 'alpha' not in state or decay:
                    state['alpha'] = torch.ones_like(p.data)
                    state['data'] = p.data
                    y = p.data
                else:
                    alpha = state['alpha']
                    data = state['data']
                    state['alpha'] = (1 + (1 + 4 * alpha**2).sqrt()) / 2
                    y = p.data + ((alpha - 1) / state['alpha']) * (p.data - data)
                    state['data'] = p.data

                mom = y - group['lr'] * grad
                p.data = self._prox(mom, group['lr'] * group['gamma'])

                # no-negative
                p.data = torch.max(p.data, torch.zeros_like(p.data))

        return loss

    def _prox(self, x, gamma):
        y = torch.max(torch.abs(x) - gamma, torch.zeros_like(x))

        return torch.sign(x) * y

class Discriminator(nn.Module):
    def __init__(self, num_classes=10):
        super(Discriminator, self).__init__()
        self.filters = [num_classes, 128, 256, 128, 1]
        block = [
            nn.Linear(self.filters[i], self.filters[i+1]) \
            for i in range(4)
        ]
        
        self.body = nn.Sequential(*block)
        self._initialize_weights()

    def forward(self, input):
        x = self.body(input)
        x = torch.sigmoid(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

def gal_main(dataloader, model_t, cfg_network, device, job_dir, n_classes):

    start_epoch = 0
    best_prec1 = 0.0
    best_prec5 = 0.0

    model_name = cfg_network["model"]
    state_dict_t = torch.load(cfg_network["load_state"], map_location=device)
    # state_dict_t = model_t.state_dict()


    for para in list(model_t.parameters())[:-2]:
        para.requires_grad = False

    model_s = eval(model_name+"_sparse")(n_classes).to(device)

    model_dict_s = model_s.state_dict()
    model_dict_s.update(state_dict_t)
    model_s.load_state_dict(model_dict_s)

    model_d = Discriminator().to(device) 

    models = [model_t, model_s, model_d]

    optimizer_d = optim.SGD(model_d.parameters(), lr=1e-2, momentum=0.9, weight_decay=2e-4)

    param_s = [param for name, param in model_s.named_parameters() if 'mask' not in name]
    param_m = [param for name, param in model_s.named_parameters() if 'mask' in name]

    optimizer_s = optim.SGD(param_s, lr=1e-2, momentum=0.9, weight_decay=2e-4)
    optimizer_m = FISTA(param_m, lr=1e-2, gamma=cfg_network["pruning"]["k"])

    scheduler_d = StepLR(optimizer_d, step_size=30, gamma=0.1)
    scheduler_s = StepLR(optimizer_s, step_size=30, gamma=0.1)
    scheduler_m = StepLR(optimizer_m, step_size=30, gamma=0.1)

    optimizers = [optimizer_d, optimizer_s, optimizer_m]
    schedulers = [scheduler_d, scheduler_s, scheduler_m]

    miu = 1
    sparse_lambda = cfg_network["pruning"]["k"]
    checkpoint = Checkpoint(job_dir)
    global num_params_before_pruning
    num_params_before_pruning = show_nonzero_params_num(model_s)
    print("NUM OF PARAMS BEFORE PRUNING: ", num_params_before_pruning)
    def train(loader_train, models, optimizers, epoch):
        losses_d = AverageMeter()
        losses_data = AverageMeter()
        losses_g = AverageMeter()
        losses_sparse = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        model_t = models[0]
        model_s = models[1]
        model_d = models[2]

        bce_logits = nn.BCEWithLogitsLoss()

        optimizer_d = optimizers[0]
        optimizer_s = optimizers[1]
        optimizer_m = optimizers[2]

        # switch to train mode
        model_d.train()
        model_s.train()
            
        num_iterations = len(loader_train)

        real_label = 1
        fake_label = 0

        for i, (inputs, targets) in enumerate(loader_train, 1):
            num_iters = num_iterations * epoch + i

            inputs = inputs.to(device)
            targets = targets.to(device)

            features_t = model_t(inputs)
            features_s = model_s(inputs)

            ############################
            # (1) Update D network
            ###########################

            for p in model_d.parameters():  
                p.requires_grad = True  

            optimizer_d.zero_grad()

            output_t = model_d(features_t.detach())
            # print('output_t',output_t)
            labels_real = torch.full_like(output_t, real_label, device=device)
            error_real = bce_logits(output_t, labels_real)

            output_s = model_d(features_s.to(device).detach())
            # print('output_s',output_t)
            labels_fake = torch.full_like(output_s, fake_label, device=device)
            error_fake = bce_logits(output_s, labels_fake)

            error_d = error_real + error_fake

            labels = torch.full_like(output_s, real_label, device=device)
            error_d += bce_logits(output_s, labels)

            error_d.backward()
            losses_d.update(error_d.item(), inputs.size(0))

            # print(
            #     'discriminator_loss', error_d.item(), num_iters)

            optimizer_d.step()

            ############################
            # (2) Update student network
            ###########################

            for p in model_d.parameters():  
                p.requires_grad = False  

            optimizer_s.zero_grad()
            optimizer_m.zero_grad()

            error_data = miu * F.mse_loss(features_t, features_s.to(device))

            losses_data.update(error_data.item(), inputs.size(0))
            # print(
            #     'data_loss', error_data.item(), num_iters)
            error_data.backward(retain_graph=True)

            # fool discriminator
            output_s = model_d(features_s.to(device))
            
            labels = torch.full_like(output_s, real_label, device=device)
            error_g = bce_logits(output_s, labels)

            losses_g.update(error_g.item(), inputs.size(0))
            # print(
            #     'generator_loss', error_g.item(), num_iters)
            error_g.backward(retain_graph=True)

            # train mask
            mask = []
            for name, param in model_s.named_parameters():
                if 'mask' in name:
                    mask.append(param.view(-1))
            mask = torch.cat(mask)
            error_sparse = sparse_lambda * torch.norm(mask, 1)
            error_sparse.backward()

            losses_sparse.update(error_sparse.item(), inputs.size(0))
            # print(
            # 'sparse_loss', error_sparse.item(), num_iters)

            optimizer_s.step()

            decay = (epoch % 30 == 0 and i == 1)
            if i % 200 == 0:
                optimizer_m.step(decay)

            prec1, prec5 = accuracy(features_s, targets, topk=(1, 5))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))

            if i % 300 == 0:
                print(
                    'Epoch[{0}]({1}/{2}):\t'
                    'Loss_sparse {loss_sparse.val:.4f} ({loss_sparse.avg:.4f})\t'
                    'Loss_data {loss_data.val:.4f} ({loss_data.avg:.4f})\t'
                    'Loss_d {loss_d.val:.4f} ({loss_d.avg:.4f})\t'
                    'Loss_g {loss_g.val:.4f} ({loss_g.avg:.4f})\t'
                    'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                    'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    epoch, i, num_iterations, loss_sparse=losses_sparse, loss_data=losses_data, loss_g=losses_g, loss_d=losses_d, top1=top1, top5=top5))
                
                mask = []
                pruned = 0
                num = 0
                
                for name, param in model_s.named_parameters():
                    if 'mask' in name:
                        weight_copy = param.clone()
                        param_array = np.array(weight_copy.detach().cpu())
                        pruned += sum(w == 0 for w in param_array)
                        num += len(param_array)
                        
                print("Pruned {} / {}".format(pruned, num))

    def test(loader_test, model_s):
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        cross_entropy = nn.CrossEntropyLoss()

        # switch to eval mode
        model_s.eval()

        with torch.no_grad():
            for i, (inputs, targets) in enumerate(loader_test, 1):
                
                inputs = inputs.to(device)
                targets = targets.to(device).to(torch.long)

                logits = model_s(inputs).to(device)
                loss = cross_entropy(logits, targets)

                prec1, prec5 = accuracy(logits, targets, topk=(1, 5))
                losses.update(loss.item(), inputs.size(0))
                top1.update(prec1.item(), inputs.size(0))
                top5.update(prec5.item(), inputs.size(0))
            
            # print('Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
            # .format(top1=top1, top5=top5))

        return top1.avg, top5.avg

    test_prec1, test_prec5 = test(dataloader.test_loader, model_s)
    print("\nTEST ACC: ", test_prec1)
    for epoch in range(start_epoch, 100):
        for s in schedulers:
            s.step(epoch)

        train(dataloader.train_loader, models, optimizers, epoch)
        test_prec1, test_prec5 = test(dataloader.test_loader, model_s)

        is_best = best_prec1 < test_prec1
        best_prec1 = max(test_prec1, best_prec1)
        best_prec5 = max(test_prec5, best_prec5)

        model_state_dict = model_s.state_dict()

        state = {
            'state_dict_s': model_state_dict,
            'state_dict_d': model_d.state_dict(),
            'best_prec1': best_prec1,
            'best_prec5': best_prec5,
            'optimizer_d': optimizer_d.state_dict(),
            'optimizer_s': optimizer_s.state_dict(),
            'optimizer_m': optimizer_m.state_dict(),
            'scheduler_d': scheduler_d.state_dict(),
            'scheduler_s': scheduler_s.state_dict(),
            'scheduler_m': scheduler_m.state_dict(),
            'epoch': epoch + 1
        }
        checkpoint.save_model(state, epoch + 1, is_best)

    print(f"Best @prec1: {best_prec1:.3f} @prec5: {best_prec5:.3f}")

    best_model = torch.load(f'{job_dir}/checkpoint/model_best.pt', map_location=device)

    if "resnet" in model_name:
        model, save_dir = resnet_preprocess(job_dir, model_name, best_model['state_dict_s'], n_classes)
    elif "vgg" in model_name:
        model, save_dir = vgg_preprocess(job_dir, model_name, best_model['state_dict_s'], n_classes)

    cfg_network["pruning"]["gal"].update({"saved_path": save_dir })


def show_nonzero_params_num(model):
    num_param = 0
    for p in model.parameters():
        for channel in p.data:
            if p.requires_grad and channel.shape and not torch.all(channel == 0):
                num_param += channel.numel()
    return num_param

class Checkpoint():
    def __init__(self, job_dir):
        now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
        today = datetime.date.today()

        self.job_dir = job_dir
        self.ckpt_dir = self.job_dir + '/checkpoint'
        self.run_dir = self.job_dir + '/run'

        def _make_dir(path):
            if not os.path.exists(path): os.makedirs(path)

        _make_dir(self.job_dir)
        _make_dir(self.ckpt_dir)
        _make_dir(self.run_dir)
        
    
    def save_model(self, state, epoch, is_best=None):
        save_path = f'{self.ckpt_dir}/model_{epoch}.pt'
        # print('=> Saving model to {}'.format(save_path))
        torch.save(state, save_path)
        if is_best:
            shutil.copyfile(save_path, f'{self.ckpt_dir}/model_best.pt')


            
def resnet_preprocess(job_dir, arch, state_dict, n_class):
    n = (56 - 2) // 6
    layers = np.arange(0, 3*n ,n)
 
    mask_block = []
    for name, weight in state_dict.items():
        if 'mask' in name:
            mask_block.append(weight.item())

    pruned_num = sum(m <= 0.0 for m in mask_block)
    pruned_blocks = [int(m) for m in np.argwhere(np.array(mask_block) <= 0.0)]

    old_block = 0
    layer = 'layer1'
    layer_num = int(layer[-1])
    new_block = 0
    new_state_dict = OrderedDict()

    for key, value in state_dict.items():
        if 'layer' in key:
            if key.split('.')[0] != layer:
                layer = key.split('.')[0]
                layer_num = int(layer[-1])
                new_block = 0

            if key.split('.')[1] != old_block:
                old_block = key.split('.')[1]

            if mask_block[layers[layer_num-1] + int(old_block)] == 0:
                if layer_num != 1 and old_block == '0' and 'mask' in key:
                    new_block = 1
                continue

            new_key = re.sub(r'\.\d+\.', '.{}.'.format(new_block), key, 1)
            if 'mask' in new_key: new_block += 1

            new_state_dict[new_key] = state_dict[key]

        else:
            new_state_dict[key] = state_dict[key]

    model = resnet56_sparse(has_mask=mask_block, n_classes=n_class).to("cuda")

    print(f"Pruned / Total: {pruned_num} / {len(mask_block)}")
    print("Pruned blocks", pruned_blocks)

    save_dir = f'{job_dir}/{arch}_pruned_{pruned_num}.pt'
    print(f'=> Saving pruned model to {save_dir}')
    
    save_state_dict = {}
    save_state_dict['state_dict_s'] = new_state_dict
    save_state_dict['mask'] = mask_block
    torch.save(save_state_dict, save_dir)

    model.load_state_dict(new_state_dict, strict=False)

    return model, save_dir


def vgg_preprocess(job_dir, arch, state_dict, n_classes):
    pruned_num = 0
    total = 0
    cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 512]
    indexes_list = [np.arange(3)]

    for name, weight in state_dict.items():
        if 'mask' in name:
            weight_copy = weight.clone()
            indexes = np.array(weight_copy.gt(0.0).cpu())  # index to retrain
            selected_index = np.squeeze(np.argwhere(indexes))
            if selected_index.size == 1:
                selected_index = np.resize(selected_index, (1,))
            filters = sum(indexes)
            size = weight_copy.size()[0]

            if (size - filters) == size:
                selected_index = [0]
                filters = 1

            if 'features' in name:
                # get the name of conv in state_dict
                idx = int(re.findall(r"\d+", name.split('.')[1])[0])
                cfg[idx] = filters
                state_dict['features.conv{}.weight'.format(idx)] = state_dict['features.conv{}.weight'.format(idx)][selected_index, :, :, :]
                state_dict['features.conv{}.weight'.format(idx)] = state_dict['features.conv{}.weight'.format(idx)][:, indexes_list[-1], :, :]
                state_dict['features.conv{}.bias'.format(idx)] = state_dict['features.conv{}.bias'.format(idx)][selected_index]
                state_dict['features.norm{}.weight'.format(idx)] = state_dict['features.norm{}.weight'.format(idx)][selected_index]
                state_dict['features.norm{}.bias'.format(idx)] = state_dict['features.norm{}.bias'.format(idx)][selected_index]
                state_dict['features.norm{}.running_mean'.format(idx)] = state_dict['features.norm{}.running_mean'.format(idx)][
                    selected_index]
                state_dict['features.norm{}.running_var'.format(idx)] = state_dict['features.norm{}.running_var'.format(idx)][
                    selected_index]

            elif 'classifier' in name:
                state_dict['classifier.linear1.weight'] = state_dict['classifier.linear1.weight'][selected_index, :]
                state_dict['classifier.linear1.weight'] = state_dict['classifier.linear1.weight'][:,indexes_list[-1]]
                state_dict['classifier.linear1.bias'] = state_dict['classifier.linear1.bias'][selected_index]
                state_dict['classifier.norm1.running_mean'] = state_dict['classifier.norm1.running_mean'][selected_index]
                state_dict['classifier.norm1.running_var'] = state_dict['classifier.norm1.running_var'][selected_index]
                state_dict['classifier.norm1.weight'] = state_dict['classifier.norm1.weight'][selected_index]
                state_dict['classifier.norm1.bias'] = state_dict['classifier.norm1.bias'][selected_index]
                cfg[-1] = filters
                state_dict['classifier.linear2.weight'] = state_dict['classifier.linear2.weight'][:, selected_index]

            indexes_list.append(selected_index)
            pruned_num += size - filters
            total += size

            # change the state dict of mask
            state_dict[name] = weight_copy[selected_index]

    model = vgg_16_bn_sparse(num_classes=n_classes, cfg=cfg).to("cuda")
    model.load_state_dict(state_dict)

    print(f"Pruned / Total: {pruned_num} / {total}")

    save_dir = f'{job_dir}/{arch}_pruned_{pruned_num}.pt'
    save_state_dict = {}
    save_state_dict['state_dict_s'] = state_dict
    save_state_dict['cfg'] = cfg
    torch.save(save_state_dict, save_dir)
    print(f'=> Saving model to {save_dir}')

    return model, save_dir
