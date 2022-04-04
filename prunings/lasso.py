import tqdm
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.autograd as autograd

# import torchvision.models as models
import models

from models.resnet_cifar import *
from models.resnet_imagenet import *
from prunings.lasso_channel import *
from dataset import *
from utils.metrics import AverageMeter, accuracy
from utils.misc import timeit, print_cuda_statistics
# from graphs import *
from math import cos, pi

cudnn.benchmark = True
import logging


class BaseAgent:
    """
    This base class will contain the base functions to be overloaded by any agent you will implement.
    """

    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger("Agent")


class AssembleNetResNet(BaseAgent):
    def __init__(self, config, dataloader, best_network, optimizer, criterion, n_class):
        super().__init__(config)

        # set device
        self.is_cuda = torch.cuda.is_available()
        self.cuda = self.is_cuda
        self.manual_seed = self.config.get("seed", 9099)
        if self.cuda:
            self.device = torch.device("cuda")

            self.logger.info("Program will run on *****GPU-CUDA*****\n")
            print_cuda_statistics()
        else:
            self.device = torch.device("cpu")
            torch.manual_seed(self.manual_seed)
            self.logger.info("Program will run on *****CPU*****\n")

        self.num_classes = n_class

        self.model = best_network   # original model graph, loss function, optimizer, learning scheduler
        self.loss_fn = criterion
        self.optimizer = optimizer
        self.model = self.model.to(self.device)
        self.loss_fn = self.loss_fn.to(self.device)
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.config["optimizer"].get("milestones", 10),
                                                        gamma=self.config.get("gamma", 0.9))

        self.data_loader = dataloader
        self.sub_data_loader = None                                 # sub data loader for sub task

        self.current_epoch = 0      # info for train
        self.current_iteration = 0
        self.best_valid_acc = 0

        self.cls_i = None
        self.channel_importance = dict()

        self.all_list = list()

        self.named_modules_list = dict()
        self.named_conv_list = dict()

        self.original_conv_output = dict()

        self.stayed_channels = dict()

        self.init_graph()

    def init_graph(self, pretrained=True, init_channel_importance=True):     # 모델 그래프와 정보를 초기화

        self.current_epoch = 0
        self.current_iteration = 0
        self.best_valid_acc = 0

        self.cls_i = None
        self.channel_importance = dict()

        self.all_list = list()

        self.named_modules_list = dict()
        self.named_conv_list = dict()

        self.original_conv_output = dict()

        self.stayed_channels = dict()

        for i, m in enumerate(self.model.children()):
            if isinstance(m, torch.nn.Sequential):
                for b in m:
                    self.all_list.append(b)
            else:
                self.all_list.append(m)

        for i, m in enumerate(self.all_list):
            if isinstance(m, models.resnet_imagenet.Bottleneck):
                self.named_modules_list['{}.conv1'.format(i)] = m.conv1
                self.named_conv_list['{}.conv1'.format(i)] = m.conv1
                self.named_modules_list['{}.bn1'.format(i)] = m.bn1
                self.named_modules_list['{}.conv2'.format(i)] = m.conv2
                self.named_conv_list['{}.conv2'.format(i)] = m.conv2
                self.named_modules_list['{}.bn2'.format(i)] = m.bn2
                self.named_modules_list['{}.conv3'.format(i)] = m.conv3
                self.named_conv_list['{}.conv3'.format(i)] = m.conv3
                self.named_modules_list['{}.bn3'.format(i)] = m.bn3

                if m.downsample is not None:
                    self.named_modules_list['{}.downsample'.format(i)] = m.downsample
                    if len(m.downsample) > 0:
                        self.named_conv_list['{}.downsample'.format(i)] = m.downsample[0]
                    else:
                        self.named_conv_list['{}.downsample'.format(i)] = m.downsample
            else:
                self.named_modules_list['{}'.format(i)] = m
                if isinstance(m, torch.nn.Conv2d):
                    self.named_conv_list['{}'.format(i)] = m

        if init_channel_importance is True:
            self.channel_importance = dict()

    def set_subtask(self, *cls_i):
        self.cls_i = cls_i
        # self.sub_data_loader = SpecializedImagenetDataLoader(self.config, *self.cls_i)
    def save_checkpoint(self, file_name="checkpoint.pth", is_best=False):
        pass

    def record_channel_importance(self):
        def save_grad(grads, idx):
            def hook(grad):
                grads[idx] = grad
            return hook

        for name, module in self.named_conv_list.items():
            self.channel_importance[name] = torch.zeros(module.out_channels)

        outputs, grads = {}, {}
        for inputs, labels in self.data_loader.train_loader:
            if self.cuda:
                inputs = inputs.cuda(non_blocking=self.config.get("async_loading", True))
                inputs.requires_grad = True

            x = inputs
            for i, m in enumerate(self.all_list):
                if isinstance(m, torch.nn.Conv2d):
                    x = outputs['{}'.format(i)] = m(x)
                    outputs['{}'.format(i)].register_hook(save_grad(grads, '{}'.format(i)))
                    continue
                elif isinstance(m, models.resnet_cifar.BasicBlock) or isinstance(m, models.resnet_imagenet.BasicBlock):
                    shortcut = x
                    # first conv in residual block
                    x = outputs['{}.conv1'.format(i)] = m.conv1(x)
                    outputs['{}.conv1'.format(i)].register_hook(save_grad(grads, '{}.conv1'.format(i)))
                    x = m.relu(m.bn1(x))
                    # second conv in residual block
                    x = m.conv2(x)
                    x = m.relu(m.bn2(x))
                    if m.downsample is not None:
                        shortcut = m.downsample(shortcut)
                        x = x + shortcut
                        x = m.relu(x)
                        outputs['{}'.format(i)] = x
                        outputs['{}'.format(i)].register_hook(save_grad(grads, '{}'.format(i)))
                    else:
                        x = x + shortcut
                        x = m.relu(x)
                elif isinstance(m, torch.nn.Linear):
                    x = m(torch.flatten(x, start_dim=1))
                    continue
                else:
                    x = m(x)
            y_hat = x
            y_hat[:, self.cls_i].backward(gradient=torch.ones_like(y_hat[:, self.cls_i]))
            self.cal_importance(outputs, grads)

    def cal_importance(self, output_list, grads_list):
        before_grad = None
        for n in self.channel_importance.keys():
            if n == '0':
                grad = grads_list[n]
                before_grad = grad
                output = output_list[n]
                before_output = output
            elif n.split('.')[-1] == "conv1":
                grad = grads_list[n]
                output = output_list[n]
            elif n.split('.')[-1] == "conv2":
                if n.split('.')[0] in grads_list:
                    grad = grads_list[n.split('.')[0]]  # case block have conv in skip connection
                    output = output_list[n.split('.')[0]]
                else:
                    grad = before_grad      # case block skip connection with no parameter
                    output = before_output
            elif n.split('.')[-1] == "downsample":
                grad = grads_list[n.split('.')[0]]
                before_grad = grad
                output = output_list[n.split('.')[0]]
                before_output = output

            importance = (grad * output).mean(dim=(2, 3))
            total_importance = torch.abs(importance).sum(dim=0)
            self.channel_importance[n] += total_importance.data.cpu()

    def record_conv_output(self, inputs):
        if self.cuda:
            inputs = inputs.cuda(non_blocking=self.config.get("async_loading", True))
        x = inputs
        for i, m in enumerate(self.all_list):
            if isinstance(m, torch.nn.Conv2d):
                x = m(x)
                self.original_conv_output['{}'.format(i)] = x
            elif isinstance(m, models.resnet_cifar.BasicBlock) or isinstance(m, models.resnet_imagenet.BasicBlock):
                shortcut = x
                # first conv in residual block
                x = m.conv1(x)
                self.original_conv_output['{}.conv1'.format(i)] = x
                x = torch.relu(m.bn1(x))
                # second conv in residual block
                x = m.conv2(x)
                self.original_conv_output['{}.conv2'.format(i)] = x
                x = torch.relu(m.bn2(x) + m.downsample(shortcut))
                assert torch.all(x == m(shortcut))

            elif isinstance(m, models.resnet_imagenet.Bottleneck):
                shortcut = x
                # first conv in residual block
                x = m.conv1(x)
                self.original_conv_output['{}.conv1'.format(i)] = x
                x = m.relu(m.bn1(x))
                # second conv in residual block
                x = m.conv2(x)
                self.original_conv_output['{}.conv2'.format(i)] = x
                x = m.relu(m.bn2(x))
                # third conv in residual block
                x = m.conv3(x)
                self.original_conv_output['{}.conv3'.format(i)] = x
                x = m.relu(m.bn3(x))

                if m.downsample is not None:
                    if len(m.downsample) > 0:
                        shortcut = m.downsample[0](shortcut)
                    else:
                        shortcut = m.downsample(shortcut)
                    
                    self.original_conv_output['{}.downsample'.format(i)] = shortcut
                    if len(m.downsample) > 1:
                        shortcut = m.downsample[1](shortcut)
                    else:
                        pass
                    x = x + shortcut
                    x = m.relu(x)
                else:
                    x = x + shortcut
                    x = m.relu(x)
            elif isinstance(m, torch.nn.Linear):
                x = torch.flatten(x, start_dim=1)
                continue
            else:
                x = m(x)

    def run(self):
        pass

    @timeit
    def compress(self, log_dir, method='gradient', k=0.5):
        if method == "first_k":
            for i, m in enumerate(self.all_list):
                if isinstance(m, models.resnet_cifar.BasicBlock) or isinstance(m, models.resnet_imagenet.BasicBlock):
                    conv1 = self.named_modules_list[str(i) + '.conv1']
                    bn1 = self.named_modules_list[str(i) + '.bn1']
                    conv2 = self.named_modules_list[str(i) + '.conv2']
                    bn2 = self.named_modules_list[str(i) + '.bn2']
                    indices_stayed = [i for i in range(int(conv1.out_channels * k))]
                    module_surgery(conv1, bn1, conv2, indices_stayed)
            return

        print("DATALOADER LABELS: ", self.data_loader.train_loader.dataset.targets[:30])
        
        inputs, _ = next(iter(self.data_loader.train_loader))
        if self.cuda:
            inputs = inputs.cuda(non_blocking=self.config.get("async_loading", True))
        self.record_conv_output(inputs)
        inputs.cpu()
        start = time.time()
        
        total_indices_stayed = [[] for i in range(len(self.all_list))]
        print(f"mmmmmmmmmmmmmmmmmmmmmmmmmm {method} mmmmmmmmmmmmmmmmmmmmmmmmmm")
        if method == 'max_output':    # 채널의 가장 큰 아웃풋을 활용한 중요도로 프루닝
            x = inputs
            for i, m in enumerate(self.all_list):
                if isinstance(m, models.resnet_cifar.BasicBlock) or isinstance(m, models.resnet_imagenet.BasicBlock):
                    conv1 = self.named_modules_list[str(i)].conv1
                    bn1 = self.named_modules_list[str(i)].bn1
                    conv2 = self.named_modules_list[str(i)].conv2
                    num_channel = int(k * conv1.out_channels)
                    channel_norm = conv1(x).norm(dim=(2, 3)).mean(dim=0)
                    indices_stayed = torch.argsort(channel_norm, descending=True)[:num_channel]  # 가장 큰 채널들의 index를 리턴
                    total_indices_stayed[i].append(indices_stayed)

                    module_surgery(conv1, bn1, conv2, indices_stayed)
                    pruned_input_feature = torch.relu(bn1(conv1(x)))
                    output_feature = self.original_conv_output[str(i) + '.conv2']
                    weight_reconstruction(conv2, pruned_input_feature, output_feature, use_gpu=self.cuda)
                elif isinstance(m, torch.nn.Linear):
                    break
                x = m(x)
        elif method == 'greedy':
            x = inputs
            for i, m in enumerate(self.all_list):
                if isinstance(m, models.resnet_imagenet.Bottleneck):
                    conv1 = self.named_modules_list[str(i)].conv1
                    bn1 = self.named_modules_list[str(i)].bn1
                    conv2 = self.named_modules_list[str(i)].conv2
                    bn2 = self.named_modules_list[str(i)].bn2
                    conv3 = self.named_modules_list[str(i)].conv3


                    f_input_feature = torch.relu(bn1(conv1(x)))
                    f_indices_stayed, f_indices_pruned = channel_selection(f_input_feature, conv2, sparsity=(1.-k), method='greedy')
                    total_indices_stayed[i].append(f_indices_stayed)
                    module_surgery(conv1, bn1, conv2, f_indices_stayed)
                    f_pruned_input_feature = torch.relu(bn1(conv1(x)))
                    f_output_feature = self.original_conv_output[str(i) + '.conv2']
                    weight_reconstruction(conv2, f_pruned_input_feature, f_output_feature, use_gpu=self.cuda)


                    s_input_feature = torch.relu(bn2(conv2(f_pruned_input_feature)))
                    s_indices_stayed, s_indices_pruned = channel_selection(s_input_feature, conv3, sparsity=(1.-k), method='greedy')
                    module_surgery(conv2, bn2, conv3, s_indices_stayed)
                    s_pruned_input_feature = torch.relu(bn2(conv2(f_pruned_input_feature)))
                    s_output_feature = self.original_conv_output[str(i) + '.conv3']
                    weight_reconstruction(conv3, s_pruned_input_feature, s_output_feature, use_gpu=self.cuda)


                elif isinstance(m, torch.nn.Linear):
                    break
                x = m(x)

        elif method == 'lasso': ###################################################################
            x = inputs
            for i, m in enumerate(self.all_list):
                print(m)
                if isinstance(m, models.resnet_cifar.BasicBlock) or isinstance(m, models.resnet_imagenet.BasicBlock):
                    conv1 = self.named_modules_list[str(i)].conv1
                    bn1 = self.named_modules_list[str(i)].bn1
                    conv2 = self.named_modules_list[str(i)].conv2
                    input_feature = torch.relu(bn1(conv1(x)))
                    
                    indices_stayed, indices_pruned = channel_selection(input_feature, conv2, sparsity=(1.-k), method='lasso')
                    total_indices_stayed[i].append(indices_stayed)
                    
                    module_surgery(conv1, bn1, conv2, indices_stayed)
                    pruned_input_feature = torch.relu(bn1(conv1(x)))
                    output_feature = self.original_conv_output[str(i) + '.conv2'].cuda() if self.cuda else self.original_conv_output[str(i) + '.conv2']
                    weight_reconstruction(conv2, pruned_input_feature, output_feature, use_gpu=self.cuda)
                elif isinstance(m, torch.nn.Linear):
                    break
                x = m(x)
        torch.save(total_indices_stayed, log_dir+"/total_indices_stayed.pt")
        print(total_indices_stayed)
        self.original_conv_output = dict()  # clear original output to save cuda memory


    def lasso_compress(self, l_cfg, log_dir):
        k = l_cfg.get("k", 0.49)
        saved_index = l_cfg.get("saved_index", None)
        batch = l_cfg.get("batch", None)
        print('\n\n\nTHIS IS LASSO COMPRESS!!!!!!!!!!!!!!!!\n\n\n')

        if batch:
            concated_input = torch.Tensor([])
            for inputs, label in (self.data_loader.train_loader):
                concated_input = torch.cat((concated_input, inputs), 0)
            if self.cuda:
                concated_input = concated_input.cuda(non_blocking=self.config.get("async_loading", True))
            concated_input = torch.split(concated_input, batch, dim=0)
            print(len(concated_input))
            loader = zip(concated_input, [0]*len(concated_input))
        else:        
            loader = self.data_loader.train_loader
        

        if not saved_index:
            total_indices_stayed = [[] for i in range(len(self.all_list))]
            for inputs, label in loader:
                if self.cuda:
                    inputs = inputs.cuda(non_blocking=self.config.get("async_loading", True))
                x = inputs
                print("nnnnnnmnmnmnmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmnmnmnmnnnnnn")
                for i, m in enumerate(self.all_list):
                    if isinstance(m, models.resnet_cifar.BasicBlock) or isinstance(m, models.resnet_imagenet.BasicBlock):
                        conv1 = self.named_modules_list[str(i)].conv1
                        bn1 = self.named_modules_list[str(i)].bn1
                        conv2 = self.named_modules_list[str(i)].conv2
                        input_feature = torch.relu(bn1(conv1(x)))
                        
                        indices_stayed, indices_pruned = channel_selection(input_feature, conv2, sparsity=(1.-k), method='lasso')
                        total_indices_stayed[i].append(indices_stayed)
                    elif isinstance(m, torch.nn.Linear):
                        break
                    x = m(x)
                break  ##########################
            torch.save(total_indices_stayed, log_dir+"/total_indices_stayed.pt")

        else:
            print("LOADING SAVED INEDX...mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmnmnmnmnnnnnn")
            total_indices_stayed = torch.load(saved_index)

        total_indices_stayed = self.get_total_top_indices(total_indices_stayed)
        print(total_indices_stayed)


        inputs, _ = next(iter(self.data_loader.train_loader))
        if self.cuda:
            inputs = inputs.cuda(non_blocking=self.config.get("async_loading", True))
        self.record_conv_output(inputs)
        x = inputs
        for i, m in enumerate(self.all_list):
            if isinstance(m, models.resnet_cifar.BasicBlock) or isinstance(m, models.resnet_imagenet.BasicBlock):
                conv1 = self.named_modules_list[str(i)].conv1
                bn1 = self.named_modules_list[str(i)].bn1
                conv2 = self.named_modules_list[str(i)].conv2
                input_feature = torch.relu(bn1(conv1(x)))
                
                indices_stayed = total_indices_stayed[i]
                
                module_surgery(conv1, bn1, conv2, indices_stayed)
                pruned_input_feature = torch.relu(bn1(conv1(x)))
                output_feature = self.original_conv_output[str(i) + '.conv2'].cuda() if self.cuda else self.original_conv_output[str(i) + '.conv2']
                weight_reconstruction(conv2, pruned_input_feature, output_feature, use_gpu=self.cuda)
            elif isinstance(m, torch.nn.Linear):
                break
            x = m(x)
        self.original_conv_output = dict()  # clear original output to save cuda memory
    


    def get_total_top_indices(self, total_indices_stayed, k=None):
        output = []
        for a_layer_indices in total_indices_stayed:
            if len(a_layer_indices)==0:
                output.append([])
            else:
                output.append(self.get_top_indices(torch.Tensor(a_layer_indices)).tolist())
        return output


    def get_top_indices(self, tensor, k=None):
        ont_hot = F.one_hot(tensor.flatten().long())
        count = torch.sum(ont_hot, axis=0)
        if k:
            _, ind = torch.topk(count, k)
        else:
            _, ind = torch.topk(count, len(tensor[0]))
        return ind

    def adjust_learning_rate(self,optimizer, epoch, iteration, num_iter):
        warmup_epoch = 5
        warmup_iter = warmup_epoch * num_iter
        current_iter = iteration + epoch * num_iter
        max_iter = 100 * num_iter

        lr = 0.1 * (1 + cos(pi * (current_iter - warmup_iter) / (max_iter - warmup_iter))) / 2

        if epoch < warmup_epoch:
            lr = 0.1* current_iter / warmup_iter

        if iteration == 0:
            print('current learning rate:{0}'.format(lr))

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def train(self, specializing=False, freeze_conv=False):
        """
        Main training function, with per-epoch model saving
        :return:
        """
        if freeze_conv:
            for param in self.model.features.parameters():
                param.requires_grad = False

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005,
                                         nesterov=True)
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.config["optimizer"].get("milestones", 10),
                                                        gamma=self.config.get("gamma", 0.9))
        self.model.to(self.device)

        history = []
        for epoch in range(self.get("current_epoch", 0), self.config["epoch"]):
            self.current_epoch = epoch
            self.train_one_epoch(specializing)

            valid_acc = self.validate(specializing)
            is_best = valid_acc > self.best_valid_acc
            if is_best:
                self.best_valid_acc = valid_acc
            self.save_checkpoint(is_best=is_best)

            history.append(valid_acc)
            self.scheduler.step(epoch)

        if freeze_conv:
            for param in self.model.parameters():
                param.requires_grad = True

        return self.best_valid_acc, history

    def train_one_epoch(self, specializing=False):
        """
        One epoch training function
        :return:
        """
        if specializing:
            tqdm_batch = tqdm.tqdm(self.sub_data_loader.binary_train_loader,
                                   total=self.sub_data_loader.binary_train_iterations,
                                   desc="Epoch-{}-".format(self.current_epoch))
        else:
            tqdm_batch = tqdm.tqdm(self.data_loader.train_loader, total=self.data_loader.train_iterations,
                                   desc="Epoch-{}-".format(self.current_epoch))

        self.model.train()

        epoch_loss = AverageMeter()
        top1_acc = AverageMeter()
        top5_acc = AverageMeter()

        current_batch = 0
        for i,(x, y) in enumerate(tqdm_batch):
            if self.cuda:
                x, y = x.cuda(non_blocking=self.config.get("async_loading", True)), y.cuda(non_blocking=self.config.get("async_loading", True))

            self.optimizer.zero_grad()
            self.adjust_learning_rate(self.optimizer, self.current_epoch, i, self.data_loader.train_iterations)

            pred = self.model(x)
            y = y.to(torch.long)
            cur_loss = self.loss_fn(pred, y)

            if np.isnan(float(cur_loss.item())):
                raise ValueError('Loss is nan during training...')

            cur_loss.backward()
            self.optimizer.step()

            if specializing:
                top1 = accuracy(pred.data, y.data)
                top1_acc.update(top1[0].item(), x.size(0))
            else:
                top1, top5 = accuracy(pred.data, y.data, topk=(1, 5))
                top1_acc.update(top1.item(), x.size(0))
                top5_acc.update(top5.item(), x.size(0))

            epoch_loss.update(cur_loss.item())

            self.current_iteration += 1
            current_batch += 1

        tqdm_batch.close()

        self.logger.info("Training at epoch-" + str(self.current_epoch) + " | " + "loss: " + str(epoch_loss.val) +
                         "\tTop1 Acc: " + str(top1_acc.val))

    def validate(self, specializing=False):
        """
        One epoch validation
        :return:
        """
        if specializing:
            tqdm_batch = tqdm.tqdm(self.sub_data_loader.binary_valid_loader,
                                   total=self.sub_data_loader.binary_valid_iterations,
                                   desc="Epoch-{}-".format(self.current_epoch))
        else:
            tqdm_batch = tqdm.tqdm(self.data_loader.valid_loader, total=self.data_loader.valid_iterations,
                                   desc="Valiation at -{}-".format(self.current_epoch))

        self.model.eval()

        epoch_loss = AverageMeter()
        top1_acc = AverageMeter()
        top5_acc = AverageMeter()

        for x, y in tqdm_batch:
            if self.cuda:
                x, y = x.cuda(non_blocking=self.config.get("async_loading", True)), y.cuda(non_blocking=self.config.get("async_loading", True))

            # model
            pred = self.model(x)
            # loss
            y = y.to(torch.long)
            cur_loss = self.loss_fn(pred, y)
            if np.isnan(float(cur_loss.item())):
                raise ValueError('Loss is nan during validation...')

            if specializing:
                top1 = accuracy(pred.data, y.data)
                top1_acc.update(top1[0].item(), x.size(0))
            else:
                top1, top5 = accuracy(pred.data, y.data, topk=(1, 5))
                top1_acc.update(top1.item(), x.size(0))
                top5_acc.update(top5.item(), x.size(0))

            epoch_loss.update(cur_loss.item())

        self.logger.info("Validation results at epoch-" + str(self.current_epoch) + " | " + "loss: " +
                         str(epoch_loss.avg) + "\tTop1 Acc: " + str(top1_acc.val))

        tqdm_batch.close()

        return top1_acc.avg

    def finalize(self):
        """
        Finalize all the operations of the 2 Main classes of the process the operator and the data loader
        :return:
        """
        pass

if __name__=="__main__":
    import argparse
    import easydict

    from utils.config import *
    from utils.count_model import *
    from torchsummaryX import summary

    config = easydict.EasyDict()
    config.exp_name = "assemblenet_imagenet"
    config.log_dir = os.path.join("experiments", config.exp_name, "logs/")
    create_dirs([config.log_dir])
    setup_logging(config.log_dir)

    config.load_file = 'experiments/cifar100_res20_100epo/best_model.pt'
    config.cuda = True
    config.gpu_device = 2
    config.seed = 1
    config.learning_rate = 0.001
    config.momentum = 0.9
    config.milestones = [5, 10,15]
    config.gamma = 0.9
    config.weight_decay = 0.0005
    config.img_size = 32
    config.num_classes = 100
    config.data_mode = "download"
    config.data_loader_workers = 4
    config.pin_memory = True
    config.async_loading = True
    config.batch_size = 128
    config.async_loading = True
    config.max_epoch = 30
    torch.cuda.init()


    agent = globals()["AssembleNetResNet"](config)
    agent.init_graph(pretrained=False)
    agent.load_checkpoint(file_path= config.load_file)
    summary(agent.model, torch.zeros((1, 3, 32, 32)).to(torch.device("cuda")))
    # agent.validate()


    agent.lasso_compress(k=0.49)
    # agent.compress("lasso", k=0.49)
    summary(agent.model, torch.zeros((1, 3, 32, 32)).to(torch.device("cuda")))

    best,history = agent.train(specializing=False, freeze_conv=False)