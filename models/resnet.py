'''
Properly implemented ResNet-s for CIFAR10 as described in paper [1].

The implementation and structure of this file is hugely influenced by [2]
which is implemented for ImageNet and doesn't have option A for identity.
Moreover, most of the implementations on the web is copy-paste from
torchvision's resnet and has wrong number of params.

Proper ResNet-s for CIFAR10 (for fair comparision and etc.) has following
number of layers and parameters:

name      | layers | params
ResNet20  |    20  | 0.27M
ResNet32  |    32  | 0.46M
ResNet44  |    44  | 0.66M
ResNet56  |    56  | 0.85M
ResNet110 |   110  |  1.7M
ResNet1202|  1202  | 19.4m

which this implementation indeed has.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
[2] https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

If you use this implementation in you work, please don't forget to mention the
author, Yerlan Idelbayev.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from copy import deepcopy
from torch.autograd import Variable


import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


__all__ = ['ResNet', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet1202']

def _weights_init(m):
    classname = m.__class__.__name__
    #print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1
    k = 0.5
    p = 1
    conv1_batch_table = []
    conv2_batch_table = []
    learning = False
    a = True

    def __init__(self, in_planes, planes, num_classes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.kernel_size1, self.stride1, self.padding1, self.bias1 = 3, stride, 1, False
        self.kernel_size2, self.stride2, self.padding2, self.bias2 = 3, 1, 1, False
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )
        self.learning_k = round(self.k*planes)
        self.conv1_wtable = torch.zeros(num_classes, planes).cuda()
        self.conv2_wtable = torch.zeros(num_classes, planes).cuda()

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        # try:
        #     out = F.relu(self.bn1(self.conv1(x)))
        # except:
        #     print(self.bn1)
        #     print(self.conv1.weight.data.shape)
        #     print(self.conv2.weight.data.shape)
        #     exit()
        if self.learning:
            self.get_most_activated_conv1(out)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        # if self.test_forward:
        #     self.get_most_activated_conv2(out)
        return out

    
    def get_most_activated_conv1(self, out):
        sum_per_filter = torch.sum(out, axis=(2,3))
        bt = []
        for one_batch in sum_per_filter:
            bt.append(torch.topk(one_batch, self.learning_k)[1])
        self.conv1_batch_table = bt

    def get_most_activated_conv2(self, out):
        sum_per_filter = torch.sum(out, axis=(2,3))
        for one_batch in sum_per_filter:
            self.conv2_batch_table.append(torch.topk(one_batch, self.learning_k)[1])

    def save_cw_table(self, mask, labels, layer_ind, ele_ind):
        for b in range(len(self.conv1_batch_table)):
            if mask[b] == 1:
                self.conv1_wtable[labels[b], self.conv1_batch_table[b]] += 1
                # self.conv2_wtable[list(labels.item())[b], filter_idx] += 1
        self.conv1_batch_table = []
        # self.conv2_batch_table = []

    def prune(self, target_classes):
        output_filter_num = round(self.p*self.conv1.weight.shape[0])
        filter_score = self.conv1_wtable[target_classes]  # [filter1_score, filter2_score, ...., filter16_score] * target_classes
        v, top_filters = torch.topk(filter_score, output_filter_num)  # [top1_filter, top2_filter, ..., top8_filter] * target_classes

        candidates = torch.unique(top_filters)
        candidates_occ = []
        for can in candidates:
            candidates_occ.append(filter_score.eq(can).sum().item())

        indices = torch.topk(torch.Tensor(candidates_occ), output_filter_num)[1]
        indices = candidates[indices].sort()[0]

        bn1_weight = self.bn1.weight.data[indices]
        bn1_bias = self.bn1.bias.data[indices]
        running_mean = self.bn1.running_mean.data[indices]
        running_var = self.bn1.running_var.data[indices]
        a = deepcopy(self.bn1)

        self.bn1 = nn.BatchNorm2d(len(indices)).cuda()
        self.bn1.weight = torch.nn.Parameter(bn1_weight, requires_grad=False)
        self.bn1.bias = torch.nn.Parameter(bn1_bias, requires_grad=False)
        self.bn1.running_mean = torch.nn.Parameter(running_mean, requires_grad=False)
        self.bn1.running_var = torch.nn.Parameter(running_var, requires_grad=False)

        conv1_weight = self.conv1.weight.data[indices,:]
        conv2_weight = self.conv2.weight[:,indices,:]
        self.conv1 = nn.Conv2d(self.conv1.in_channels, output_filter_num, kernel_size=self.kernel_size1, stride=self.stride1, padding=self.padding1, bias=self.bias1).cuda()
        self.conv2 = nn.Conv2d(output_filter_num, self.conv2.out_channels, kernel_size=self.kernel_size2, stride=self.stride2, padding=self.padding2, bias=self.bias2).cuda()

        self.conv1.weight = torch.nn.Parameter(conv1_weight, requires_grad=False)
        self.conv2.weight = torch.nn.Parameter(conv2_weight, requires_grad=False)

    def show_table(self):
        print(self.conv1_wtable[:10])

    def learn_table(self):
        self.learning = True

    def finish_learning_table(self):
        self.learning = False


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes):
        super(ResNet, self).__init__()
        self.in_planes = 16
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, self.num_classes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    def update_table(self, mask, labels):
        for i in range(len(self.layer1)):
            self.layer1[i].save_cw_table(mask, labels, 1, i)
        for i in range(len(self.layer2)):
            self.layer2[i].save_cw_table(mask, labels, 2, i)
        for i in range(len(self.layer3)):
            self.layer3[i].save_cw_table(mask, labels, 3, i)

    def prune(self, target_classes):
        for i in range(len(self.layer1)):
            self.layer1[i].prune(target_classes)
        for i in range(len(self.layer2)):
            self.layer2[i].prune(target_classes)
        for i in range(len(self.layer3)):
            self.layer3[i].prune(target_classes)

    def learn_table(self):
        for i in range(len(self.layer1)):
            self.layer1[i].learn_table()
        for i in range(len(self.layer2)):
            self.layer2[i].learn_table()
        for i in range(len(self.layer3)):
            self.layer3[i].learn_table()

    def finish_learning_table(self):
        for i in range(len(self.layer1)):
            self.layer1[i].finish_learning_table()
        for i in range(len(self.layer2)):
            self.layer2[i].finish_learning_table()
        for i in range(len(self.layer3)):
            self.layer3[i].finish_learning_table()

    def show_table(self):
        print("layer1")
        for i in range(len(self.layer1)):
            self.layer1[i].show_table()
        print("layer2")
        for i in range(len(self.layer2)):
            self.layer2[i].show_table()
        print("layer3")
        for i in range(len(self.layer3)):
            self.layer3[i].show_table()


def resnet20(num_classes, whether_prune):
    return ResNet(BasicBlock, [3, 3, 3], num_classes)


def resnet32(num_classes, whether_prune):
    return ResNet(BasicBlock, [5, 5, 5], num_classes)


def resnet44(num_classes, whether_prune):
    return ResNet(BasicBlock, [7, 7, 7], num_classes)


def resnet56(num_classes, whether_prune):
    return ResNet(BasicBlock, [9, 9, 9], num_classes)


def resnet110(num_classes, whether_prune):
    return ResNet(BasicBlock, [18, 18, 18], num_classes)


def resnet1202(num_classes, whether_prune):
    return ResNet(BasicBlock, [200, 200, 200], num_classes)


def test(net):
    import numpy as np
    total_params = 0

    for x in filter(lambda p: p.requires_grad, net.parameters()):
        total_params += np.prod(x.data.numpy().shape)
    print("Total number of params", total_params)
    print("Total layers", len(list(filter(lambda p: p.requires_grad and len(p.data.size())>1, net.parameters()))))


if __name__ == "__main__":
    for net_name in __all__:
        if net_name.startswith('resnet'):
            print(net_name)
            test(globals()[net_name]())
            print()