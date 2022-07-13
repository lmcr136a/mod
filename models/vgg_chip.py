import torch
import torch.nn as nn
from collections import OrderedDict

defaultcfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512]
relucfg = [2, 6, 9, 13, 16, 19, 23, 26, 29, 33, 36, 39, 42]

class VGG(nn.Module):
    def __init__(self, sparsity, target_classes, cfg=None, num_classes=10):
        super(VGG, self).__init__()

        if cfg is None:
            cfg = defaultcfg
        if not sparsity:
            sparsity = [0.]*100
        self.relucfg = relucfg

        self.sparsity = sparsity[:]
        self.sparsity.append(0.0)

        self.features1, self.features2, in_channels = self._make_layers(cfg)
        self.classifier = nn.Sequential(OrderedDict([
            ('linear1', nn.Linear(in_channels, cfg[-1])),
            ('norm1', nn.BatchNorm1d(cfg[-1])),
            ('relu1', nn.ReLU(inplace=True)),
            ('linear2', nn.Linear(cfg[-1], num_classes)),
        ]))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        if target_classes:
            self.output_mask = torch.zeros(num_classes)
            self.output_mask[target_classes] = 1
        else:
            self.output_mask = torch.ones(num_classes)
        self.output_mask.to(torch.device('cuda:0'))

    def _make_layers(self, cfg):

        layers1 = nn.Sequential()
        layers2 = nn.Sequential()
        in_channels = 3
        cnt=0

        inter = 7
        for i, x in enumerate(cfg[:inter]):
            if x == 'M':
                layers1.add_module('pool%d' % i, nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                x = int(x * (1-self.sparsity[cnt]))

                cnt+=1
                # print(in_channels, x)
                conv2d = nn.Conv2d(in_channels, x, kernel_size=3, padding=1)
                layers1.add_module('conv%d' % i, conv2d)
                layers1.add_module('norm%d' % i, nn.BatchNorm2d(x))
                layers1.add_module('relu%d' % i, nn.ReLU(inplace=True))
                in_channels = x

        
        for i, x in enumerate(cfg[inter:]):
            i += inter
            if x == 'M':
                layers2.add_module('pool%d' % i, nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                x = int(x * (1-self.sparsity[cnt]))

                cnt+=1
                # print(in_channels, x)
                conv2d = nn.Conv2d(in_channels, x, kernel_size=3, padding=1)
                layers2.add_module('conv%d' % i, conv2d)
                layers2.add_module('norm%d' % i, nn.BatchNorm2d(x))
                layers2.add_module('relu%d' % i, nn.ReLU(inplace=True))
                in_channels = x

        return layers1, layers2, in_channels

    def forward(self, x):
        # print("\n", x.shape)
        x = self.features1(x)
        x = self.features2(x)
        # print(x.shape)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        out = torch.mul(out, self.output_mask)
        return x

def vgg_16_bn_chip(sparsity, num_classes, target_classes):
    return VGG(sparsity=sparsity, num_classes=num_classes, target_classes=target_classes)