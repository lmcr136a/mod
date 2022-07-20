import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import models

from models.resnet_cifar import *
from models.resnet_imagenet import *
from prunings.lasso_channel import *
from dataset import *

cudnn.benchmark = True


class LASSOVGGBN:
    def __init__(self, config, dataloader, best_network, n_class):
        self.config = config

        # set device
        self.is_cuda = torch.cuda.is_available()
        self.cuda = self.is_cuda
        self.manual_seed = self.config.get("seed", 9099)
        self.device = torch.device("cpu")
        torch.manual_seed(self.manual_seed)

        self.num_classes = n_class

        self.model = best_network   # original model graph, loss function, optimizer, learning scheduler
        self.model = self.model.to(self.device)

        self.data_loader = dataloader

        self.all_list = list()
        self.named_modules_list = dict()
        self.named_conv_list = dict()

        self.original_conv_output = dict()
        self.stayed_channels = dict()
        self.init_graph()


    def init_graph(self):     # 모델 그래프와 정보를 초기화
        # set model graph & information holder

        self.named_modules_idx_list = dict()
        self.named_modules_list = dict()
        self.named_conv_list = dict()
        self.named_conv_idx_list = dict()

        self.original_conv_output = dict()

        self.stayed_channels = dict()

        i = 0
        for idx, m in enumerate(self.model.features):
            if isinstance(m, torch.nn.Conv2d):
                self.named_modules_idx_list['{}.conv'.format(i)] = idx
                self.named_modules_list['{}.conv'.format(i)] = m
                self.named_conv_idx_list['{}.conv'.format(i)] = idx
                self.named_conv_list['{}.conv'.format(i)] = m
            elif isinstance(m, torch.nn.BatchNorm2d):
                self.named_modules_idx_list['{}.bn'.format(i)] = idx
                self.named_modules_list['{}.bn'.format(i)] = m
                i += 1


    def record_conv_output(self, inputs):
        x = inputs
        i = 0
        for m in self.model.features:
            x = m(x)
            if isinstance(m, torch.nn.Conv2d):
                self.original_conv_output['{}.conv'.format(i)] = x.data
                i += 1


    def compress(self, p_cfg, log_dir):
        self.k = p_cfg["k"]
        saved_index =  p_cfg['lasso'].get(f"saved_index", None)
        print(f'THIS IS LASSO COMPRESS!!!!! k:{self.k}!!!!!!!! saved index: {saved_index}!!!')
        total_indices_stayed = [[] for i in range(len(self.all_list))]
        inputs, _ = next(iter(self.data_loader.train_loader))
        self.record_conv_output(inputs)

        if not saved_index:
            for i, m in enumerate(list(self.named_conv_list.values())[:-1]):    # 마지막 레이어 전까지
                if isinstance(m, torch.nn.Conv2d):
                    next_m_idx = self.named_modules_idx_list[str(i + 1) + '.conv']
                    bn, next_m = self.named_modules_list[str(i) + '.bn'], self.named_modules_list[str(i + 1) + '.conv']
                    next_input_features = self.model.features[:next_m_idx](inputs)

                    indices_stayed, indices_pruned = channel_selection(next_input_features, next_m, sparsity=(1. - self.k), method='lasso')
                    total_indices_stayed[i].append(indices_stayed)
                    module_surgery(m, bn, next_m, indices_stayed)

                    next_output_features = self.original_conv_output[str(i + 1) + '.conv']
                    next_m_idx = self.named_conv_idx_list[str(i + 1) + '.conv']
                    pruned_next_inputs_features = self.model.features[:next_m_idx](inputs)
                    weight_reconstruction(next_m, pruned_next_inputs_features, next_output_features, use_gpu=self.cuda)
                    self.stayed_channels[str(i) + '.conv'] = set(indices_stayed)
            torch.save(total_indices_stayed, log_dir+"/total_indices_stayed.pt")

        else:
            print("LOADING SAVED INEDX...mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmnmnmnmnnnnnn")
            total_indices_stayed = torch.load(saved_index)
            total_indices_stayed = self.get_total_top_indices(total_indices_stayed)
            print(total_indices_stayed)

            for i, m in enumerate(self.all_list):
                if isinstance(m, models.resnet_cifar.BasicBlock) or isinstance(m, models.resnet_imagenet.BasicBlock):
                    next_m_idx = self.named_modules_idx_list[str(i + 1) + '.conv']
                    bn, next_m = self.named_modules_list[str(i) + '.bn'], self.named_modules_list[str(i + 1) + '.conv']
                    next_input_features = self.model.features[:next_m_idx](inputs)

                    indices_stayed = total_indices_stayed[i]
                    module_surgery(m, bn, next_m, indices_stayed)

                    next_output_features = self.original_conv_output[str(i + 1) + '.conv']
                    next_m_idx = self.named_conv_idx_list[str(i + 1) + '.conv']
                    pruned_next_inputs_features = self.model.features[:next_m_idx](inputs)
                    weight_reconstruction(next_m, pruned_next_inputs_features, next_output_features, use_gpu=self.cuda)

            

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