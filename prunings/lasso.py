import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import models

from models.resnet_cifar import *
from models.resnet_imagenet import *
from prunings.lasso_channel import *
from dataset import *
from math import cos, pi

cudnn.benchmark = True


class LASSOResNet:
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


    def init_graph(self):  
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
            if isinstance(m, models.resnet_imagenet.Bottleneck): # 헐??
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


    def record_conv_output(self, inputs, nocuda=False):
        # if self.cuda:
        #     inputs = inputs.cuda(non_blocking=self.config.get("async_loading", True))
        x = inputs
        for i, m in enumerate(self.all_list):
            # if self.cuda and not nocuda:
            #     m.to("cuda")
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
                # print(x.shape, m)
                x = m(x)

    def compress(self, p_cfg, log_dir):
        self.k = p_cfg["k"]
        saved_index =  p_cfg['lasso'].get(f"saved_index", None)
        print(f'THIS IS LASSO COMPRESS!!!!! k:{self.k}!!!!!!!! saved index: {saved_index}!!!')
        total_indices_stayed = [[] for i in range(len(self.all_list))]
        inputs, _ = next(iter(self.data_loader.train_loader))
        self.record_conv_output(inputs)
        x = inputs

        if not saved_index:
            for i, m in enumerate(self.all_list):
                if isinstance(m, models.resnet_cifar.BasicBlock) or isinstance(m, models.resnet_imagenet.BasicBlock):
                    conv1 = self.named_modules_list[str(i)].conv1
                    bn1 = self.named_modules_list[str(i)].bn1
                    conv2 = self.named_modules_list[str(i)].conv2
                    input_feature = torch.relu(bn1(conv1(x)))
                    
                    indices_stayed, indices_pruned = channel_selection(input_feature, conv2, sparsity=(1.-self.k), method='lasso')
                    total_indices_stayed[i].append(indices_stayed)

                    module_surgery(conv1, bn1, conv2, indices_stayed)
                    pruned_input_feature = torch.relu(bn1(conv1(x)))
                    output_feature = self.original_conv_output[str(i) + '.conv2'].cuda() if self.cuda else self.original_conv_output[str(i) + '.conv2']
                    weight_reconstruction(conv2, pruned_input_feature, output_feature, use_gpu=self.cuda)

                elif isinstance(m, torch.nn.Linear):
                    break
                x = m(x)
            torch.save(total_indices_stayed, log_dir+"/total_indices_stayed.pt")

        else:
            print("LOADING SAVED INEDX...mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmnmnmnmnnnnnn")
            total_indices_stayed = torch.load(saved_index)
            total_indices_stayed = self.get_total_top_indices(total_indices_stayed)
            print(total_indices_stayed)

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


    # def lasso_compress(self, p_cfg, log_dir, save_i=0):
    #     self.k = p_cfg["k"]
    #     saved_index = p_cfg['lasso'].get(f"saved_index_{save_i}", None)
    #     # batch = l_cfg.get("batch", None)
    #     print(f'THIS IS LASSO COMPRESS!!!!! k:{self.k}!!!!!!!!!!!')
    #     if not saved_index:
    #         total_indices_stayed = [[] for i in range(len(self.all_list))]
    #         p = torch.multiprocessing.Pool(p_cfg['process_num'])
    #         inputs, _ = next(iter(self.data_loader.train_loader))
    #         # if self.cuda:
    #         #     inputs = inputs.cuda(non_blocking=self.config.get("async_loading", True))
    #         x = inputs

    #         map_list = []
    #         for i, m in enumerate(self.all_list):
    #             if isinstance(m, models.resnet_cifar.BasicBlock) or isinstance(m, models.resnet_imagenet.BasicBlock):
    #                 map_list.append([i, m, x.detach().cpu()])
    #             elif isinstance(m, torch.nn.Linear):
    #                 break
    #             x = m(x)

    #         torch.multiprocessing.set_sharing_strategy('file_system')
    #         print('Multiprocessing Start:  ', p_cfg['process_num'], '-process')
    #         start = time.time()
    #         indices = p.map(self.merge_pool, map_list)
    #         print(f'Multiprocessing End:  {round((time.time() - start)/3600, 2)} hour')
    #         p.close()

    #         c=0
    #         for i, m in indices:
    #             print(i, m)
    #             total_indices_stayed[i].append(m)
    #             c+=1
    #         torch.save(total_indices_stayed, log_dir+f"/total_indices_stayed_{save_i}.pt")

    #     else:
    #         print("LOADING SAVED INEDX...mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmnmnmnmnnnnnn")
    #         total_indices_stayed = torch.load(saved_index)

    #     total_indices_stayed = self.get_total_top_indices(total_indices_stayed)
    #     print(total_indices_stayed)


    #     inputs, _ = next(iter(self.data_loader.train_loader))
    #     # if self.cuda:
    #     #     inputs = inputs.cuda(non_blocking=self.config.get("async_loading", True))
    #     self.record_conv_output(inputs)
    #     x = inputs
    #     for i, m in enumerate(self.all_list):
    #         if isinstance(m, models.resnet_cifar.BasicBlock) or isinstance(m, models.resnet_imagenet.BasicBlock):
    #             conv1 = self.named_modules_list[str(i)].conv1
    #             bn1 = self.named_modules_list[str(i)].bn1
    #             conv2 = self.named_modules_list[str(i)].conv2
    #             input_feature = torch.relu(bn1(conv1(x)))
                
    #             indices_stayed = total_indices_stayed[i]
                
    #             module_surgery(conv1, bn1, conv2, indices_stayed)
    #             pruned_input_feature = torch.relu(bn1(conv1(x)))
    #             output_feature = self.original_conv_output[str(i) + '.conv2'].cuda() if self.cuda else self.original_conv_output[str(i) + '.conv2']
    #             weight_reconstruction(conv2, pruned_input_feature, output_feature, use_gpu=self.cuda)
    #         elif isinstance(m, torch.nn.Linear):
    #             break
    #         x = m(x)
    #     self.original_conv_output = dict()  # clear original output to save cuda memory
        
    def get_indices(self, i, m, x):
        conv1 = self.named_modules_list[str(i)].conv1
        bn1 = self.named_modules_list[str(i)].bn1
        conv2 = self.named_modules_list[str(i)].conv2
        input_feature = torch.relu(bn1(conv1(x)))
        
        indices_stayed, indices_pruned = channel_selection(input_feature, conv2, sparsity=(1.-self.k), method='lasso')
        return indices_stayed

    def merge_pool(self, args):
        i, m, x = args[0], args[1], args[2]
        return i, self.get_indices(i, m, x)

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
