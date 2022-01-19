import os
import torch
import torchvision
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms, utils
from torchvision.transforms import (
    ToTensor, Lambda, Compose, Resize, RandomHorizontalFlip, RandomRotation)
from torchvision.transforms.transforms import RandomPerspective

def get_data_set(cfg_data):
    """
    Args: cfg_data

    output: dictionary that contains three datasets
            {"train": trainDataSet, "val": valDataSet, "test": testDataSet}
    """

    # cudnn.benchmark = True
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    aug_config = cfg_data["train"]["augmentation"]
    train_transforms = [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ]
    test_transforms = [
            transforms.ToTensor(),
            normalize,
        ]
    val_transforms = [
            transforms.ToTensor(),
            normalize,
        ]
    
    preprocess = {
        'train': Compose(train_transforms),
        'val': Compose(val_transforms),
        'test': Compose(test_transforms)
    }

    print("[ DATADIR ] ", cfg_data["dir"])

    if cfg_data["dir"] == "CIFAR10":
        imgsets = {'train': torchvision.datasets.CIFAR10(root="./data",train=True,transform=preprocess['train'],download=True),
                   'val': torchvision.datasets.CIFAR10(root="./data",train=False,transform=preprocess['val'],download=True),
                   'test': torchvision.datasets.CIFAR10(root="./data",train=False,transform=preprocess['test'],download=True)}
    elif cfg_data["dir"] == "CIFAR100":
        imgsets = {'train': torchvision.datasets.CIFAR100(root="./data",train=True,transform=preprocess['train'],download=True),
                   'val': torchvision.datasets.CIFAR100(root="./data",train=False,transform=preprocess['val'],download=True),
                   'test': torchvision.datasets.CIFAR100(root="./data",train=False,transform=preprocess['test'],download=True)}
    else:
        imgsets = {x: datasets.ImageFolder(os.path.join(cfg_data["dir"], x), preprocess[x])
                    for x in ['train', 'val', 'test']}

    n_class = len(imgsets['train'].classes)
    
    print("[ DATASET ]",end='')
    for x in ['train', 'val', 'test']:
        print(" [{}] number of classes:{}, size:{}".format(x, len(imgsets[x].classes), len(imgsets[x])), end='')
        if len(imgsets[x].classes) != n_class:
            raise ("[WARNING] n_class are different! Reformulate your dataset!")
    print()

    return imgsets, n_class


def get_data_loader(imgsets, cfg_data):
    """
    Args: datasets from getDataset, cfg_data

    output: dictionary that contains three dataloaders
            {"train": trainDataLoader, "val": valDataLoader, "test": testDataLoader}
    """
    imgloaders = {x: DataLoader(imgsets[x], 
                                batch_size = cfg_data[x]["batch_size"],
                                shuffle = cfg_data[x]["shuffle"],
                                num_workers = cfg_data[x]["n_workers"],
                                pin_memory=cfg_data[x]["pin_memory"])
                    for x in ['train','val','test']}
    return imgloaders

"""
Cifar100 Dataloader implementation
"""
import logging
import numpy as np
import copy
import random
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from torch.utils.data import DataLoader





class Cifar100DataLoader:
    def __init__(self, config):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.manual_seed(config.seed)
        torch.cuda.manual_seed(config.seed)
        random.seed(config.seed)
        np.random.seed(config.seed)
        self.config = config
        self.logger = logging.getLogger("Cifar100DataLoader")
        self.mean, self.std = [0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]

        if config.data_mode == "download":
            self.logger.info("Loading DATA.....")

            train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std)])

            valid_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std)])

            train_set = datasets.CIFAR100("./data", train=True, download=True, transform=train_transform)
            valid_set = datasets.CIFAR100("./data", train=False, transform=valid_transform)
        else:
            raise Exception("Please specify in the json a specified mode in data_mode")

        self.train_loader = DataLoader(train_set, batch_size=self.config.batch_size, shuffle=True,
                                       num_workers=self.config.data_loader_workers, pin_memory=self.config.pin_memory)
        self.valid_loader = DataLoader(valid_set, batch_size=self.config.batch_size, shuffle=False,
                                       num_workers=self.config.data_loader_workers, pin_memory=self.config.pin_memory)
        self.train_iterations = len(self.train_loader)
        self.valid_iterations = len(self.valid_loader)

    def finalize(self):
        pass


class SpecializedCifar100DataLoader:
    def __init__(self, config, subset_labels):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.manual_seed(config.seed)
        torch.cuda.manual_seed(config.seed)
        random.seed(config.seed)
        np.random.seed(config.seed)
        self.config = config
        self.logger = logging.getLogger("Cifar100DataLoader")
        self.mean, self.std = [0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]

        if config.data_mode == "download":
            self.logger.info("Loading DATA.....")

            train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std)])

            valid_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std)])

            binary_train_set = datasets.CIFAR100("./data", train=True, download=True, transform=train_transform)
            binary_valid_set = datasets.CIFAR100("./data", train=False, transform=valid_transform)

            train_set = copy.deepcopy(binary_train_set)

            mapping_table = dict()  # 0은 others labels (dustbin class), 그 후 subset label은 1부터 순서대로 mapping

            for new_label, label in enumerate(subset_labels):
                mapping_table[label] = new_label + 1


            train_subset_indices = [i for i, e in enumerate(binary_train_set.targets) if e in subset_labels] # 500 * 3 = 1500

            # change label to binary label
            _binary_train_set = [(x, mapping_table[y]) if y in subset_labels else (x, 0) for x, y in binary_train_set]

            binary_train_set = torch.utils.data.dataset.Subset(_binary_train_set, train_subset_indices )
            self.binary_train_loader = DataLoader(binary_train_set, batch_size=config.batch_size, shuffle=True)

            # make valid set for binary train
            valid_subset_indices = [i for i, e in enumerate(binary_valid_set.targets) if e in subset_labels] # subset_lables = [1,2,3] -- oth_cls = 97
            # valid_num_sample = len(valid_subset_indices) // len(oth_cls)  # 300 // 97 = 3
            # d = list()
            # for val in oth_cls: # oth_cls = 97
            #     d += list(np.where(binary_valid_set.targets == val)[0][:valid_num_sample])
            # vl_oth = d # 3 * 97 = 291
            _binary_valid_set = [(x, mapping_table[y]) if y in subset_labels else (x, 0) for x, y in binary_valid_set]
            sub_binary_valid_set = torch.utils.data.dataset.Subset(_binary_valid_set, valid_subset_indices )
            self.binary_valid_loader = DataLoader(sub_binary_valid_set, batch_size=config.batch_size)

            # set part train set
            _part_train_set = [(x, mapping_table[y]) if y in subset_labels else (x, 0) for x, y in train_set]
            part_train_set = torch.utils.data.dataset.Subset(_part_train_set, train_subset_indices)
            self.part_train_loader = DataLoader(part_train_set, batch_size=config.batch_size, shuffle=False)


            self.binary_train_iterations = len(self.binary_train_loader)
            self.binary_valid_iterations =len(self.binary_valid_loader)
            self.part_train_iterations = len(self.part_train_loader)

        else:
            raise Exception("Please specify in the json a specified mode in data_mode")

    def finalize(self):
        pass
