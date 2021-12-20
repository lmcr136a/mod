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
    resize = cfg_data["resize"]
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
