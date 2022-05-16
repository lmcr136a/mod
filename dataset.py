"""
Cifar100 Dataloader implementation
"""
from ast import Raise
import logging
from pkgutil import get_data
import numpy as np
import copy
import random
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from torch.utils.data import DataLoader



def get_dataloader(cfg, for_test=False, get_only_targets=None):
    cfg_data = cfg["data"]

    if for_test:
        flag="for_test"
    else:
        flag="for_trainNval"

    if get_only_targets is None:
        get_only_targets = cfg_data["target_"+flag]

    target_classes=cfg_data.get("target_classes", None)

    print(flag, ":    ", cfg_data[flag])
    if cfg_data[flag] == "CIFAR10":
        return CifarDataLoader(cfg, 10, get_only_targets=get_only_targets, target_classes=target_classes), max(10, len(target_classes))
    elif cfg_data[flag] == "CIFAR100":
        return CifarDataLoader(cfg, 100, get_only_targets=get_only_targets, target_classes=target_classes), max(100, len(target_classes))
    elif cfg_data[flag] == "IMAGENET":
        return InversionDataLoader(cfg, 'imagenet', 1000, get_only_targets=get_only_targets, target_classes=target_classes, imagenet=True), max(1000, len(target_classes))
    elif cfg_data[flag] == "NINV10":
        return InversionDataLoader(cfg, 'ninv', 10, get_only_targets=get_only_targets, target_classes=target_classes), max(10, len(target_classes))
    elif cfg_data[flag] == "NINV100":
        return InversionDataLoader(cfg, 'ninv', 100, get_only_targets=get_only_targets, target_classes=target_classes), max(100, len(target_classes))
    elif cfg_data[flag] == "DINV10":
        return InversionDataLoader(cfg, 'dinv', 10, get_only_targets=get_only_targets, target_classes=target_classes), max(10, len(target_classes))
    elif cfg_data[flag] == "DINV1000":
        return InversionDataLoader(cfg, 'dinv', 1000, get_only_targets=get_only_targets, target_classes=target_classes), max(1000, len(target_classes))


def get_test_dataloader(cfg, dataloader, get_only_targets=True):
    cfg_data = cfg["data"]
    if cfg_data["for_test"] == cfg_data["for_trainNval"]\
        and cfg_data['target_for_test'] == cfg_data['target_for_trainNval']:
        pass
    else:
        test_loader, _ = get_dataloader(cfg, for_test=True, get_only_targets=get_only_targets)
        dataloader.valid_loader = test_loader.test_loader
        dataloader.test_loader = test_loader.test_loader


class CifarDataLoader:
    def __init__(self, config, class_num, get_only_targets, target_classes):
        config = config["run"]
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.manual_seed(config.get("seed", 9099))
        torch.cuda.manual_seed(config.get("seed", 9099))
        random.seed(config.get("seed", 9099))
        np.random.seed(config.get("seed", 9099))
        self.config = config
        if class_num == 1000:
            print(f"Imagenet DataLoader")
        else:
            print(f"Cifar{class_num} Dataloader")


        data_transformer = transforms.Compose([transforms.ToTensor()])
        if class_num == 10:
            train_set = datasets.CIFAR10("./data", train=True, download=True, transform=data_transformer)
            valid_set = datasets.CIFAR10("./data", train=False, transform=data_transformer)
            test_set = datasets.CIFAR10("./data", train=False, transform=data_transformer)
        elif class_num == 100:
            train_set = datasets.CIFAR100("./data", train=True, download=True, transform=data_transformer)
            valid_set = datasets.CIFAR100("./data", train=False, download=True, transform=data_transformer)
            test_set = datasets.CIFAR100("./data", train=False, transform=data_transformer)
        else:
            raise ValueError('check class num in cifar dataloader')

        # mean, std = [0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]
        tmean, tstd = [0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]
        vmean, vstd = [0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]
        testmean, teststd = [0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]

        # tmean, tstd = get_mean_std(train_set)
        # vmean, vstd = get_mean_std(valid_set)
        # testmean, teststd = get_mean_std(test_set)

        train_set.transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(tmean, tstd)])

        valid_set.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(vmean, vstd)])

        test_set.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(testmean, teststd)])


        if target_classes and get_only_targets:
            print("DATALOADER FOR CLASS: ", target_classes)
            for dataset in [train_set, valid_set, test_set]:
                idx = torch.Tensor([i for i, v in enumerate(dataset.targets) if v in target_classes]).long()
                dataset.targets = torch.Tensor(dataset.targets)[idx]
                dataset.data = torch.Tensor(dataset.data)[idx].cpu().detach().numpy().astype(np.uint8)
        else:
            print("DATALOADER FOR ALL CLASS")


        self.train_loader = DataLoader(train_set, batch_size=self.config.get("batch_size", 128), shuffle=True,
                                       num_workers=self.config.get("n_workers", 4), pin_memory=self.config.get("pin_memory", True))
        self.valid_loader = DataLoader(valid_set, batch_size=self.config.get("batch_size", 128), shuffle=True,
                                       num_workers=self.config.get("n_workers", 4), pin_memory=self.config.get("pin_memory", True))
        self.test_loader = DataLoader(test_set, batch_size=self.config.get("batch_size", 128), shuffle=True,
                                       num_workers=self.config.get("n_workers", 4), pin_memory=self.config.get("pin_memory", True))
        self.train_iterations = len(self.train_loader)
        self.valid_iterations = len(self.valid_loader)
        self.test_iterations = len(self.test_loader)



class InversionDataLoader:
    def __init__(self, config, dirname, class_num, get_only_targets, target_classes, imagenet=False):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.manual_seed(config["run"].get("seed", 9099))
        torch.cuda.manual_seed(config["run"].get("seed", 9099))
        random.seed(config["run"].get("seed", 9099))
        np.random.seed(config["run"].get("seed", 9099))
        self.config = config["run"]
        print(f"{dirname} cifar{class_num} dataloader")


        data_transformer = transforms.Compose([transforms.ToTensor()])
        
        if imagenet:
            train_set = datasets.ImageFolder("./data/imagenet/train", transform=data_transformer)
            valid_set = datasets.ImageFolder("./data/imagenet/val", transform=data_transformer)
            test_set = datasets.ImageFolder("./data/imagenet/test", transform=data_transformer)
        else:
            if "resnet" in config["network"]["model"]:
                data_path = f'./data/{dirname}/{config["data"]["for_test"].lower()}_r{config["network"]["model"][-2:]}/'
            else:
                pass
            train_set = datasets.ImageFolder(data_path+"train", transform=data_transformer)
            valid_set = datasets.ImageFolder(data_path+"train", transform=data_transformer)
            test_set = datasets.ImageFolder(data_path+"train", transform=data_transformer)

        if imagenet:
            tmean, tstd = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            vmean, vstd = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            testmean, teststd = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        else:
            tmean, tstd = get_mean_std(train_set)
            vmean, vstd = get_mean_std(valid_set)
            testmean, teststd = get_mean_std(test_set)

        if class_num == 1000:
            tfs = [transforms.Resize(256), transforms.CenterCrop(224)]
        else:
            tfs = [transforms.RandomCrop(32, padding=4)]
        
        # mean, std = [0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]
        # tmean, tstd, vmean, vstd, testmean, teststd = mean, std, mean, std, mean, std

        train_set.transform = transforms.Compose(tfs+[
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(tmean, tstd)])

        valid_set.transform = transforms.Compose(tfs+[
            transforms.ToTensor(),
            transforms.Normalize(vmean, vstd)])

        test_set.transform = transforms.Compose(tfs+[
            transforms.ToTensor(),
            transforms.Normalize(testmean, teststd)])


        if target_classes and get_only_targets:
            print("INV DATALOADER FOR CLASS: ", target_classes)
            for dataset in [train_set, valid_set, test_set]:
                idx = torch.Tensor([i for i, v in enumerate(dataset.targets) if v in target_classes]).long()
                dataset.targets = torch.Tensor(dataset.targets)[idx]
                dataset.imgs = [dataset.imgs[i] for i in idx]
                dataset.samples = [dataset.samples[i] for i in idx]
        else:
            print("INV DATALOADER FOR ALL CLASS")


        self.train_loader = DataLoader(train_set, batch_size=self.config.get("batch_size", 128), shuffle=True,
                                       num_workers=self.config.get("n_workers", 4), pin_memory=self.config.get("pin_memory", True))
        self.valid_loader = DataLoader(valid_set, batch_size=self.config.get("batch_size", 128), shuffle=True,
                                       num_workers=self.config.get("n_workers", 4), pin_memory=self.config.get("pin_memory", True))
        self.test_loader = DataLoader(test_set, batch_size=self.config.get("batch_size", 128), shuffle=True,
                                       num_workers=self.config.get("n_workers", 4), pin_memory=self.config.get("pin_memory", True))
        self.train_iterations = len(self.train_loader)
        self.valid_iterations = len(self.valid_loader)
        self.test_iterations = len(self.test_loader)



def get_mean_std(dataset):
    meanRGB = [np.mean(x.numpy(), axis=(1,2)) for x,_ in dataset]
    stdRGB = [np.std(x.numpy(), axis=(1,2)) for x,_ in dataset]

    meanR = np.mean([m[0] for m in meanRGB])
    meanG = np.mean([m[1] for m in meanRGB])
    meanB = np.mean([m[2] for m in meanRGB])
    stdR = np.mean([s[0] for s in stdRGB])
    stdG = np.mean([s[1] for s in stdRGB])
    stdB = np.mean([s[2] for s in stdRGB])
    return [meanR, meanG, meanB], [stdR, stdG, stdB]