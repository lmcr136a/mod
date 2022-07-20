from .resnet_cifar import resnet20, resnet32, resnet44, resnet56, resnet110
from .resnet_chip_imagenet import resnet50_chip, resnet34_chip
from .resnet_hrank_imagenet import resnet34_hrank, resnet50_hrank
from .resnet_imagenet import resnet18, resnet34, resnet50, resnet101
from .resnet_chip import resnet56_chip
from .resnet_hrank import resnet56_hrank
from .resnet_gal import resnet56_gal
from .vgg_chip import vgg_16_bn_chip
from .vgg_gal import vgg_16_bn_gal
from .vgg_hrank import vgg_16_bn_hrank
# from .resnet_gal_imagenet import resnet34_gal, resnet50_gal
import torchvision.models as models
import torch
def get_resnet(cfg_network, n_class, sparsity, target_classes):
    model_name = cfg_network["model"]

    if cfg_network.get("pruning", False) and (cfg_network["pruning"].get("chip", False) or cfg_network["pruning"].get("hrank", False) or cfg_network["pruning"].get("gal", False)) :
        if cfg_network["pruning"].get("gal", None):
            model_name = model_name +"_gal"
            print("[ NETWORK ] ", model_name)
            return eval(model_name)(num_classes=n_class)

        elif cfg_network["pruning"].get("hrank", None):
            model_name = model_name +"_hrank"
            print("[ NETWORK ] ", model_name)
            return eval(model_name)(num_classes=n_class, compress_rate=sparsity)

        elif cfg_network["pruning"].get("chip", None):
            model_name = model_name +"_chip"
            print("[ NETWORK ] ", model_name)
            return eval(model_name)(num_classes=n_class, sparsity=sparsity)

    # if  cfg_network['load_state'] and cfg_network['load_state'] == 'from pytorch':
    #     return models.resnet34(pretrained=True)
    return {
        "resnet20": resnet20(n_class, target_classes),
        "resnet32": resnet32(n_class, target_classes),
        "resnet44": resnet44(n_class, target_classes),
        "resnet56": resnet56(n_class, target_classes),
        "resnet110": resnet110(n_class, target_classes),
        
        "resnet18": resnet18(n_class, target_classes),
        "resnet34": resnet34(n_class, target_classes),
        "resnet50": resnet50(n_class, target_classes),
        "resnet101": resnet101(n_class, target_classes),
    }[model_name]


def get_network(cfg_network, n_class, sparsity=None, target_classes=None):
    if 'resnet' in cfg_network['model']:
        model = get_resnet(cfg_network, n_class, sparsity, target_classes)
    elif 'vgg' in cfg_network['model']:
        if cfg_network['load_state'] == 'from pytorch':
            print("load state dict from pytorch")
            model = models.vgg16_bn(pretrained=True)
        else:
            model = vgg_16_bn_chip(num_classes=n_class, sparsity=0, target_classes=target_classes)


    print("[ NETWORK ] ", cfg_network['model'])
    print("[MODEL] Number of parameters : ", sum(p.numel() for p in model.parameters() if p.requires_grad))
    return model
