from .resnet_cifar import resnet20, resnet32, resnet44, resnet56, resnet110
from .resnet_chip import resnet56_chip, resnet110_chip
from .resnet_imagenet import resnet18, resnet34, resnet50, resnet101, resnet152

def get_resnet(cfg_network, n_class, sparsity):
    model_name = cfg_network["model"]
    print("[ NETWORK ] ", model_name)

    if cfg_network.get("pruning", False) and (cfg_network["pruning"].get("chip", False) or cfg_network["pruning"].get("hrank", False)) :
        return {
            "resnet56": resnet56_chip(n_class, sparsity=sparsity),
            "resnet110": resnet56_chip(n_class, sparsity=sparsity),

        }[model_name]
    return {
        "resnet20": resnet20(n_class),
        "resnet32": resnet32(n_class),
        "resnet44": resnet44(n_class),
        "resnet56": resnet56(n_class),
        "resnet110": resnet110(n_class),
        
        "resnet18": resnet18(n_class),
        "resnet34": resnet34(n_class),
        "resnet50": resnet50(n_class),
        "resnet101": resnet101(n_class),
        "resnet152": resnet152(n_class),
    }[model_name]


def get_network(cfg_network, n_class, sparsity=[0.]*100):
    model = get_resnet(cfg_network, n_class, sparsity)
    print("[MODEL] Number of parameters : ", sum(p.numel() for p in model.parameters() if p.requires_grad))
    return model