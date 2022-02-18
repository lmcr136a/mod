from .resnet_cifar import resnet20, resnet32, resnet44, resnet56, resnet110
from .resnet_chip import resnet_56, resnet_110
from .resnet_imagenet import resnet18, resnet34, resnet50, resnet101, resnet152

def get_resnet(cfg_network, n_class):
    model_name = cfg_network["model"]
    print("[ NETWORK ] ", model_name)

    try:
        if cfg_network["pruning"]["chip"]:
            return {
                "resnet56": resnet_56(n_class),
                "resnet110": resnet_110(n_class),
                
                "resnet50": resnet50(n_class),
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
    except:
        raise (f"Model {model_name} not available")


def get_network(cfg_network, n_class):
    model = get_resnet(cfg_network, n_class)
    print("[MODEL] Number of parameters : ", sum(p.numel() for p in model.parameters() if p.requires_grad))
    return model