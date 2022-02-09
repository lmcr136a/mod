from .resnet_cifar import resnet20, resnet32, resnet44, resnet56, resnet110
from .resnet_imagenet import resnet18, resnet34, resnet50, resnet101, resnet152

def get_resnet(model_name, n_class):
    
    print("[ NETWORK ] ", model_name)
    try:
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


def get_network(model_name, n_class):
    model = get_resnet(model_name, n_class)
    print("[MODEL] Number of parameters : ", sum(p.numel() for p in model.parameters() if p.requires_grad))
    return model