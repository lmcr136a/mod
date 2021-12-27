from .resnet import resnet20, resnet32, resnet44, resnet56, resnet110


def get_resnet(cfg_network, n_class):
    name = cfg_network["backbone"]
    whether_prune = cfg_network["prune"]
    
    print("[ NETWORK ]",end='')
    for cfg in cfg_network:
        print(" [{}] {}".format(cfg, cfg_network[cfg]),end='')
    print()
    
    try:
        return {
            "resnet20": resnet20(n_class, whether_prune),
            "resnet32": resnet32(n_class, whether_prune),
            "resnet44": resnet44(n_class, whether_prune),
            "resnet56": resnet56(n_class, whether_prune),
            "resnet110": resnet110(n_class, whether_prune),
        }[name]
    except:
        raise (f"Model {name} not available")


def get_network(cfg_network, n_class):
    model = get_resnet(cfg_network, n_class)
    print("[MODEL] Number of parameters : ", sum(p.numel() for p in model.parameters() if p.requires_grad))
    return model