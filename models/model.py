from .resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from skills.texture import GaborLayerLearnable


def get_resnet(cfg_network, n_class):
    name = cfg_network["backbone"]
    n_cv = cfg_network["n_cv"]
    
    print("[ NETWORK ]",end='')
    for cfg in cfg_network:
        print(" [{}] {}".format(cfg, cfg_network[cfg]),end='')
    print()
    
    try:
        return {
            "resnet18": resnet18(n_class, n_cv),
            "resnet34": resnet34(n_class, n_cv),
            "resnet50": resnet50(n_class, n_cv),
            "resnet101": resnet101(n_class, n_cv),
            "resnet152": resnet152(n_class, n_cv),
        }[name]
    except:
        raise (f"Model {name} not available")


def get_network(cfg_network, n_class):
    resnet = get_resnet(cfg_network, n_class)
    if cfg_network["gabor"]:
        resnet.conv1 = GaborLayerLearnable(
            in_channels=resnet.first_layer_input_size, 
            out_channels=resnet.first_layer_output_size,
            stride=resnet.first_layer_stride, 
            padding=resnet.first_layer_padding, 
            kernel_size=resnet.first_layer_kernel_size,
        )
    return resnet