import torch


filenames = ["vgg16_bn_cifar100_f12"]

for filename in filenames:
    path = f"models/pretrained/{filename}.pt"
    sd = torch.load(path)
    # sd = sd['state_dict']
    keylist = []
    # keylist = ["layer2.0.downsample.0.weight", "layer2.0.downsample.1.weight", "layer2.0.downsample.1.bias", "layer2.0.downsample.1.running_mean", "layer2.0.downsample.1.running_var", "layer2.0.downsample.1.num_batches_tracked", "layer3.0.downsample.0.weight", "layer3.0.downsample.1.weight", "layer3.0.downsample.1.bias", "layer3.0.downsample.1.running_mean", "layer3.0.downsample.1.running_var", "layer3.0.downsample.1.num_batches_tracked"]
    # keylist += ["layer2.0.shortcut.0.weight", "layer2.0.shortcut.1.weight", "layer2.0.shortcut.1.bias", "layer2.0.shortcut.1.running_mean", "layer2.0.shortcut.1.running_var", "layer2.0.shortcut.1.num_batches_tracked", "layer3.0.shortcut.0.weight", "layer3.0.shortcut.1.weight", "layer3.0.shortcut.1.bias", "layer3.0.shortcut.1.running_mean", "layer3.0.shortcut.1.running_var", "layer3.0.shortcut.1.num_batches_tracked", "layer4.0.shortcut.0.weight", "layer4.0.shortcut.1.weight", "layer4.0.shortcut.1.bias", "layer4.0.shortcut.1.running_mean", "layer4.0.shortcut.1.running_var", "layer4.0.shortcut.1.num_batches_tracked"]
    keys = list(sd.keys())
    for key in keys:
        if "features" in key and "features1" not in key and 'features2' not in key:
            print(key)
            if int(key.split(".")[1][4:]) < 7:
                new_key = key.replace("features", "features1")
                print(1)
            else:
                new_key = key.replace("features", "features2")
                print(2)
            # new_key = key.replace("features1", "features1")
            sd[new_key] = sd[key]
            del sd[key]
            keys.remove(key)
            print(f"changed the key : {key} -> {new_key}")
        # if key in keylist:
        #     print(f"removed the key : {key}")
        #     del sd[key]

        # if key in ["fc.weight", "fc.bias"]:
        #     sd["linear.weight"] = sd["fc.weight"]
        #     sd["linear.bias"] = sd["fc.bias"]
        #     del sd["fc.weight"]
        #     del sd["fc.bias"]
        #     keys.remove("fc.weight")
        #     keys.remove("fc.bias")
        #     print(f"changed the key : fc -> linear")
        # if key in ["linear.weight", "linear.bias"]:
        #     sd["fc.weight"] = sd["linear.weight"]
        #     sd["fc.bias"] = sd["linear.bias"]
        #     del sd["linear.weight"]
        #     del sd["linear.bias"]
        #     keys.remove("linear.weight")
        #     keys.remove("linear.bias")
        #     print(f"changed the key : linear -> fc")


        # if "module." in key:
        #     sd[key[7:]] = sd[key]
        #     del sd[key]
        #     print(f"removed module. from the key : {key}")

    # print(sd.keys())
    torch.save(sd, f"models/pretrained/vgg16_bn_cifar100_f12.pt")