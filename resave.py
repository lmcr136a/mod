import torch


filename = "resnet56_cifar10"

path = f"models/pretrained/{filename}.pt"
sd = torch.load(path)
# sd = sd['state_dict']

keylist = ["layer2.0.downsample.0.weight", "layer2.0.downsample.1.weight", "layer2.0.downsample.1.bias", "layer2.0.downsample.1.running_mean", "layer2.0.downsample.1.running_var", "layer2.0.downsample.1.num_batches_tracked", "layer3.0.downsample.0.weight", "layer3.0.downsample.1.weight", "layer3.0.downsample.1.bias", "layer3.0.downsample.1.running_mean", "layer3.0.downsample.1.running_var", "layer3.0.downsample.1.num_batches_tracked"]

keys = list(sd.keys())
for key in keys:
    if key in keylist:
        print(f"removed the key : {key}")
        del sd[key]

    if key in ["fc.weight", "fc.bias"]:
        sd["linear.weight"] = sd["fc.weight"]
        sd["linear.bias"] = sd["fc.bias"]
        del sd["fc.weight"]
        del sd["fc.bias"]
        keys.remove("fc.weight")
        keys.remove("fc.bias")
        print(f"changed the key : fc -> linear")


    if "module." in key:
        sd[key[7:]] = sd[key]
        del sd[key]
        print(f"removed module. from the key : {key}")

# print(sd.keys())
torch.save(sd, f"models/pretrained/{filename}.pt")