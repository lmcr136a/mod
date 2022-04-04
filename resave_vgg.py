import torch


filenames = ["vgg16_bn_cifar10"]

for filename in filenames:
    path = f"models/pretrained/{filename}.pt"
    sd = torch.load(path)
    # sd = sd["state_dict"]
    # keylist = []
    
    n=0
    keys = list(sd.keys())
    for key in keys:
        if "conv" in key:
            n +=1
#     for key in keys:
#         if "features" in key:
#             new_key = key.replace("shortcut", "downsample")
#             sd[new_key] = sd[key]
#             del sd[key]
#             keys.remove(key)
#             print(f"changed the key : {key} -> {new_key}")
#         # if key in keylist:
#         #     print(f"removed the key : {key}")
#         #     del sd[key]

#         # if key in ["fc.weight", "fc.bias"]:
#         #     sd["linear.weight"] = sd["fc.weight"]
#         #     sd["linear.bias"] = sd["fc.bias"]
#         #     del sd["fc.weight"]
#         #     del sd["fc.bias"]
#         #     keys.remove("fc.weight")
#         #     keys.remove("fc.bias")
#         #     print(f"changed the key : fc -> linear")
#         if key in ["linear.weight", "linear.bias"]:
#             sd["fc.weight"] = sd["linear.weight"]
#             sd["fc.bias"] = sd["linear.bias"]
#             del sd["linear.weight"]
#             del sd["linear.bias"]
#             keys.remove("linear.weight")
#             keys.remove("linear.bias")
#             print(f"changed the key : linear -> fc")


#         if "module." in key:
#             sd[key[7:]] = sd[key]
#             del sd[key]
#             print(f"removed module. from the key : {key}")

#     # print(sd.keys())
    print(n)
    # torch.save(sd, f"models/pretrained/{filename}.pt")