import torch

path = "experiments/pretrained/resnet56.pt"
sd = torch.load(path)
sd = sd['state_dict']
# sd["linear.weight"] = sd["fc.weight"]
# sd["linear.bias"] = sd["fc.bias"]
# del sd["fc.weight"]
# del sd["fc.bias"]
torch.save(sd, "experiments/pretrained/resnet56_1.pt")