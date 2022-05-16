import torchvision.models as models
import torch

network = models.resnet34(pretrained=True)


torch.save(network.state_dict(), "resnet34_imagenet.pt")