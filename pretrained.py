import torch
model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)

sd = model.state_dict()


torch.save(sd, f"models/pretrained/resnet50_imade.pt")