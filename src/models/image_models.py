import torch
import torchvision
from torch import nn


class PretrainedEfficienNetBO(nn.Module):
    def __init__(self, frozen=True):
        super().__init__()
        weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
        self.model = torchvision.models.efficientnet_b0(weights=weights)
        self.model.classifier = nn.Sequential()
        self.out_features = 1280

        # freeze the weights
        if frozen:
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor):
        return self.model(x)


class PretrainedResNet50(nn.Module):
    def __init__(self, frozen=True):
        super().__init__()
        weights = torchvision.models.ResNet50_Weights.DEFAULT
        self.model = torchvision.models.resnet50(weights=weights)
        self.model.classifier = nn.Sequential()
        self.out_features = 1000
        self.preprocess = weights.transforms()

        # freeze the weights
        if frozen:
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor):
        return self.model(x)
