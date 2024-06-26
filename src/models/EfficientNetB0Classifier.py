import torch
from torch import nn
from image_models import PretrainedEfficienNetBO
from networks import TwoHiddenLayerClassifier


class EfficienetBOClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = PretrainedEfficienNetBO()
        classifier_in_features = self.features.out_features
        self.classifier = TwoHiddenLayerClassifier(
            input_shape=classifier_in_features, hidden_units=500
        )

        self.model = nn.Sequential(self.features, self.classifier)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
