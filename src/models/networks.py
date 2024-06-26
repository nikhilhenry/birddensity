import torch
from torch import nn


class TwoHiddenLayerClassifier(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape=1):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(in_features=input_shape, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=1),
        )

    def forward(self, x: torch.Tensor):
        return self.model(x)
