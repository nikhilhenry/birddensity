import torch
from torch import nn


def get_prediction_probs(x: torch.Tensor):
    return nn.functional.sigmoid(x)


def get_predictions(probs: torch.Tensor) -> list[int]:
    probs = probs.cpu()
    return [1 if prob >= 0.5 else 0 for prob in probs]
