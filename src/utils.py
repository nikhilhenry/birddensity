"""
Contains various utility functions for PyTorch model training and saving.
"""

import torch
from pathlib import Path
from torch.utils.tensorboard.writer import SummaryWriter
from datetime import datetime
import os


def save_model(model: torch.nn.Module, target_dir: str, model_name: str):
    """Saves a PyTorch model to a target directory."""
    # Create target directory
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True, exist_ok=True)

    # Create model save path
    assert model_name.endswith(".pth") or model_name.endswith(
        ".pt"
    ), "model_name should end with '.pt' or '.pth'"
    model_save_path = target_dir_path / model_name

    # Save the model state_dict()
    print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(obj=model.state_dict(), f=model_save_path)


def create_writer(
    experiment_name: str, model_name: str
) -> torch.utils.tensorboard.writer.SummaryWriter():

    timestamp = datetime.now().strftime("%Y-%m-%d")

    log_dir = os.path.join("experiments", timestamp, experiment_name, model_name)

    print(f"[INFO] Created SummaryWriter, saving to: {log_dir}...")
    return SummaryWriter(log_dir=log_dir)
