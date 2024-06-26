"""
Contains functions for training and testing a PyTorch model.
"""

import torch
from tqdm.auto import tqdm
from typing import Dict, List, Tuple
from torchmetrics.classification import BinaryAccuracy
from torch.utils.tensorboard.writer import SummaryWriter


def train_step(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    """Trains a PyTorch model for a single epoch"""

    model.train()

    train_loss, train_acc = 0, 0

    metric = BinaryAccuracy().to(device)

    with tqdm(enumerate(dataloader), unit="batch", total=len(dataloader)) as tepoch:
        for batch, (X, y) in tepoch:
            X, y = X.to(device), y.to(device)

            y_logits = model(X)

            loss = loss_fn(y_logits, y)
            train_loss += loss.item()

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            train_acc += metric(y_logits, y)

            tepoch.set_postfix(accuracy=train_acc.item() / (batch + 1))

    # Adjust metrics to get average loss and accuracy per batch
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc


def test_step(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Tests a PyTorch model for a single epoch."""
    model.eval()

    test_loss, test_acc = 0, 0

    metric = BinaryAccuracy().to(device)

    with torch.inference_mode():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)

            test_pred_logits = model(X)

            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()

            test_acc += metric(test_pred_logits, y)

    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    return test_loss, test_acc


def train(
    model: torch.nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    test_dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    epochs: int,
    device: torch.device,
    writer: SummaryWriter,
) -> Dict[str, List]:
    """Trains and tests a PyTorch model."""

    results = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}

    for epoch in range(epochs):
        train_loss, train_acc = train_step(
            model=model,
            dataloader=train_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
        )
        test_loss, test_acc = test_step(
            model=model, dataloader=test_dataloader, loss_fn=loss_fn, device=device
        )

        print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"test_acc: {test_acc:.4f}"
        )

        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

        writer.add_scalars(
            main_tag="Loss",
            tag_scalar_dict={"train_loss": train_loss, "test_loss": test_loss},
            global_step=epoch,
        )

        # Add accuracy results to SummaryWriter
        writer.add_scalars(
            main_tag="Accuracy",
            tag_scalar_dict={"train_acc": train_acc, "test_acc": test_acc},
            global_step=epoch,
        )

        # Track the PyTorch model architecture
        writer.add_graph(
            model=model,
            # Pass in an example input
            input_to_model=torch.randn(32, 3, 224, 224).to(device),
        )

    writer.close()

    return results
