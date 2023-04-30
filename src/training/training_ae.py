"""
Implements a training loop for the models defined 
in src/models
"""

from pathlib import Path
from typing import Tuple

from torch.utils.data.dataloader import DataLoader
import torch
from src.models.ae import ROOT_DIR

from utils.data import load_dataloaders

from models.ae import Autoencoder

ROOT_DIR = Path(__file__).parent.parent.parent.resolve()
MODELS_DIR = ROOT_DIR / "data" / "trained_models"
MODELS_DIR.mkdir(exist_ok=True, parents=True)


def training_loop(
    model: Autoencoder,
    training_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
) -> float:
    """
    Runs an epoch of training, returing the average training loss.
    """
    losses = []
    for (batch,) in training_loader:
        # Reset the gradients
        optimizer.zero_grad()

        # Compute the loss (forward pass is inside)
        loss = model.loss_function(batch)

        # Run a backward pass
        loss.backward()

        # Update the parameters
        optimizer.step()

        # Append for logging
        losses.append(loss.item())

    return sum(losses) / len(losses)


def testing_loop(model: Autoencoder, testing_loader: DataLoader) -> float:
    """
    Runs an epoch of testing, returing the average testing loss.
    """
    losses = []
    with torch.no_grad():
        for (batch,) in testing_loader:
            # Compute the loss (forward pass is inside)
            loss = model.loss_function(batch)

            # Append for logging
            losses.append(loss.item())

    return sum(losses) / len(losses)


def train_autoencoder(
    max_epochs: int = 100,
    batch_size: int = 256,
    dataset_name: str = "SUPER-SMALL-CID-SELFIES",
    max_length: int = 300,
    latent_dim: int = 64,
    lr: float = 1e-3,
) -> Tuple[Autoencoder, float]:
    """
    Trains an autoencoder on the provided dataset.
    """
    # Define the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Autoencoder(
        dataset_name=dataset_name,
        max_length=max_length,
        latent_dim=latent_dim,
        device=device,
    )

    # Load the data
    training_loader, testing_loader = load_dataloaders(
        dataset_name=dataset_name, batch_size=batch_size, max_length=max_length
    )

    # Define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Run the training loop
    best_testing_loss = float("inf")
    for epoch in range(max_epochs):
        # Run a training epoch
        training_loss = training_loop(model, training_loader, optimizer)

        # Run a testing epoch
        testing_loss = testing_loop(model, testing_loader)

        # Log the results
        print(
            f"Epoch {epoch + 1}/{max_epochs}: Training loss: {training_loss:.4f}, Testing loss: {testing_loss:.4f}"
        )

        # Save the best model so far
        if epoch == 0 or testing_loss < best_testing_loss:
            best_testing_loss = testing_loss
            torch.save(model.state_dict(), MODELS_DIR / f"ae_{dataset_name}.pt")


if __name__ == "__main__":
    train_autoencoder()
