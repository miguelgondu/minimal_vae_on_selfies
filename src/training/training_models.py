"""
Implements a training loop for the models defined 
in src/models
"""

from pathlib import Path
from typing import Tuple, Union

import matplotlib.pyplot as plt

from torch.utils.data.dataloader import DataLoader
import torch

from utils.data import load_dataloaders

from models.ae import AutoencoderSelfies
from models.vae import VAESelfies

ROOT_DIR = Path(__file__).parent.parent.parent.resolve()
MODELS_DIR = ROOT_DIR / "data" / "trained_models"
MODELS_DIR.mkdir(exist_ok=True, parents=True)


def training_loop(
    model: Union[AutoencoderSelfies, VAESelfies],
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


def testing_loop(
    model: Union[AutoencoderSelfies, VAESelfies], testing_loader: DataLoader
) -> float:
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


def train_model(
    model: Union[AutoencoderSelfies, VAESelfies],
    max_epochs: int = 100,
    batch_size: int = 256,
    dataset_name: str = "TINY-CID-SELFIES",
    max_length: int = 50,
    latent_dim: int = 64,
    lr: float = 1e-3,
) -> Tuple[AutoencoderSelfies, float]:
    """
    Trains an autoencoder on the provided dataset.
    """
    # Define the training names for logging
    model_name = model.__class__.__name__

    # Define the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load the data
    training_loader, testing_loader = load_dataloaders(
        dataset_name=dataset_name, batch_size=batch_size, max_token_length=max_length
    )

    # Define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Run the training loop
    best_testing_loss = float("inf")

    # Saving the losses for plotting
    training_losses = []
    testing_losses = []

    for epoch in range(max_epochs):
        # Run a training epoch
        training_loss = training_loop(model, training_loader, optimizer)

        # Run a testing epoch
        testing_loss = testing_loop(model, testing_loader)

        # Save the losses for plotting
        training_losses.append(training_loss)
        testing_losses.append(testing_loss)

        # Log the results
        print(
            f"Epoch {epoch + 1}/{max_epochs}: Training loss: {training_loss:.4f}, Testing loss: {testing_loss:.4f}"
        )

        # Save the best model so far
        if epoch == 0 or testing_loss < best_testing_loss:
            best_testing_loss = testing_loss
            torch.save(
                model.state_dict(), MODELS_DIR / f"{model_name}_{dataset_name}.pt"
            )

    print("Best testing loss: ", best_testing_loss)

    # Plotting the losses
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    ax.plot(range(len(training_losses)), training_losses, label="Training loss")
    ax.plot(range(len(testing_losses)), testing_losses, label="Testing loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    fig.savefig("losses.jpg")


if __name__ == "__main__":
    # Define the model
    model = VAESelfies(
        dataset_name="TINY-CID-SELFIES-20",
        max_token_length=20,
        latent_dim=64,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )

    train_model(
        model=model,
        max_epochs=200,
        batch_size=256,
        dataset_name="TINY-CID-SELFIES-20",
        max_length=20,
        latent_dim=64,
        lr=1e-3,
    )
