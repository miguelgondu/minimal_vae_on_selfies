from typing import Union
from pathlib import Path

import torch
import matplotlib.pyplot as plt

from utils.data.zinc_250k import load_zinc_250k_dataloaders
from training.training_models import training_loop, testing_loop
from models.vae import VAESelfies

ROOT_DIR = Path(__file__).parent.parent.parent.parent.resolve()
MODELS_DIR = ROOT_DIR / "data" / "trained_models"
MODELS_DIR.mkdir(exist_ok=True, parents=True)


def train_model(
    model: VAESelfies,
    max_epochs: int = 100,
    batch_size: int = 256,
    lr: float = 1e-3,
    overfit_to_a_single_batch: bool = False,
):
    # Defining the experiment's name
    experiment_name = "vae_on_zinc_250k"

    # Loading up the dataloaders
    train_loader, test_loader = load_zinc_250k_dataloaders(
        batch_size=batch_size, overfit_to_a_single_batch=overfit_to_a_single_batch
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
        training_loss = training_loop(model, train_loader, optimizer)

        # Run a testing epoch
        testing_loss = testing_loop(model, test_loader)

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
                model.state_dict(),
                MODELS_DIR / f"{experiment_name}_latent_dim_{model.latent_dim}.pt",
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
    model = VAESelfies(
        dataset_name="zinc_250k",
        max_token_length=70,
        latent_dim=2,
        device=torch.device("mps" if torch.backends.mps.is_available() else "cpu"),
    )

    train_model(
        model,
        max_epochs=1000,
        batch_size=64,
        overfit_to_a_single_batch=True,
    )
