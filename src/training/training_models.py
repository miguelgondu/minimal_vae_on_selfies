"""
Implements a training loop for the models defined 
in src/models
"""

from pathlib import Path
from typing import Tuple, Union

import matplotlib.pyplot as plt

from torch.utils.data.dataloader import DataLoader
import torch

from tqdm import tqdm

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
    for (batch,) in tqdm(training_loader):
        # The batch contains the inputs to the sequence as token ids,
        # so we need to transform them to one-hot encodings.
        batch = torch.nn.functional.one_hot(
            batch.to(torch.int64),
            num_classes=len(model.tokens_dict),
        ).to(torch.get_default_dtype())

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
        for (batch,) in tqdm(testing_loader):
            # The batch contains the inputs to the sequence as token ids,
            # so we need to transform them to one-hot encodings.
            batch = torch.nn.functional.one_hot(
                batch.to(torch.int64),
                num_classes=len(model.tokens_dict),
            ).to(torch.get_default_dtype())

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
    max_token_length: int = 50,
    lr: float = 1e-3,
    early_stopping: bool = True,
    max_patience: int = 15,
) -> Tuple[Union[AutoencoderSelfies, VAESelfies], float]:
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
        dataset_name=dataset_name,
        batch_size=batch_size,
        max_token_length=max_token_length,
        as_onehot=False,
        as_token_ids=True,
        device=device,
    )

    # Define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Run the training loop
    best_testing_loss = float("inf")

    # Saving the losses for plotting
    training_losses = []
    testing_losses = []

    # Sets up patience for early stopping
    patience = 0

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
                model.state_dict(),
                MODELS_DIR
                / f"{model_name}_{dataset_name}_latent_dim_{model.latent_dim}.pt",
            )

        # Early stopping
        if early_stopping:
            if testing_loss > best_testing_loss:
                patience += 1
                if patience >= max_patience:
                    print("Stopping early!")
                    break
            else:
                patience = 0

        print("Best testing loss: ", best_testing_loss)

        # Plotting the losses
        fig, ax = plt.subplots(1, 1, figsize=(7, 7))
        ax.plot(range(len(training_losses)), training_losses, label="Training loss")
        ax.plot(range(len(testing_losses)), testing_losses, label="Testing loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
        fig.savefig(f"losses_{dataset_name}_{model.latent_dim}_{batch_size}.jpg")
        plt.close(fig)


if __name__ == "__main__":
    # Hyperparameters for this training
    dataset_name = "SMALL-CID-SELFIES-20"
    max_token_length = int(dataset_name.split("-")[-1])
    latent_dim = 64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 1024
    lr = 1e-3
    max_epochs = 200

    print("Training on device: ", device)

    # Define the model
    model = VAESelfies(
        dataset_name=dataset_name,
        max_token_length=max_token_length,
        latent_dim=latent_dim,
        device=device,
    )

    train_model(
        model=model,
        max_epochs=max_epochs,
        batch_size=batch_size,
        dataset_name=dataset_name,
        max_token_length=max_token_length,
        lr=lr,
        early_stopping=True,
        max_patience=15,
    )
