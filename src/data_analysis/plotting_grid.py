"""
This script visualizes the latent space of the VAE model, by plotting
the latent space of the model on a grid. This functionality is already
implemented inside the vae model itself.
"""
from pathlib import Path

import torch

import matplotlib.pyplot as plt

from models.vae import VAESelfies


ROOT_DIR = Path(__file__).parent.parent.parent.resolve()

if __name__ == "__main__":
    _, ax = plt.subplots(1, 1, figsize=(10, 10))

    # Load the model
    model = VAESelfies(
        dataset_name="TINY-CID-SELFIES-20",
        max_token_length=20,
        latent_dim=2,
        device=torch.device("mps" if torch.backends.mps.is_available() else "cpu"),
    )
    model.load_state_dict(
        torch.load(
            ROOT_DIR
            / "data"
            / "trained_models"
            / "VAESelfies_TINY-CID-SELFIES-20_latent_dim_2.pt"
        )
    )
    model.eval()

    # Plotting a grid in latent space.
    x_lims = y_lims = (-3, 3)
    model.plot_grid(x_lims=x_lims, y_lims=y_lims, ax=ax)
    ax.axis("off")

    plt.show()
