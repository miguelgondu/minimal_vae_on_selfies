"""
In this script we visualize the 2-dimensional latent space
of the model trained on TINY-CID-SELFIES-20. The core output
of this script is an image of the latent space, where we plot
a grid of molecules in latent space.
"""
from pathlib import Path
from itertools import product
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch

from models.vae import VAESelfies

from utils.visualization import selfie_to_image
from utils.tokens import from_tensor_to_selfie

ROOT_DIR = Path(__file__).parent.parent.parent.resolve()
DATA_DIR = ROOT_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"


def plot_grid(
    model: VAESelfies,
    x_lims=(-5, 5),
    y_lims=(-5, 5),
    n_rows=10,
    n_cols=10,
    ax=None,
    strict: bool = False,
):
    z1 = np.linspace(*x_lims, n_cols)
    z2 = np.linspace(*y_lims, n_rows)

    zs = np.array([[a, b] for a, b in product(z1, z2)])

    selfies_dist = model.decode(torch.from_numpy(zs).type(torch.float))

    selfies = from_tensor_to_selfie(selfies_dist.probs, model.tokens_dict)
    images = [selfie_to_image(selfie, strict=strict) for selfie in selfies]
    img_dict = {(z[0], z[1]): img for z, img in zip(zs, images)}

    positions = {
        (x, y): (i, j) for j, x in enumerate(z1) for i, y in enumerate(reversed(z2))
    }

    pixels = 200
    final_img = np.zeros((n_cols * pixels, n_rows * pixels, 3))
    for z, (i, j) in positions.items():
        final_img[
            i * pixels : (i + 1) * pixels, j * pixels : (j + 1) * pixels
        ] = img_dict[z]

    final_img = final_img.astype(int)

    if ax is not None:
        ax.imshow(final_img, extent=[*x_lims, *y_lims])

    return final_img


if __name__ == "__main__":
    # Some hyperparameters for the visualization
    n_rows = 20
    n_cols = 20
    x_lims = (-2.5, 2.5)
    y_lims = (-2.5, 2.5)

    dataset_name = "TINY-TAIL-CID-SELFIES-20"

    # Load the model
    model = VAESelfies(
        dataset_name=dataset_name,
        max_token_length=20,
        latent_dim=2,
        device=torch.device("cpu"),
    )
    model.load_state_dict(
        torch.load(
            ROOT_DIR
            / "data"
            / "trained_models"
            / f"VAESelfies_{dataset_name}_latent_dim_2.pt",
            map_location=torch.device("cpu"),
        )
    )

    fig, ax = plt.subplots(1, 1, figsize=(n_rows, n_cols))
    plot_grid(
        model,
        x_lims=x_lims,
        y_lims=y_lims,
        n_rows=n_rows,
        n_cols=n_cols,
        ax=ax,
        strict=True,
    )
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(
        ROOT_DIR
        / "data"
        / "figures"
        / f"latent_space_{dataset_name}_dim_2_{n_rows}_x_{n_cols}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()
