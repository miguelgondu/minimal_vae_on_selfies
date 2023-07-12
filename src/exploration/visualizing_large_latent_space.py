"""
In this script, we visualize high-dimensional latent
spaces by encoding molecules from the training and test
set into the high-dimensional latent space, and then
reducing their dimensionality to 2 dimensions using PCA,
t-SNE and UMAP.
"""
from typing import Union
from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler

from models.vae import VAESelfies

from utils.data import load_dataloaders
from utils.visualization import selfie_to_image, selfie_to_png
from utils.tokens import from_tensor_to_selfie

ROOT_DIR = Path(__file__).parent.parent.parent.resolve()


def plot_reduced_latent_space(model: VAESelfies, reduction: Union[PCA, TSNE] = PCA):
    """
    Plot the reduced latent space of the model
    by encoding (some of) the training and test set,
    and reducing their dimensionality using PCA, t-SNE
    or UMAP
    """
    np.random.seed(42)

    CLUSTER_FIG_PATH = ROOT_DIR / "data" / "figures" / "clusters"
    CLUSTER_FIG_PATH.mkdir(exist_ok=True, parents=True)

    # Load the test data as one-hot vectors
    _, test_loader = load_dataloaders(
        dataset_name=model.dataset_name, as_onehot=False, as_token_ids=True
    )

    subset_of_test = test_loader.dataset.tensors[0][
        np.random.choice(len(test_loader.dataset), size=5000, replace=False)
    ]
    subset_of_test_onehot = torch.nn.functional.one_hot(
        subset_of_test.to(torch.int64), num_classes=len(model.tokens_dict)
    ).type(torch.float)

    # Encode the test data
    q_z_given_x = model.encode(subset_of_test_onehot)
    mean = q_z_given_x.mean.detach().numpy()

    # Reduce the dimensionality of the latent space
    reduced = reduction(n_components=2).fit_transform(mean)

    # Scale the data
    # reduced = StandardScaler().fit_transform(reduced)

    # Fit the clustering
    # clustering = KMeans(n_clusters=50).fit(reduced)
    clustering = DBSCAN().fit(reduced)
    print(f"Number of clusters: {len(np.unique(clustering.labels_))}")

    # Plot the reduced latent space
    title = f"{reduction().__class__.__name__}"
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.scatter(reduced[:, 0], reduced[:, 1], s=8, c=clustering.labels_)
    ax.set_title(title)
    ax.axis("off")
    FIG_PATH = ROOT_DIR / "data" / "figures" / "reduced_latent_space"
    FIG_PATH.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        FIG_PATH / f"{title}.jpg",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    # Save images per cluster
    for cluster in np.unique(clustering.labels_):
        indices = np.where(clustering.labels_ == cluster)[0]
        selfies = from_tensor_to_selfie(
            subset_of_test_onehot[indices], model.tokens_dict
        )

        CLUSTER_FIG_PATH_C = CLUSTER_FIG_PATH / f"cluster_{cluster}"
        CLUSTER_FIG_PATH_C.mkdir(exist_ok=True, parents=True)

        for i, selfie in enumerate(selfies):
            image = selfie_to_image(selfie, strict=False)
            with open(CLUSTER_FIG_PATH_C / f"{i}.jpg", "wb") as fp:
                image.save(fp, format="JPEG", quality=100)


if __name__ == "__main__":
    # Load the model
    model = VAESelfies(dataset_name="SMALL-CID-SELFIES-20")
    model.load_state_dict(
        torch.load(
            ROOT_DIR
            / "data"
            / "trained_models"
            / "VAESelfies_SMALL-CID-SELFIES-20_latent_dim_64.pt",
            map_location=torch.device("cpu"),
        )
    )

    # Plot the reduced latent space
    plot_reduced_latent_space(model, reduction=TSNE)
