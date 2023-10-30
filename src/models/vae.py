"""
Implements a Variational Autoencoder that can be trained on
SELFIES data. It is originally designed to handle the
TINY-CID-SELFIES-20 datasets, but it could be used on the 
SMALL-CID-SELFIES-20 with a little bit of work.
"""
from typing import Tuple, Dict
from pathlib import Path
from itertools import product
import json

import numpy as np

import torch
import torch.nn as nn

from torch.distributions import Normal, Categorical

from utils.visualization import selfie_to_numpy_image_array

ROOT_DIR = Path(__file__).parent.parent.parent.resolve()


class VAESelfies(nn.Module):
    def __init__(
        self,
        dataset_name: str = "TINY-CID-SELFIES-20",
        max_token_length: int = 20,
        latent_dim: int = 64,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        super().__init__()
        self.dataset_name = dataset_name
        self.max_token_length = max_token_length
        self.latent_dim = latent_dim

        # assert max_token_length == int(dataset_name.split("-")[-1]), (
        #     f"max_token_length ({max_token_length}) must match the "
        #     f"dataset_name's token size ({dataset_name})!"
        # )

        self.device = device

        # Load the token dictionary
        with open(
            ROOT_DIR / "data" / "processed" / f"tokens_{dataset_name}.json", "r"
        ) as fp:
            self.tokens_dict = json.load(fp)

        # Define the input length: length of a given SELFIES
        # (always padded to be {max_length}), times the number of tokens
        self.input_length = max_token_length * len(self.tokens_dict)

        # Define the model
        self.encoder = nn.Sequential(
            nn.Linear(self.input_length, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        )
        self.encoder_mu = nn.Linear(256, latent_dim)
        self.encoder_log_var = nn.Linear(256, latent_dim)

        # The decoder, which outputs the logits of the categorical
        # distribution over the vocabulary.
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, self.input_length),
        )

        # Defines the prior
        self.p_z = Normal(
            loc=torch.zeros(latent_dim, device=device),
            scale=torch.ones(latent_dim, device=device),
        )

        # Moves to device
        self.to(device)

    def encode(self, x: torch.Tensor) -> Normal:
        """
        Computes the approximate posterior q(z|x) over
        the latent variable z.
        """
        hidden = self.encoder(x.flatten(start_dim=1).to(self.device))
        mu = self.encoder_mu(hidden)
        log_var = self.encoder_log_var(hidden)

        return Normal(loc=mu, scale=torch.exp(0.5 * log_var))

    def decode(self, z: torch.Tensor) -> Categorical:
        """
        Returns a categorical likelihood over the vocabulary
        """
        logits = self.decoder(z.to(self.device))

        # The categorical distribution expects (batch_size, ..., num_classes)
        return Categorical(
            logits=logits.reshape(-1, self.max_token_length, len(self.tokens_dict))
        )

    def forward(self, x: torch.Tensor) -> Tuple[Normal, Categorical]:
        """
        Computes a forward pass through the VAE, returning
        the distributions q_z_given_x and p_x_given_z.
        """
        q_z_given_x = self.encode(x)
        z = q_z_given_x.rsample()

        p_x_given_z = self.decode(z)

        return q_z_given_x, p_x_given_z

    def loss_function(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the ELBO loss for a given batch {x}.
        """
        q_z_given_x, p_x_given_z = self.forward(x)

        # Computes the KL divergence between q(z|x) and p(z)
        kl_div = torch.distributions.kl_divergence(q_z_given_x, self.p_z).sum(dim=-1)

        # Computes the reconstruction loss
        recon_loss = -p_x_given_z.log_prob(x.argmax(dim=-1).to(self.device)).sum(dim=-1)

        # Computes the ELBO loss
        return (kl_div + recon_loss).mean()

    def plot_grid(
        self,
        x_lims=(-5, 5),
        y_lims=(-5, 5),
        n_rows=10,
        n_cols=10,
        sample=False,
        ax=None,
    ) -> np.ndarray:
        """
        A helper function which plots, as images, the levels in a
        fine grid in latent space, specified by the provided limits,
        number of rows and number of columns.

        The figure can be plotted in a given axis; if none is passed,
        a new figure is created.

        This function also returns the final image (which is the result
        of concatenating all the individual decoded images) as a numpy
        array.
        """
        img_width_and_height = 200
        z1 = np.linspace(*x_lims, n_cols)
        z2 = np.linspace(*y_lims, n_rows)

        zs = np.array([[a, b] for a, b in product(z1, z2)])

        selfies_dist = self.decode(torch.from_numpy(zs).type(torch.float))
        if sample:
            selfies_as_ints = selfies_dist.sample()
        else:
            selfies_as_ints = selfies_dist.probs.argmax(dim=-1)

        inverse_alphabet: Dict[int, str] = {v: k for k, v in self.tokens_dict.items()}
        selfies_strings = [
            "".join([inverse_alphabet[i] for i in row])
            for row in selfies_as_ints.numpy(force=True)
        ]

        selfies_as_images = np.array(
            [
                selfie_to_numpy_image_array(
                    selfie, width=img_width_and_height, height=img_width_and_height
                )
                for selfie in selfies_strings
            ]
        )
        img_dict = {(z[0], z[1]): img for z, img in zip(zs, selfies_as_images)}

        positions = {
            (x, y): (i, j) for j, x in enumerate(z1) for i, y in enumerate(reversed(z2))
        }

        pixels = img_width_and_height
        final_img = np.zeros((n_cols * pixels, n_rows * pixels, 3))
        for z, (i, j) in positions.items():
            final_img[
                i * pixels : (i + 1) * pixels, j * pixels : (j + 1) * pixels
            ] = img_dict[z]

        final_img = final_img.astype(int)

        if ax is not None:
            ax.imshow(final_img, extent=[*x_lims, *y_lims])

        return final_img
