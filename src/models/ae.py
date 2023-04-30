"""
Implements a simple autoencoder that decodes
the logits of a categorical distribution over
the vocabulary (as defined by a certain
token_{dataset_name}.json file)
"""
from pathlib import Path
import json

import torch

ROOT_DIR = Path(__file__).parent.parent.parent.resolve()


class Autoencoder(torch.nn.Module):
    def __init__(
        self,
        database_name: str = "SUPER-SMALL-CID-SELFIES",
        max_length: int = 300,
        latent_dim: int = 64,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        self.database_name = database_name
        self.device = device

        # Load the token dictionary
        with open(
            ROOT_DIR / "data" / "processed" / f"tokens_{database_name}.json", "r"
        ) as fp:
            self.token_dict = json.load(fp)

        # Define the input length: length of a given SELFIES
        # (always padded to be {max_length}), times the number of tokens
        self.input_length = max_length * len(self.token_dict)

        # Define the model
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(self.input_length, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, latent_dim),
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, self.input_length),
        )

        self.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Takes a batch of SELFIES, encoded as one-hot vectors
        and runs a forward pass through the autoencoder.

        Input:
            x: torch.Tensor of shape (batch_size, length_of_sequence, n_tokens).

        Output:
            logits_rec: torch.Tensor of shape (batch_size, length_of_sequence, n_tokens).
        """
        z = self.encoder(x.flatten(start_dim=1).to(self.device))
        logits_rec = self.decoder(z)
        return logits_rec.reshape(x.shape)

    def loss_function(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns a simple reconstruction loss: the multiclass
        cross entropy.
        """
        return torch.nn.functional.cross_entropy(
            input=self.forward(x.to(self.device)).permute(0, 2, 1),
            target=x.argmax(dim=-1).to(self.device),
            reduction="mean",
        )