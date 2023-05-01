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


class AutoencoderSelfies(torch.nn.Module):
    def __init__(
        self,
        dataset_name: str = "TINY-CID-SELFIES",
        max_token_length: int = 50,
        latent_dim: int = 64,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        self.dataset_name = dataset_name
        self.device = device
        self.max_token_length = max_token_length

        # Load the token dictionary
        # making sure that the file exists in the first place.
        assert (
            ROOT_DIR / "data" / "processed" / f"tokens_{dataset_name}.json"
        ).exists(), (
            f"tokens_{dataset_name}.json does not exist in "
            f"{ROOT_DIR / 'data' / 'processed'}! Did you forget to run "
            "src/tokenizing/compute_tokens.py?"
        )

        with open(
            ROOT_DIR / "data" / "processed" / f"tokens_{dataset_name}.json", "r"
        ) as fp:
            self.token_dict = json.load(fp)

        # Define the input length: length of a given SELFIES
        # (always padded to be {max_length}), times the number of tokens
        self.input_length = max_token_length * len(self.token_dict)

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

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Takes a batch of SELFIES, encoded as one-hot vectors
        and runs a forward pass through the encoder.

        Input:
            x: torch.Tensor of shape (batch_size, length_of_sequence, n_tokens).

        Output:
            z: torch.Tensor of shape (batch_size, latent_dim).
        """
        return self.encoder(x.flatten(start_dim=1).to(self.device))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Takes a batch of latent vectors and runs a forward
        pass through the decoder.

        Input:
            z: torch.Tensor of shape (batch_size, latent_dim).

        Output:
            logits_rec: torch.Tensor of shape (batch_size, length_of_sequence, n_tokens).
        """
        return self.decoder(z).reshape(
            z.shape[0], self.max_token_length, len(self.token_dict)
        )

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
