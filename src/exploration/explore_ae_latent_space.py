"""
Loads the Autoencoder model in trained_models/ae_{database_name}.pt
and checks how interpolations in the latet space look.
"""

from pathlib import Path

import torch

from models.ae import Autoencoder

ROOT_DIR = Path(__file__).parent.parent.parent.resolve()


def explore(model_name: str):
    model = Autoencoder(
        database_name="SUPER-SMALL-CID-SELFIES",
        max_length=300,
        latent_dim=64,
        device=torch.device("cpu"),
    )

    model.load_state_dict(
        torch.load(
            ROOT_DIR / "data" / "trained_models" / f"{model_name}.pt",
            map_location=torch.device("cpu"),
        )
    )

    model.eval()
    print(model)


if __name__ == "__main__":
    explore("ae_SUPER-SMALL-CID-SELFIES")
