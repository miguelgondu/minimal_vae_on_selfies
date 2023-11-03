from pathlib import Path

import torch

from src.models.vae import VAESelfies


ROOT_DIR = Path(__file__).parent.parent.parent.parent.resolve()


def load_model() -> VAESelfies:
    """Loads the model w. the weights"""
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

    return model
