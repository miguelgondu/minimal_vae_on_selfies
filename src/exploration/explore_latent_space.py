"""
Loads the Autoencoder model in trained_models/ae_{database_name}.pt
and checks how interpolations in the latet space look.
"""
from typing import Union
from pathlib import Path
import json

import torch
from torch.distributions import Normal, Categorical

from models.ae import AutoencoderSelfies
from models.vae import VAESelfies
from utils.data import load_dataloaders

from utils.data import load_dataset_as_dataframe
from utils.tokens import from_selfie_to_tokens

ROOT_DIR = Path(__file__).parent.parent.parent.resolve()


def explore(model: Union[AutoencoderSelfies, VAESelfies]):
    dataset_name = model.dataset_name

    # Load the tokens dictionary
    with open(
        ROOT_DIR / "data" / "processed" / f"tokens_{dataset_name}.json", "r"
    ) as fp:
        tokens_dict = json.load(fp)

    train_df, test_df = load_dataset_as_dataframe(dataset_name=dataset_name)
    train_loader, test_loader = load_dataloaders(dataset_name=dataset_name)
    some_tensors = test_loader.dataset.tensors[0][:10]

    # Get the latent space representation of the test set
    with torch.no_grad():
        latent_representations = model.encode(some_tensors.flatten(start_dim=1))

        if isinstance(latent_representations, Normal):
            latent_representations = latent_representations.mean

        # Get the reconstruction
        reconstruction = model.decode(latent_representations)
        if isinstance(reconstruction, Categorical):
            reconstruction = reconstruction.probs

        reconstruction = reconstruction.argmax(dim=-1)

    # Convert the reconstruction to SELFIES
    inv_tokens_dict = {v: k for k, v in tokens_dict.items()}
    reconstruction_selfies = [
        "".join([inv_tokens_dict[i] for i in x if i != 0])
        for x in reconstruction.tolist()
    ]

    for i, reconstructed_selfie in enumerate(reconstruction_selfies):
        print(test_df.iloc[i]["SELFIES"], "vs.", reconstructed_selfie, "\n")


if __name__ == "__main__":
    # Load the model and set it to eval mode
    # model_name = "AutoencoderSelfies_TINY-CID-SELFIES-20"
    # model = AutoencoderSelfies(
    #     dataset_name="TINY-CID-SELFIES-20",
    #     max_token_length=20,
    #     latent_dim=64,
    #     device=torch.device("cpu"),
    # )

    model_name = "VAESelfies_TINY-CID-SELFIES-20"
    model = VAESelfies(
        dataset_name="TINY-CID-SELFIES-20",
        max_token_length=20,
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
    explore(model)
