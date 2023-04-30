"""
Loads the Autoencoder model in trained_models/ae_{database_name}.pt
and checks how interpolations in the latet space look.
"""

from pathlib import Path
import json

import torch

from models.ae import Autoencoder
from src.utils.data import load_dataloaders

from utils.data import load_dataset_as_dataframe
from utils.tokens import from_selfie_to_tokens

ROOT_DIR = Path(__file__).parent.parent.parent.resolve()


def explore(model_name: str):
    dataset_name = model_name.split("_")[-1]

    # Load the tokens dictionary
    with open(
        ROOT_DIR / "data" / "processed" / f"tokens_{dataset_name}.json", "r"
    ) as fp:
        tokens_dict = json.load(fp)

    # Load the model and set it to eval mode
    model = Autoencoder(
        dataset_name=dataset_name,
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

    train_df, test_df = load_dataset_as_dataframe()
    train_loader, test_loader = load_dataloaders()
    test_tensor = train_loader.dataset.tensors[0]

    # Get the latent space representation of the test set
    with torch.no_grad():
        latent_representations = model.encoder(test_tensor.flatten(start_dim=1))
        reconstruction = (
            model.decoder(latent_representations)
            .reshape(test_tensor.shape)
            .argmax(dim=-1)
        )

    # Convert the reconstruction to SELFIES
    inv_tokens_dict = {v: k for k, v in tokens_dict.items()}
    reconstruction_selfies = [
        "".join([inv_tokens_dict[i] for i in x if i != 0])
        for x in reconstruction.tolist()
    ]
    # print(reconstruction_selfies)

    for i, reconstructed_selfie in enumerate(reconstruction_selfies):
        print(train_df.iloc[i]["SELFIES"], "vs.", reconstructed_selfie, "\n")


if __name__ == "__main__":
    explore("ae_SUPER-SMALL-CID-SELFIES")
