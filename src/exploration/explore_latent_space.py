"""
Loads a Variational Autoencoder model in
trained_models/VAESelfies_{database_name}.pt and
checks how samples/random walks look in latent space look.
"""
from typing import Union
from pathlib import Path
import json

import torch
from torch.distributions import Normal, Categorical

import selfies as sf

from rdkit import Chem
from rdkit.Chem.QED import qed

from models.ae import AutoencoderSelfies
from models.vae import VAESelfies

from utils.data import load_dataloaders
from utils.data import load_dataset_as_dataframe
from utils.tokens import (
    from_selfie_to_tokens,
    from_selfie_to_ids,
    from_selfie_to_tensor,
    from_tensor_to_selfie,
)
from utils.visualization import selfie_to_png


ROOT_DIR = Path(__file__).parent.parent.parent.resolve()


def decode_some_test_data(model: Union[AutoencoderSelfies, VAESelfies]):
    dataset_name = model.dataset_name

    # Load the tokens dictionary
    with open(
        ROOT_DIR / "data" / "processed" / f"tokens_{dataset_name}.json", "r"
    ) as fp:
        tokens_dict = json.load(fp)

    _, test_df = load_dataset_as_dataframe(dataset_name=dataset_name)
    _, test_loader = load_dataloaders(dataset_name=dataset_name)
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


def unconditioned_samples_from_latent_space(
    model: Union[AutoencoderSelfies, VAESelfies], n_samples: int = 32
):
    """
    Samples {n_samples} latent vectors from the prior distribution, decodes them,
    transforms them to SELFIES, and plots them.
    """
    # Sample {n_samples} latent vectors from the prior distribution
    with torch.no_grad():
        latent_representations = model.p_z.sample((n_samples,))

        # Get the reconstruction
        reconstruction = model.decode(latent_representations)
        if isinstance(reconstruction, Categorical):
            reconstruction = reconstruction.probs

        reconstruction = reconstruction.argmax(dim=-1)

    # Convert the reconstruction to SELFIES
    inv_tokens_dict = {v: k for k, v in model.tokens_dict.items()}
    reconstruction_selfies = [
        "".join([inv_tokens_dict[i] for i in x if i != 0])
        for x in reconstruction.tolist()
    ]

    # Plot the selfies
    IMGS_DIR = ROOT_DIR / "data" / "figures"
    IMGS_DIR.mkdir(exist_ok=True, parents=True)

    for i, reconstructed_selfie in enumerate(reconstruction_selfies):
        try:
            selfie_to_png(reconstructed_selfie, IMGS_DIR / f"sample_{i}.png")
        except AssertionError as e:
            print(e)


def visualize_a_random_walk(
    model: Union[AutoencoderSelfies, VAESelfies], noise_scale: float = 0.5
):
    """
    Takes the SMILES for Aspirin, transfroms it to SELFIES,
    and then walks around it in the latent space of the VAESelfies
    we trained.
    """
    # Defining the SMILES and SELFIES for Aspirin
    aspirin_smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"
    aspirin_selfie = sf.encoder(aspirin_smiles)

    print("QED for Aspirin:", qed(Chem.MolFromSmiles(aspirin_smiles)))

    # Defining the saving paths for images
    IMGS_DIR = ROOT_DIR / "data" / "figures"
    IMGS_DIR.mkdir(exist_ok=True, parents=True)
    selfie_to_png(aspirin_selfie, IMGS_DIR / "aspirin_0.png", width=300, height=300)

    # Computing the one-hot representation of Aspirin
    x = from_selfie_to_tensor(aspirin_selfie, model.tokens_dict)

    # Encode the SELFIES for aspirin
    with torch.no_grad():
        latent_representation = model.encode(x)
        if isinstance(latent_representation, Normal):
            latent_representation = latent_representation.mean

    # Walk around the latent space
    for i in range(10):
        with torch.no_grad():
            latent_representation += (
                torch.randn_like(latent_representation) * noise_scale
            )
            reconstruction = model.decode(latent_representation)
            if isinstance(reconstruction, Categorical):
                reconstruction = reconstruction.probs

            selfie = from_tensor_to_selfie(reconstruction, model.tokens_dict)
            # print(latent_representation)
            print(selfie, qed(Chem.MolFromSmiles(sf.decoder(selfie))))

            # Draw the molecule
            selfie_to_png(
                selfie, IMGS_DIR / f"aspirin_{i+1}.png", width=300, height=300
            )


if __name__ == "__main__":
    # Load the model and set it to eval mode
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

    # unconditioned_samples_from_latent_space(model)
    visualize_a_random_walk(model, noise_scale=0.05)
