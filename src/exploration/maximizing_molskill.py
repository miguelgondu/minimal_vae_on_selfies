"""
Uses MolSkill [1] to find the best molecules in the latent space of a VAE.

[1]: Learning chemical intuition from humans in the loop, Cheoung et al., 2023
     https://doi.org/10.26434/chemrxiv-2023-knwnv-v2
"""

from pathlib import Path

import torch

from molskill.scorer import MolSkillScorer

import selfies as sf

from models.vae import VAESelfies

from utils.tokens import from_tensor_to_selfie
from utils.visualization import selfie_to_png


ROOT_DIR = Path(__file__).parent.parent.parent.resolve()


def molskill_on_simple_example():
    """
    Tests whether molskill works on the examples
    they provide on their GitHub page.
    """

    smiles_strs = ["CCO", "O=C(Oc1ccccc1C(=O)O)C"]

    scorer = MolSkillScorer()
    scores = scorer.score(smiles_strs)

    print(scores)


def molskill_on_random_samples():
    """
    Loads our model, samples some latent codes,
    decodes them as SELFIES, translates them
    to SMLIES and computes their MolSkill score.
    """
    # Load model
    model_path = (
        ROOT_DIR / "data" / "trained_models" / "VAESelfies_TINY-CID-SELFIES-20.pt"
    )
    vae = VAESelfies()
    vae.load_state_dict(torch.load(model_path, map_location="cpu"))

    # Sample latent codes
    n_samples = 100
    z = vae.p_z.sample((n_samples,))

    # Decode latent codes
    selfie_probs = vae.decode(z).probs
    selfies = from_tensor_to_selfie(selfie_probs, vae.tokens_dict)

    # Translate to SMILES
    smiles = [sf.decoder(s) for s in selfies]

    # Compute MolSkill scores
    scorer = MolSkillScorer()
    scores = scorer.score(smiles)

    FIG_PATH = ROOT_DIR / "data" / "figures" / "molskill_samples"
    FIG_PATH.mkdir(exist_ok=True, parents=True)
    for i, selfie in enumerate(selfies):
        selfie_to_png(selfie, FIG_PATH / f"selfie_{i}_molskill_{scores[i]:.2f}.png")

    print("Selfies: ", selfies)
    print("SMILES: ", smiles)
    print("Scores: ", scores)


if __name__ == "__main__":
    # molskill_on_simple_example()
    molskill_on_random_samples()
