"""
Uses MolSkill [1] to find the best molecules in the latent space of a VAE.

[1]: Learning chemical intuition from humans in the loop, Cheoung et al., 2023
     https://doi.org/10.26434/chemrxiv-2023-knwnv-v2
"""

from pathlib import Path

import torch

from models.vae import VAESelfies

from molskill.scorer import MolSkillScorer


def test_molskill_on_simple_example():
    """
    Tests whether molskill works on the examples
    they provide on their GitHub page.
    """

    smiles_strs = ["CCO", "O=C(Oc1ccccc1C(=O)O)C"]

    scorer = MolSkillScorer()
    scores = scorer.score(smiles_strs)

    print(scores)


if __name__ == "__main__":
    test_molskill_on_simple_example()
