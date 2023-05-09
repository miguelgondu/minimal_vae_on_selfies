"""
In this script, we deploy evolutionary strategies
with the aim of maximizing the QED of molecules
in the latent space of our VAESelfies.
"""
from typing import Dict
from pathlib import Path
import json
from time import time

import torch

import selfies as sf

from rdkit import Chem
from rdkit.Chem.QED import qed

from models.vae import VAESelfies

from search_functions.evolutionary_strategies import EvolutionaryStrategy
from search_functions import CMA_ES, SimpleEvolutionStrategy, PGPE, SNES

from utils.tokens import from_tensor_to_selfie, from_latent_code_to_selfie

ROOT_DIR = Path(__file__).parent.parent.parent.resolve()
MODEL_DIR = ROOT_DIR / "data" / "trained_models"
RESULTS_DIR = ROOT_DIR / "data" / "evolutionary_search_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def random_search(n_samples: int):
    """
    Samples n_samples latent vectors, decode them
    to molecules and computes their QED score.
    """
    # Load model
    model_path = (
        ROOT_DIR / "data" / "trained_models" / "VAESelfies_TINY-CID-SELFIES-20.pt"
    )
    vae = VAESelfies()
    vae.load_state_dict(torch.load(model_path, map_location="cpu"))

    # Sample latent codes
    z = vae.p_z.sample((n_samples,))

    # Decode latent codes
    selfie_probs = vae.decode(z).probs
    selfies = from_tensor_to_selfie(selfie_probs, vae.tokens_dict)

    # Translate to SMILES
    smiles = [sf.decoder(s) for s in selfies]

    # Make sure that all molecules are valid
    valid_smiles = []
    unvalid_smiles = []
    for i, smile in enumerate(smiles):
        mol = Chem.MolFromSmiles(smile)
        if mol is not None:
            valid_smiles.append((i, mol))
        else:
            unvalid_smiles.append((i, mol))

    # Compute MolSkill scores
    valid_scores = torch.Tensor([qed(s) for _, s in valid_smiles])
    scores = torch.nan * torch.ones(n_samples)
    for (i, smile), valid_score in zip(valid_smiles, valid_scores):
        scores[i] = valid_score.item()

    # Save results
    results = [{"selfie": s, "score": scores[i].item()} for i, s in enumerate(selfies)]
    timestamp = str(int(time()))
    with open(RESULTS_DIR / f"{timestamp}_QED_Random.json", "w") as fp:
        json.dump(results, fp)


def running_latent_space_evolutionary_strategy(
    evolutionary_strategy: EvolutionaryStrategy,
    n_generations: int = 100,
    population_size: int = 50,
    exploration: float = 0.1,
) -> Dict[str, float]:
    """
    Implements CMA-ES in the latent space of our VAESelfies
    model. It uses MolSkill to evaluate the molecules, and
    thus finds molecules that maximize "chemist's intuitions".
    """
    # Loads model to define the objective function
    # for our CMA-ES
    model_path = MODEL_DIR / "VAESelfies_TINY-CID-SELFIES-20.pt"
    vae = VAESelfies()
    vae.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))

    # Defines the objective function
    def objective_function(z: torch.Tensor) -> torch.Tensor:
        """
        TODO: what happens if there's a decoder error?
        I guess we should be filling those zs with 0.0
        """
        # Decode latent codes
        selfie_probs = vae.decode(z).probs
        selfies = from_tensor_to_selfie(selfie_probs, vae.tokens_dict)

        # Translate to SMILES
        smiles = [sf.decoder(s) for s in selfies]

        # Check if the SMILES are valid molecules using Chem
        valid_smiles = []
        unvalid_smiles = []
        for i, smile in enumerate(smiles):
            mol = Chem.MolFromSmiles(smile)
            if mol is not None:
                valid_smiles.append((i, mol))
            else:
                unvalid_smiles.append((i, mol))

        # Compute QED scores, and clean NaNs
        valid_scores = torch.Tensor([qed(mol) for _, mol in valid_smiles])
        # valid_scores[valid_scores == torch.nan] = 0.0

        scores = torch.nan * torch.ones(z.shape[0])
        for (i, smile), valid_score in zip(valid_smiles, valid_scores):
            scores[i] = valid_score

        return torch.Tensor(scores)

    evolutionary_strategy_ = evolutionary_strategy(
        objective_function=objective_function,
        population_size=population_size,
        exploration=exploration,
        solution_length=vae.latent_dim,  # Size of the latent space
        limits=(
            -5 * torch.ones(vae.latent_dim),
            5 * torch.ones(vae.latent_dim),
        ),
    )

    results = []
    timestamp = str(int(time()))
    for generation in range(n_generations):
        evolutionary_strategy_.step()
        current_best = evolutionary_strategy_.get_current_best()
        current_best_selfie = from_latent_code_to_selfie(current_best, vae)
        current_best_mol = Chem.MolFromSmiles(sf.decoder(current_best_selfie))
        current_best_selfie_score = qed(current_best_mol)
        print(
            generation,
            current_best_selfie,
            current_best_selfie_score,
        )
        results.append(
            {
                "selfie": current_best_selfie,
                "score": current_best_selfie_score,
            }
        )
        with open(
            RESULTS_DIR / f"{timestamp}_QED_{evolutionary_strategy.__name__}.json",
            "w",
        ) as fp:
            json.dump(results, fp)


if __name__ == "__main__":
    # running_latent_space_evolutionary_strategy(CMA_ES, n_generations=100)
    running_latent_space_evolutionary_strategy(
        SimpleEvolutionStrategy, n_generations=100
    )
    running_latent_space_evolutionary_strategy(PGPE, n_generations=100)
    running_latent_space_evolutionary_strategy(SNES, n_generations=100)
    # random_search(100 * 50)
