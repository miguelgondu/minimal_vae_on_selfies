"""
Uses MolSkill [1] and several evolutionary strategies [2] to
find the best molecules in the latent space of a VAE.

[1]: Learning chemical intuition from humans in the loop, Cheoung et al., 2023
     https://doi.org/10.26434/chemrxiv-2023-knwnv-v2
[2]: Benchmarking Evolutionary Strategies and Bayesian Optimization,
     Najarro & González-Duque, 2023.
     https://github.com/real-itu/benchmarking_evolution_and_bo
"""
from typing import Dict
from pathlib import Path
import json
from time import time

import torch

from molskill.scorer import MolSkillScorer

import selfies as sf

from rdkit import Chem

from models.vae import VAESelfies

from search_functions import CMA_ES, SimpleEvolutionStrategy, PGPE, SNES
from search_functions.evolutionary_strategies import EvolutionaryStrategy

from utils.tokens import from_tensor_to_selfie, from_latent_code_to_selfie
from utils.visualization import selfie_to_png


ROOT_DIR = Path(__file__).parent.parent.parent.resolve()
MODEL_DIR = ROOT_DIR / "data" / "trained_models"
RESULTS_DIR = ROOT_DIR / "data" / "evolutionary_search_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def molskill_on_simple_example():
    """
    Tests whether molskill works on the examples
    they provide on their GitHub page.
    """

    smiles_strs = ["CCO", "O=C(Oc1ccccc1C(=O)O)C"]

    scorer = MolSkillScorer()
    scores = scorer.score(smiles_strs)

    print(scores)


def molskill_on_random_samples(n_samples: int = 100):
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
            valid_smiles.append((i, smile))
        else:
            unvalid_smiles.append((i, smile))

    # Compute MolSkill scores
    scorer = MolSkillScorer()
    valid_scores = scorer.score([s for _, s in valid_smiles])
    scores = torch.nan * torch.ones(n_samples)
    for (i, smile), valid_score in zip(valid_smiles, valid_scores):
        scores[i] = valid_score.item()

    # Save results
    results = [{"selfie": s, "score": scores[i].item()} for i, s in enumerate(selfies)]
    timestamp = str(int(time()))
    with open(RESULTS_DIR / f"{timestamp}_results_Random.json", "w") as fp:
        json.dump(results, fp)


def running_latent_space_evolutionary_strategy(
    evolutionary_strategy: EvolutionaryStrategy,
    n_generations: int = 100,
    population_size: int = 50,
    exploration: float = 0.1,
) -> Dict[str, float]:
    """
    Implements a given Evolutionary Strategy in the latent space
    of our VAESelfies model. It uses MolSkill to evaluate the
    molecules, and thus finds molecules that maximize "chemist's intuitions".
    """
    # Loads model to define the objective function
    # for our CMA-ES
    model_path = MODEL_DIR / "VAESelfies_TINY-CID-SELFIES-20.pt"
    vae = VAESelfies()
    vae.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))

    # Define MolSkill scorer
    scorer = MolSkillScorer()

    # Defines the objective function
    def objective_function(z: torch.Tensor) -> torch.Tensor:
        """
        The closure function that we will pass to our
        evolutionary strategy. It takes a batch of
        latent codes and returns a batch of scores.

        In the process, it checks whether the molecules
        are valid, and if not, it assigns them a score
        of 0.0 or torch.nan.
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
                valid_smiles.append((i, smile))
            else:
                unvalid_smiles.append((i, smile))

        # Compute MolSkill scores, and clean NaNs
        valid_scores = torch.Tensor(scorer.score(valid_smiles))
        valid_scores[valid_scores == torch.nan] = 0.0

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
        current_best_selfie_score = scorer.score([sf.decoder(current_best_selfie)])[0]
        print(
            generation,
            current_best_selfie,
            current_best_selfie_score,
        )
        results.append(
            {
                "selfie": current_best_selfie,
                "score": current_best_selfie_score.item(),
            }
        )
        with open(
            RESULTS_DIR / f"{timestamp}_results_{evolutionary_strategy.__name__}.json",
            "w",
        ) as fp:
            json.dump(results, fp)


if __name__ == "__main__":
    # Running an evolutionary strategy on the latent space
    n_generations = 20
    population_size = 50

    # evolutionary_strategy_ = CMA_ES
    # evolutionary_strategy_ = SimpleEvolutionStrategy
    # evolutionary_strategy_ = SNES
    # evolutionary_strategy_ = PGPE

    # running_latent_space_evolutionary_strategy(
    #     evolutionary_strategy=evolutionary_strategy_,
    #     n_generations=20,
    #     population_size=50,
    # )

    molskill_on_random_samples(n_samples=n_generations * population_size)
