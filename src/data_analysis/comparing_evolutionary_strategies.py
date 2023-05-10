"""
This script takes the results in data/evolutionary_search_results
and compares them among each other. We are interested in seeing
what the best performing evolutionary strategy is, and whether it
beats random search.

The structure of each of these files is a list of dictionaries,
[
    # The result for the first generation
    {
        "selfie": ...,  # The SELFIE string
        "score": ...    # Its score (either MolSkill or QED)
    },
    ...
]

Except for the random results, which are just n_generations * pop_size random
samples from the latent space prior N(0, 1).
"""
from pathlib import Path
import json

import pandas as pd

import seaborn as sns

from utils.visualization import selfie_to_png

ROOT_DIR = Path(__file__).parent.parent.parent.resolve()
DATA_DIR = ROOT_DIR / "data" / "evolutionary_search_results"
FIG_PATH = ROOT_DIR / "data" / "figures" / "evolutionary_strategies"
FIG_PATH.mkdir(parents=True, exist_ok=True)


def compare_algorithms(pattern: str = "results"):
    """
    Loads up all algorithms and builds a Pandas DataFrame
    with the best SELFIE and its score for each algorithm.

    It accepts a pattern, which is "results" (i.e. MolSkill)
    by default. It could also be "QED".
    """
    filepaths = DATA_DIR.glob(f"*_{pattern}_*.json")

    # This list will contain the rows of a table
    # with columns "Algorithm", "Best SELFIES", "Score"
    results = []
    for filepath in filepaths:
        with open(filepath, "r") as fp:
            # Load the results
            data = json.load(fp)

        # Sort these by score
        data = sorted(data, key=lambda x: x["score"], reverse=True)

        # Add the row to the list
        results.append(
            {
                "Algorithm": "".join(filepath.stem.split("_")[2:]),
                "Best SELFIES": data[0]["selfie"],
                "Score": data[0]["score"],
            }
        )

    # Make the dataframe
    table = pd.DataFrame(results)
    print(table)

    for algorithm, best_selfie, score in zip(
        table["Algorithm"], table["Best SELFIES"], table["Score"]
    ):
        # Save the figure of this best selfie
        if algorithm == "SimpleEvolutionStrategy":
            algorithm = "Simple ES"

        selfie_to_png(
            best_selfie,
            save_path=FIG_PATH / f"{pattern}_{algorithm}.png",
            title=f"{algorithm}: {score:.2f}",
            width=300,
            height=300,
        )


if __name__ == "__main__":
    compare_algorithms(pattern="QED")
