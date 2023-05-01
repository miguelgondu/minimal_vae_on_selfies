"""
Computes the tokens for the SUPER-SMALL-CID-SELFIE dataset,
and stores it in a JSON file in data/processed/tokens.json.
These tokens are sorted by their frequency in the dataset,
except for the padding token, which is always the first one.
"""

from pathlib import Path
import json
from collections import defaultdict

import pandas as pd

from utils.tokens import from_selfie_to_tokens

ROOT_DIR = Path(__file__).parent.parent.parent.resolve()

def load_dataset(dataset_path: Path) -> pd.DataFrame:
    return pd.read_csv(
        dataset_path,
        sep="\t",
        header=None,
        names=["cid", "selfie"],
    )

if __name__ == "__main__":
    # Setting up data paths
    PROCESSED_DATA_PATH = ROOT_DIR / "data" / "processed"
    dataset_name = "TINY-CID-SELFIES"
    
    # Loading the dataset
    dataset = load_dataset(PROCESSED_DATA_PATH / dataset_name)

    # Computing the tokens
    token_counts = defaultdict(int)
    for selfie in dataset["selfie"]:
        for token in from_selfie_to_tokens(selfie):
            token_counts[token] += 1

    # Sorting the tokens by their frequency
    sorted_tokens = sorted(token_counts.items(), key=lambda x: x[1], reverse=True)

    # Adding the padding token
    tokens = ["[<PAD>]"] + [k for k, _ in sorted_tokens]

    # Saving the tokens
    with open(PROCESSED_DATA_PATH / f"tokens_{dataset_name}.json", "w") as fp:
        json.dump({token: i for i, token in enumerate(tokens)}, fp)
