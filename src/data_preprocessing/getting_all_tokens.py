"""
This script loads the CID-SELFIES dataset and extracts all
possible tokens, which are represented betweek brackets.

It saves a tokens.json file with pairs (token_id, token).
These will be used to one-hot encode the SELFIES.
"""
from collections import defaultdict
from pathlib import Path
import re
import json

import pandas as pd

ROOT_DIR = Path(__file__).parent.parent.parent.resolve()

if __name__ == "__main__":
    # Loading the dataset in chuncks
    DATA_PATH = ROOT_DIR / "data"
    PROCESSED_DATA_PATH = DATA_PATH / "processed"

    keyword_args = {
        "sep": "\t",
        "header": None,
        "chunksize": 1e6,
    }
    tokens_with_count = defaultdict(int)
    with pd.read_csv(PROCESSED_DATA_PATH / "CID-SELFIES", **keyword_args) as reader:
        for i, chunk in enumerate(reader):
            # In the column for selfies (i.e. 1), check for all
            # texts between square brackets using a regular expression
            # and save them in a set
            print(f"Processing chunk {i}...")
            selfies = chunk[1]
            for selfie in selfies.values:
                for token in set(re.findall(r"\[.*?\]", selfie)):
                    tokens_with_count[token] += 1

            # Saving the tokens
            with open(PROCESSED_DATA_PATH / "tokens.json", "w") as fp:
                json.dump(tokens_with_count, fp)

    print(tokens_with_count)
