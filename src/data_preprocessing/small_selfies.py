"""
This script transforms the CID-SELFIES dataset
by removing all SELFIES that are larger than 20 tokens.

It saves a SMALL-CID-SELFIES file with these,
and it also saves a TINY-CID-SELFIES file
with only the first 50000 SELFIES in SMALL-CID-SELFIES.
"""

from pathlib import Path
import os

import pandas as pd

from utils.tokens import from_selfie_to_tokens

ROOT_DIR = Path(__file__).parent.parent.parent.resolve()

if __name__ == "__main__":
    MAX_TOKEN_LENGTH = 20

    # Setting up the data paths
    DATA_PATH = ROOT_DIR / "data"
    PROCESSED_DATA_PATH = DATA_PATH / "processed"

    # Loading the CID-SELFIES dataset
    # by chunks
    keyword_args = {
        "sep": "\t",
        "header": None,
        "chunksize": 1e6,
    }
    with pd.read_csv(PROCESSED_DATA_PATH / "CID-SELFIES", **keyword_args) as reader:
        for i, chunk in enumerate(reader):
            print(f"Processing chunk {i}...")

            # Removing all SELFIES that are larger than MAX_TOKEN_LENGTH tokens
            chunk = chunk[
                chunk[1].apply(lambda x: len(from_selfie_to_tokens(x)))
                <= MAX_TOKEN_LENGTH
            ]

            # Saving the SELFIES
            chunk.to_csv(
                PROCESSED_DATA_PATH / f"SMALL-CID-SELFIES-{MAX_TOKEN_LENGTH}-{i}",
                sep="\t",
                header=False,
                index=False,
            )

            # Merging the chunks using the command line
            os.system(
                f"cat {PROCESSED_DATA_PATH}/SMALL-CID-SELFIES-{MAX_TOKEN_LENGTH}-* >> "
                f"{PROCESSED_DATA_PATH}/SMALL-CID-SELFIES-{MAX_TOKEN_LENGTH}"
            )
            os.system(
                f"rm {PROCESSED_DATA_PATH}/SMALL-CID-SELFIES-{MAX_TOKEN_LENGTH}-{i}"
            )

    # Saving a super small version of the dataset
    # composed of the first 5000 SELFIES.
    tiny_selfies = pd.read_csv(
        PROCESSED_DATA_PATH / f"SMALL-CID-SELFIES-{MAX_TOKEN_LENGTH}",
        sep="\t",
        header=None,
        nrows=50000,
    )
    tiny_selfies.to_csv(
        PROCESSED_DATA_PATH / f"TINY-CID-SELFIES-{MAX_TOKEN_LENGTH}",
        sep="\t",
        header=False,
        index=False,
    )
