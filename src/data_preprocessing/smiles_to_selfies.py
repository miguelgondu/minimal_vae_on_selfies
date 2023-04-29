"""
This script loads the SMILES dataset and converts it to SELFIES.
"""
import os
from pathlib import Path

import pandas as pd

from selfies import encoder, EncoderError


ROOT_DIR = Path(__file__).parent.parent.parent.resolve()


def encode_if_possible(smiles: str) -> str:
    try:
        return encoder(smiles)
    except EncoderError:
        return pd.NA


def process_chuck(chunk: pd.DataFrame) -> pd.DataFrame:
    """
    Takes a chunk of the SMILES dataset and converts it to SELFIES,
    dropping the ones that cannot be converted according to Aspuru-Guzik's
    library `selfies` [1].

    [1]: https://github.com/aspuru-guzik-group/selfies
    """
    chunk["SELFIES"] = chunk[1].apply(encode_if_possible)
    chunk.drop(columns=[1], inplace=True)
    chunk.dropna(inplace=True)
    return chunk


if __name__ == "__main__":
    # Making sure the file exists
    DATA_PATH = ROOT_DIR / "data"
    PROCESSED_DATA_PATH = DATA_PATH / "processed"
    PROCESSED_DATA_PATH.mkdir(exist_ok=True, parents=True)

    if not (DATA_PATH / "raw" / "CID-SMILES").exists():
        raise FileNotFoundError(
            "The CID-SMILES dataset was not found. "
            "Please run download_dataset.py first."
        )

    # Loading the dataset in chuncks
    keyword_args = {
        "sep": "\t",
        "header": None,
        "chunksize": 1e4,
    }
    with pd.read_csv(DATA_PATH / "raw" / "CID-SMILES", **keyword_args) as reader:
        for i, chunk in enumerate(reader):
            print(f"Processing chunk {i}...")
            chunk_selfies = process_chuck(chunk)
            chunk_selfies.to_csv(
                PROCESSED_DATA_PATH / f"CID-SELFIES-{i}",
                sep="\t",
                header=False,
                index=False,
            )

            # Merging the chunk with the current dataset
            os.system(
                f"cat {PROCESSED_DATA_PATH / ('CID-SELFIES-' + str(i))} "
                f">> {PROCESSED_DATA_PATH / 'CID-SELFIES'}"
            )
            os.system(f"rm {PROCESSED_DATA_PATH / ('CID-SELFIES-' + str(i))}")
