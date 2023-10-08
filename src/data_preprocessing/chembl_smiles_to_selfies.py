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
    subchunk = chunk[["ChEMBL ID", "Smiles"]]
    subchunk.dropna(inplace=True)
    subchunk["Selfies"] = chunk["Smiles"].apply(encode_if_possible)
    subchunk.dropna(inplace=True)

    return subchunk


if __name__ == "__main__":
    # Making sure the file exists
    DATA_PATH = ROOT_DIR / "data"
    PROCESSED_DATA_PATH = DATA_PATH / "processed"
    PROCESSED_DATA_PATH.mkdir(exist_ok=True, parents=True)
    dataset_name = "CHEMBL_small_molecules_dataset"

    if not (DATA_PATH / "raw" / f"{dataset_name}.csv").exists():
        raise FileNotFoundError(
            "The CHEMBL dataset was not found. " "Please run download_dataset.py first."
        )

    # Loading the dataset in chuncks
    keyword_args = {
        "sep": ";",
        "chunksize": 1e4,
    }
    with pd.read_csv(
        DATA_PATH / "raw" / f"{dataset_name}.csv", **keyword_args
    ) as reader:
        for i, chunk in enumerate(reader):
            print(f"Processing chunk {i}...")
            chunk_selfies = process_chuck(chunk)
            chunk_selfies.to_csv(
                PROCESSED_DATA_PATH / f"{dataset_name}-{i}",
                sep="\t",
                header=False,
                index=False,
            )

            # Merging the chunk with the current dataset
            os.system(
                f"cat {PROCESSED_DATA_PATH / (f'{dataset_name}-' + str(i))} "
                f">> {PROCESSED_DATA_PATH / f'{dataset_name}'}"
            )
            os.system(f"rm {PROCESSED_DATA_PATH / (f'{dataset_name}-' + str(i))}")
