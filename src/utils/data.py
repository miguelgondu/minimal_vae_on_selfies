"""
Implements utilities for loading data.
"""
from typing import Tuple
from pathlib import Path
import json

import pandas as pd

import torch
from torch.utils.data import DataLoader, TensorDataset

from utils.tokens import from_selfie_to_ids

ROOT_DIR = Path(__file__).parent.parent.parent.resolve()


def load_data(
    dataset_name: str = "SUPER-SMALL-CID-SELFIES",
    batch_size: int = 128,
    max_length: int = 300,
    train_ratio: float = 0.8,
    random_seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """
    Loads the dataset based on the dataset name, returing
    two dataloaders with a train/test split according to the
    provided train_ratio.

    These DataLoaders contain tensors of shape (batch_size, max_length, n_tokens).

    By default, the database used is SUPER-SMALL-CID-SELFIES, which is a
    small subset of the CID-SELFIES database, containing 5000 molecules. These
    have at most 300 tokens.
    """
    # Load the token dictionary
    with open(
        ROOT_DIR / "data" / "processed" / f"tokens_{dataset_name}.json", "r"
    ) as fp:
        tokens_dict = json.load(fp)

    # Load the data
    df = pd.read_csv(
        ROOT_DIR / "data" / "processed" / f"{dataset_name}",
        sep="\t",
        header=None,
        names=["CID", "SELFIES"],
    )

    # Split the SELFIES into train and test
    train_df = df.sample(frac=train_ratio, random_state=random_seed)
    test_df = df.drop(train_df.index)

    # Computing the tokens for each selfie
    train_df["tokens"] = train_df["SELFIES"].apply(
        lambda selfie: from_selfie_to_ids(
            selfie,
            tokens_dict=tokens_dict,
            max_length=max_length,
            dataset_name=dataset_name,
        )
    )
    test_df["tokens"] = test_df["SELFIES"].apply(
        lambda selfie: from_selfie_to_ids(
            selfie,
            tokens_dict=tokens_dict,
            max_length=max_length,
            dataset_name=dataset_name,
        )
    )

    # Build the one-hot vectors
    train_data = torch.zeros(
        (len(train_df), max_length, len(tokens_dict)), dtype=torch.float32
    )
    test_data = torch.zeros(
        (len(test_df), max_length, len(tokens_dict)), dtype=torch.float32
    )

    # Populate the one-hot vectors
    for i, tokens in enumerate(train_df["tokens"]):
        train_data[i, torch.arange(len(tokens)), tokens] = 1.0

    for i, tokens in enumerate(test_df["tokens"]):
        test_data[i, torch.arange(len(tokens)), tokens] = 1.0

    # Turn them into tensor datasets
    train_data = TensorDataset(train_data)
    test_data = TensorDataset(test_data)

    # Build the dataloaders
    train_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True, drop_last=True
    )
    test_loader = DataLoader(
        test_data, batch_size=batch_size, shuffle=True, drop_last=True
    )

    return train_loader, test_loader


if __name__ == "__main__":
    train_loader, test_loader = load_data()
    print(train_loader)
    print(test_loader)