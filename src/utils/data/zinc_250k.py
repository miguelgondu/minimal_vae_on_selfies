"""Utilities for loading the Zinc 250k dataset"""
from typing import Dict, Tuple
from pathlib import Path
import json

import numpy as np

import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT_DIR = Path(__file__).parent.parent.parent.parent.resolve()


def load_zinc_250k_dataset() -> np.ndarray:
    """Returns the small molecule dataset of one-hot encoded SELFIES strings.

    Using the alphabet computed during preprocessing, this method
    loads the dataset of SELFIES strings, and one-hot encodes them.
    """
    dataset_path = (
        ROOT_DIR
        / "data"
        / "processed"
        / "zinc_250k"
        / "zinc_250k_onehot_and_integers.npz"
    )

    return np.load(dataset_path)["onehot"]


def load_zinc_250k_alphabet() -> Dict[str, int]:
    """
    Returns the alphabet (dict[str, int]) of SELFIES characters.
    """
    alphabet_path = ROOT_DIR / "data" / "processed" / "zinc_250k" / "alphabet_stoi.json"

    with open(alphabet_path, "r") as f:
        alphabet = json.load(f)

    return alphabet


def load_zinc_250k_dataloaders(
    random_seed: int = 42,
    train_test_split: float = 0.8,
    batch_size: int = 256,
    overfit_to_a_single_batch: bool = False,
) -> Tuple[DataLoader, DataLoader]:
    """
    Returns a train-test split for the Zinc 250k dataset.
    The inputs are shuffled according to the provided seed using
    numpy, and the dataloaders have shuffling turned on.
    """
    # Loading the one-hot representation
    one_hot_molecules = load_zinc_250k_dataset()

    # Shuffling according to the seed provided
    np.random.seed(random_seed)
    np.random.shuffle(one_hot_molecules)

    # Split the data into train and test using the
    # specified percentage.
    training_index = int(len(one_hot_molecules) * train_test_split)
    train_data = torch.from_numpy(one_hot_molecules[:training_index]).to(
        torch.get_default_dtype()
    )
    test_data = torch.from_numpy(one_hot_molecules[training_index:]).to(
        torch.get_default_dtype()
    )

    # Overfit to a single batch if specified
    if overfit_to_a_single_batch:
        train_data = train_data[:batch_size]
        test_data = test_data[:batch_size]

    # Building the datasets
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
    onehot = load_zinc_250k_dataset()
    print(onehot.shape)
