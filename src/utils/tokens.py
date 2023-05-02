"""
A set of utilities to transform SELFIE strings into tokens and vice versa.
"""
from pathlib import Path
from typing import List, Dict

import json
import re

import torch

ROOT_DIR = Path(__file__).parent.parent.parent.resolve()


def from_selfie_to_tokens(selfie: str) -> List[str]:
    """
    Given a selfie string, returns a list of all
    occurences of [.*?] in the string (i.e. whatever
    is between square brackes).
    """
    return list(re.findall(r"\[.*?\]", selfie))


def from_tokens_to_ids(
    tokens: List[str],
    tokens_dict: Dict[str, int] = None,
    max_token_length: int = 20,
    dataset_name: str = "TINY-CID-SELFIES",
) -> List[int]:
    """
    Returns the list of ids corresponding to the given tokens, always padded
    to have the same length. If the tokens_dict is not given, it is loaded
    from the data/processed/tokens_{dataset_name}.json file.

    By default, the dataset_name is set to SUPER-SMALL-CID-SELFIES, which
    is the dataset used for the experiments in this repository, and max_length is set to 20, which is the maximum length of a SELFIE string in the SMALL-CID-SELFIES and TINY-CID-SELFIES datasets.
    """
    if tokens_dict is None:
        with open(
            ROOT_DIR / "data" / "processed" / f"tokens_{dataset_name}.json", "r"
        ) as fp:
            tokens_dict = json.load(fp)

    # Returns the list of ids corresponding to the given tokens, always padded
    # to have the same length.
    return [tokens_dict[token] for token in tokens] + [0] * (
        max_token_length - len(tokens)
    )


def from_selfie_to_ids(
    selfie: str,
    tokens_dict: Dict[str, int] = None,
    max_token_length: int = 20,
    dataset_name: str = "TINY-CID-SELFIES-20",
) -> List[int]:
    """
    Returns a list of ids corresponding to the tokens
    inside the given selfie string.
    """
    tokens = from_selfie_to_tokens(selfie)
    return from_tokens_to_ids(
        tokens,
        tokens_dict=tokens_dict,
        max_token_length=max_token_length,
        dataset_name=dataset_name,
    )


def from_ids_to_tensor(
    ids: List[int],
    token_dict: Dict[str, int],
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """
    Returns a tensor of the given ids as one-hot encodings,
    on the given device.

    The output tensor has shape (1, length_of_sequence, size_of_vocabulary).
    """
    # Start by creating a tensor of zeros
    # of shape (1, length_of_sequence, size_of_vocabulary)
    one_hot_encoding = torch.zeros(1, len(ids), len(token_dict), device=device)

    # Set the corresponding indices to 1
    one_hot_encoding[0, torch.arange(len(ids)), ids] = 1

    return one_hot_encoding


def from_selfie_to_tensor(
    selfie: str, token_dict: Dict[str, int], device: torch.device = torch.device("cpu")
) -> torch.Tensor:
    """
    Returns a tensor of the given selfie string as one-hot encodings,
    on the given device.

    The output tensor has shape (1, length_of_sequence, size_of_vocabulary).
    """
    ids = from_selfie_to_ids(selfie, token_dict)
    return from_ids_to_tensor(ids, token_dict, device=device)


def from_tensor_to_selfie(x: torch.Tensor, tokens_dict: Dict[str, int]) -> str:
    """
    Returns a selfie string from a tensor of one-hot encodings.
    """
    # Get the inverse of the tokens_dict
    inv_tokens_dict = {v: k for k, v in tokens_dict.items()}

    # Get the indices of the maximum values
    indices = torch.argmax(x, dim=-1).squeeze()
    selfie = "".join(
        [inv_tokens_dict[index.item()] for index in indices if index.item() != 0]
    )

    return selfie
