"""
A set of utilities to transform SELFIE strings into tokens and vice versa.
"""
from pathlib import Path
from typing import List, Dict

import json
import re

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
    max_length: int = 300,
    dataset_name: str = "SUPER-SMALL-CID-SELFIES",
) -> List[int]:
    """
    Returns the list of ids corresponding to the given tokens, always padded
    to have the same length. If the tokens_dict is not given, it is loaded
    from the data/processed/tokens_{dataset_name}.json file.

    By default, the dataset_name is set to SUPER-SMALL-CID-SELFIES, which
    is the dataset used for the experiments in this repository, and max_length
    is set to 300, which is the maximum length of a SELFIE string in the
    SUPER-SMALL-CID-SELFIES dataset.
    """
    if tokens_dict is None:
        with open(
            ROOT_DIR / "data" / "processed" / f"tokens_{dataset_name}.json", "r"
        ) as fp:
            tokens_dict = json.load(fp)

    # Returns the list of ids corresponding to the given tokens, always padded
    # to have the same length.
    return [tokens_dict[token] for token in tokens] + [0] * (max_length - len(tokens))


def from_selfie_to_ids(
    selfie: str,
    tokens_dict: Dict[str, int] = None,
    max_length: int = 300,
    dataset_name: str = "SUPER-SMALL-CID-SELFIES",
) -> List[int]:
    """
    Returns a list of ids corresponding to the tokens
    inside the given selfie string.
    """
    tokens = from_selfie_to_tokens(selfie)
    return from_tokens_to_ids(
        tokens,
        tokens_dict=tokens_dict,
        max_length=max_length,
        dataset_name=dataset_name,
    )
