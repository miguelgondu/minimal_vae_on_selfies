"""
A set of utilities to transform SELFIE strings into tokens and vice versa.
"""
from pathlib import Path
from typing import List, Dict

import json
import re

ROOT_DIR = Path(__file__).parent.parent.parent.resolve()


def from_selfie_to_tokens(selfie: str) -> List[str]:
    return re.findall(r"\[.*?\]", selfie)


def from_tokens_to_ids(
    tokens: List[str], tokens_dict: Dict[str, int] = None
) -> List[int]:
    if tokens_dict is None:
        with open(ROOT_DIR / "data" / "processed" / "tokens.json", "r") as fp:
            tokens_dict = json.load(fp)

    return [tokens_dict[token] for token in tokens]
