"""
This module tests whether the model loads properly.

Types of tests:
 - Unit tests.
 - Integration tests.
 - Regression tests: the ones that you implement as soon as
   you find a bug.
- End-to-end tests. These are the ones that you implement
    to test the whole system.

A test is a promise I'm making to the user.
"""
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent.parent.resolve()
import sys

sys.path.append(str(ROOT_DIR))
sys.path.append(str(ROOT_DIR / "src"))

from src.models.vae import VAESelfies

from src.utils.models.load import load_model


def test_model_loading():
    model = load_model()
    assert isinstance(model, VAESelfies)

    # example_selfies = "[C][C][C]"

    # selfies_as_one_hot = transform_selfies(example_selfies)

    # prediction = model(selfies_as_one_hot)


if __name__ == "__main__":
    test_model_loading()
