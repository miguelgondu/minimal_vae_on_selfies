"""
A set of utilities to draw molecules from SELFIE and SMILES strings,
using RDKit and cairosvg.
"""
from pathlib import Path
import PIL.Image as Image
import io

from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D

import numpy as np

import cairosvg

import selfies as sf


def draw_molecule_from_selfies(
    selfie: str, width: int = 200, height: int = 200, title: str = None
) -> str:
    if title is not None:
        # Expand the image a bit, to give room to the title
        # at the bottom
        height += int(height * 0.15)

    # Convert selfie to mol
    mol = Chem.MolFromSmiles(sf.decoder(selfie))
    assert mol is not None, f"Couldn't convert {selfie} to mol"

    # Render high resolution molecule
    drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()

    # Add title to the image
    svg = drawer.GetDrawingText()
    if title is not None:
        svg = svg.replace(
            "</svg>",
            f'<text x="{width // 3}" y="{height - 20}" font-size="15" fill="black">{title}</text></svg>',
        )

    return svg


def selfie_to_png(
    selfie: str, save_path: Path, width: int = 200, height: int = 200, title: str = None
):
    """
    Save a molecule (specified as a selfie string) as png file.

    Taken and adapted from the following stack overflow answer:
    https://stackoverflow.com/a/73449342/3516175
    """
    svg = draw_molecule_from_selfies(selfie, width, height, title)

    # Export to png
    cairosvg.svg2png(bytestring=svg.encode(), write_to=str(save_path))


def selfie_to_image(
    selfie: str, width: int = 200, height: int = 200, title: str = None
) -> Image:
    svg = draw_molecule_from_selfies(selfie, width, height, title)
    s_png = cairosvg.svg2png(bytestring=svg)
    s_img = Image.open(io.BytesIO(s_png))

    return s_img

def selfie_to_numpy_image_array(
    selfie: str, width: int = 200, height: int = 200, title: str = None
) -> np.ndarray:
    """
    Returns a numpy array representing the image of the molecule
    represented by the given selfie string.
    """
    img = selfie_to_image(selfie, width, height, title)
    return np.array(img)
