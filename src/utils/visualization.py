"""
A set of utilities to draw molecules from SELFIE and SMILES strings,
using RDKit and cairosvg.
"""
from pathlib import Path

from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D

import cairosvg

import selfies as sf


def selfie_to_png(
    selfie: str, save_path: Path, width: int = 200, height: int = 200, title: str = None
):
    """
    Save a molecule (specified as a selfie string) as png file.

    Taken and adapted from the following stack overflow answer:
    https://stackoverflow.com/a/73449342/3516175
    """
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

    # Export to png
    cairosvg.svg2png(bytestring=svg.encode(), write_to=str(save_path))
