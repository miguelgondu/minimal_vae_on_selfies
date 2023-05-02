"""
A set of utilities to draw molecules from SELFIE and SMILES strings,
using RDKit and cairosvg.
"""
from pathlib import Path

from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D

import cairosvg

import selfies as sf


def selfie_to_png(selfie: str, save_path: Path, width: int = 200, height: int = 200):
    """
    Save a molecule (specified as a selfie string) as png file.

    Taken and adapted from the following stack overflow answer:
    https://stackoverflow.com/a/73449342/3516175
    """
    # Convert selfie to mol
    mol = Chem.MolFromSmiles(sf.decoder(selfie))
    assert mol is not None, f"Couldn't convert {selfie} to mol"

    # Render high resolution molecule
    drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()

    # Export to png
    cairosvg.svg2png(
        bytestring=drawer.GetDrawingText().encode(), write_to=str(save_path)
    )
