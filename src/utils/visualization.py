"""
A set of utilities to draw molecules from SELFIE and SMILES strings,
using RDKit and cairosvg.
"""
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D

import cairosvg

import selfies as sf


def selfie_to_png(selfie: str, save_path: Path, width: int = 300, height: int = 300):
    """
    Save substance structure as jpg

    Taken and adapted from the following stack overflow answer:
    https://stackoverflow.com/a/73449342/3516175
    """
    # Convert selfie to mol
    mol = Chem.MolFromSmiles(sf.decoder(selfie))

    # Render high resolution molecule
    drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()

    # Export to png
    cairosvg.svg2png(
        bytestring=drawer.GetDrawingText().encode(), write_to=str(save_path)
    )
