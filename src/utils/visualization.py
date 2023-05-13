"""
A set of utilities to draw molecules from SELFIE and SMILES strings,
using RDKit and cairosvg.
"""
from pathlib import Path
from PIL import Image
import io

from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D

import cairosvg

import selfies as sf


def selfie_to_png(
    selfie: str,
    save_path: Path = None,
    width: int = 200,
    height: int = 200,
    title: str = None,
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

    # Export to png, and return it as a bytestring
    return cairosvg.svg2png(bytestring=svg.encode(), write_to=save_path)


def selfie_to_image(
    selfie: str,
    width: int = 200,
    height: int = 200,
    title: str = None,
    strict: bool = True,
) -> Image:
    """
    Tries to convert a selfie string to an image, and returns it as a PIL Image.

    If the conversion fails, terminates with an error. But if strict is set to False, it returns a blank image instead.
    """
    try:
        image_as_bytes = selfie_to_png(selfie, width=width, height=height, title=title)
    except AssertionError as e:
        if strict:
            raise e
        else:
            # Create a blank image using Image
            return Image.new("RGB", (width, height), (255, 255, 255))

    return Image.open(io.BytesIO(image_as_bytes))


if __name__ == "__main__":
    SELFIES = "[C][C][C][C][C][Branch1][C][C][C][=Branch1][C][C][C][=Branch1][C][=Branch1][C][=O][O]"
    image = selfie_to_image(SELFIES)
    print(image)
