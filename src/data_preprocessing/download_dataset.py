"""
Connects to PubChem's FTP server and downloads the dataset
of all CID's SMILES.

The dataset is available at:
ftp://ftp.ncbi.nlm.nih.gov/pubchem/Compound/Extras/CID-SMILES.gz

See this answer in Cheminformatics StackExchange for more details:
https://chemistry.stackexchange.com/a/122118
"""
from pathlib import Path

import urllib.request

ROOT_DIR = Path(__file__).parent.parent.parent.resolve()

if __name__ == "__main__":
    # Making sure the data path exists
    DATA_PATH = ROOT_DIR / "data" / "raw"
    DATA_PATH.mkdir(exist_ok=True, parents=True)

    # Print a warning if the file already exists
    if (DATA_PATH / "CID-SMILES.gz").exists():
        print("WARNING: The dataset already exists. Overwriting...")

    # Downloading the dataset
    print("Downloading...")
    urllib.request.urlretrieve(
        "ftp://ftp.ncbi.nlm.nih.gov/pubchem/Compound/Extras/CID-SMILES.gz",
        DATA_PATH / "CID-SMILES.gz",
    )
