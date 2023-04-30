# Minimal VAE on SELFIES

SELFIES are a robust and discrete representation of molecules[^1], which are a sort of successor to SMILES[^2]. This repository contains a minimal working example of an MLP variational autoencoder that can be trained on SELFIES, including how to download a database of SELFIES strings, how to process these as categorical data, the training loops, and an analysis of the resulting latent space.

## Prerequisites

This code was written with Python 3.9 in mind. If you are using conda, try

```sh
conda create -n minimal-vae-on-selfies python=3.9
```

following with

```sh
pip install -r requirements.txt
```

This `requirements.txt` file includes `RDKit`, which might be tricky to install depending on your OS. Make sure everything is setup properly by running

```
python -c "import rdkit; print(rdkit.__version__)"
```

If you run into trouble, [check RDKit's documentation here](https://www.rdkit.org/docs/Install.html).

## Data preprocessing

In `src/data_preprocessing` you can find files for downloading the dataset (which is PubChem's `CID-SMILES` saved at `ftp://ftp.ncbi.nlm.nih.gov/pubchem/Compound/Extras/CID-SMILES.gz`) and processing it to a SELFIES dataset, extracting tokens as well.

- `download_dataset.py` lets you download the dataset and decompress it. **Warning**: you will need 8Gb of disk space, since the decompressed CID-SMILES file is quite large.
- `smiles_to_strings.py` reads the file at `data/raw/CID-SMILES` in chuncks, and progressively translates the SMILES to SELFIES using The Matter's Lab translator.[^3] Each processed chunck is appended at the end of a CID-SELFIES file in `data/processed/CID-SELFIES`.
- `getting_all_tokens.py` reads `data/processed/CID-SELFIES` by chuncks and saves, at each iteration, a json file with all the tokens it found in format `{token: count}`.




[^1]: TODO: add cite to SELFIES.
[^2]: TODO: cite SMILES.
[^3]: https://github.com/aspuru-guzik-group/selfies