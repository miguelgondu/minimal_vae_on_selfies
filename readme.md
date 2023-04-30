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

You should add the `src/` file to your **PYTHONPATH**. If you're using VSCode for running and debugging, this can be done by adding this key-value pair to your `launch.json` :
```json
"env": {
    "PYTHONPATH": "${workspaceFolder}${pathSeparator}src:${env:PYTHONPATH}"
}
```

## Data preprocessing

In `src/data_preprocessing` you can find files for downloading the dataset (which is PubChem's `CID-SMILES` saved at `ftp://ftp.ncbi.nlm.nih.gov/pubchem/Compound/Extras/CID-SMILES.gz`) and processing it to a small dataset of 5000 SELFIES strings in `data/processed/SUPER-SMALL-CID-SELFIES`, which is **already available in the repo**.

If you want access to the other datasets (for, say, a larger training run), you can run the following scripts. **Warning:** you will need plenty of disk space since the uncompressed CID-SMILES is already 8Gb.

- `download_dataset.py` lets you download the dataset and decompress it.
- `smiles_to_strings.py` reads the file at `data/raw/CID-SMILES` in chunks and progressively translates the SMILES to SELFIES using The Matter's Lab translator.[^3] Each processed chunk is appended at the end of a CID-SELFIES file in `data/processed/CID-SELFIES`.
- `small_selfies.py` filters all the SELFIES in the dataset that are larger than 300 tokens, outputting a file in `data/processed/SMALL-CID-SELFIES`. Finally, a subset of only 5000 of these SELFIES is stored in `data/processed/SUPER-SMALL-CID-SELFIES`, which is the file used for training. However, the models and training pipeline are built in such a way that training on the entire `SMALL-CID-SELFIES` should be feasible.

This figure shows the data preprocessing.

[TODO: add figure]

## Tokenizing


## Model's definition




[^1]: TODO: add cite to SELFIES.
[^2]: TODO: cite SMILES.
[^3]: https://github.com/aspuru-guzik-group/selfies