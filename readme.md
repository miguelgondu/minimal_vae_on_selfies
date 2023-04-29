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






[^1]: TODO: add cite to SELFIES.
[^2]: TODO: cite SMILES.