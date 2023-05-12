"""
This script loads the dataset `CID-SELFIES`, and analyses
the length (in tokens) of all the molecules in it, outputting
a histogram of the lengths in static/hist_of_lengths_CID-SELFIES.jpg
"""
from pathlib import Path
from collections import defaultdict
import json

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from utils.tokens import from_selfie_to_tokens

ROOT_DIR = Path(__file__).parent.parent.parent.resolve()


def save_counts(dataset_name: str = "CID-SELFIES"):
    """
    Loads the dataset in data/processed/{dataset_name},
    saves a JSON file with pairs (length in tokens, count)
    for all the molecules, and saves a histogram plot.
    """
    # Load the dataset in chunks
    dataset_path = ROOT_DIR / "data" / "processed" / f"{dataset_name}"

    # Get the lengths of all the molecules, storing them
    # in a dictionary (memory efficiency)
    lengths = defaultdict(int)
    for chunk in pd.read_csv(dataset_path, chunksize=100000, header=None, sep="\t"):
        print(chunk)
        for molecule in chunk[1]:
            lengths[len(from_selfie_to_tokens(molecule))] += 1

        # Save the lengths to a file
        with open(
            ROOT_DIR / "data" / "processed" / f"lengths_{dataset_name}.json", "w"
        ) as fp:
            json.dump(lengths, fp)


def plot_histogram(dataset_name: str = "CID-SELFIES"):
    """
    Plots the histogram of the lengths of the molecules
    by loading the JSON file saved by save_counts.
    """
    # Load the lengths
    with open(
        ROOT_DIR / "data" / "processed" / f"lengths_{dataset_name}.json", "r"
    ) as fp:
        lengths = json.load(fp)

    # Sort items by key, and store them in a list
    lengths = [(int(k), v) for k, v in lengths.items()]
    lengths = sorted(lengths, key=lambda x: x[0])

    sequence_lengths = np.array([k for k, _ in lengths])
    sequence_counts = np.array([v for _, v in lengths])

    # Plot the histogram
    # sns.set_theme()
    fig, (ax_barplot, ax_cummulative) = plt.subplots(
        2, 1, figsize=(1 * 6, 2 * 4), sharex=True
    )
    sns.barplot(
        x=sequence_lengths,
        y=sequence_counts,
        ax=ax_barplot,
    )
    ax_barplot.set_ylabel("Count")

    # Plot the cummulative histogram
    cummulative = np.cumsum(sequence_counts)
    sns.lineplot(x=sequence_lengths, y=cummulative, ax=ax_cummulative)
    ax_cummulative.set_ylabel("Cummulative count")
    ax_cummulative.set_xlabel("Length in tokens")

    # The x-axis is too crowded, so we only show the ticks
    # every 100 tokens
    xticks = ax_barplot.get_xticks()
    ax_barplot.set_xticks(xticks[::100])
    ax_barplot.set_xticklabels(xticks[::100])

    # Determining where 99% of the data is
    total_count = np.sum(sequence_counts)
    cummulative_count = 0
    for seq_length, seq_count in zip(sequence_lengths, sequence_counts):
        cummulative_count += seq_count
        if cummulative_count / total_count > 0.99:
            print(f"99% of the data is in molecules of length  <= {seq_length}")
            break
    ax_barplot.set_xlim([0, seq_length + 100])

    # Print the length of the longest sequence
    print(f"The longest sequence has length {sequence_lengths[-1]}")

    # Drawing a vertical line at the 99% mark
    ax_cummulative.axvline(x=seq_length, color="red", linestyle="--")

    fig.savefig(
        ROOT_DIR / "static" / f"hist_of_lengths_{dataset_name}.jpg",
        dpi=300,
        bbox_inches="tight",
    )


if __name__ == "__main__":
    # save_counts()
    plot_histogram()
