"""Reproduce the public BNB four-channel data/prediction panels from HEPData."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from sterile_fit.binning import BNB_FOUR_CHANNELS
from sterile_fit.published_inputs import load_bnb_four_channel_inputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True, help="PNG destination")
    arguments = parser.parse_args()
    inputs = load_bnb_four_channel_inputs()
    # The released table has 25 bins from 0 to 2.5 GeV and one overflow bin.
    edges_GeV = np.append(np.arange(0.0, 2.6, 0.1), 3.0)
    centres_GeV = (edges_GeV[:-1] + edges_GeV[1:]) / 2.0
    figure, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
    for axis, channel in zip(axes.flat, BNB_FOUR_CHANNELS, strict=True):
        start, stop = channel.first_global_bin, channel.stop_global_bin
        observed = inputs.observed_counts[start:stop]
        total = inputs.published_total_prediction_counts[start:stop]
        background = inputs.published_background_counts[start:stop]
        error_up = inputs.observed_statistical_error_up[start:stop]
        error_down = inputs.observed_statistical_error_down[start:stop]
        axis.stairs(background, edges_GeV, label="Published background", color="tab:blue", linestyle="--")
        axis.stairs(total, edges_GeV, label="Published signal + background", color="black")
        axis.errorbar(centres_GeV, observed, yerr=[error_down, error_up], fmt="o", color="tab:red", label="Published data")
        axis.set_title(channel.identifier)
        axis.set_xlabel("Reconstructed neutrino energy [GeV]")
        axis.set_ylabel("Counts per bin")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    figure.suptitle("MicroBooNE BNB: public four-channel input reproduction")
    figure.tight_layout()
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(arguments.output, dpi=180, bbox_inches="tight")


if __name__ == "__main__":
    main()
