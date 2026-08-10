"""Reproduce the public BNB four-channel data/prediction panels from HEPData."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sterile_fit.binning import BNB_FOUR_CHANNELS
from sterile_fit.published_inputs import DEFAULT_SPECTRUM_PATH, load_bnb_four_channel_inputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True, help="PNG destination")
    arguments = parser.parse_args()
    inputs = load_bnb_four_channel_inputs()
    # The released table has 25 bins from 0 to 2.5 GeV and one overflow bin.
    edges_GeV = np.append(np.arange(0.0, 2.6, 0.1), 3.0)
    centres_GeV = (edges_GeV[:-1] + edges_GeV[1:]) / 2.0
    figure, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
    audit_rows: list[dict[str, object]] = []
    for axis, channel in zip(axes.flat, BNB_FOUR_CHANNELS, strict=True):
        start, stop = channel.first_global_bin, channel.stop_global_bin
        observed = inputs.observed_counts[start:stop]
        total = inputs.published_total_prediction_counts[start:stop]
        background = inputs.published_background_counts[start:stop]
        error_up = inputs.observed_statistical_error_up[start:stop]
        error_down = inputs.observed_statistical_error_down[start:stop]
        prediction_systematic_sigma = np.sqrt(np.diag(inputs.systematic_covariance)[start:stop])
        prediction_lower = np.maximum(total - prediction_systematic_sigma, 0.0)
        prediction_upper = total + prediction_systematic_sigma
        axis.stairs(background, edges_GeV, label="Published background", color="tab:blue", linestyle="--")
        axis.fill_between(
            edges_GeV,
            np.append(prediction_lower, prediction_lower[-1]),
            np.append(prediction_upper, prediction_upper[-1]),
            step="post",
            color="0.55",
            alpha=0.28,
            label="Published prediction systematic (diagonal)",
        )
        axis.stairs(total, edges_GeV, label="Published signal + background", color="black")
        axis.errorbar(centres_GeV, observed, yerr=[error_down, error_up], fmt="o", color="tab:red", label="Published data")
        axis.set_title(channel.identifier)
        axis.set_xlabel("Reconstructed neutrino energy [GeV]")
        axis.set_ylabel("Counts per bin")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
        for local_bin in range(26):
            audit_rows.append({
                "channel": channel.identifier,
                "channel_reco_bin": local_bin,
                "reco_energy_low_GeV": edges_GeV[local_bin],
                "reco_energy_high_GeV": edges_GeV[local_bin + 1],
                "is_overflow_bin": local_bin == 25,
                "plotted_data_counts": observed[local_bin],
                "plotted_background_counts": background[local_bin],
                "plotted_signal_plus_background_counts": total[local_bin],
                "prediction_systematic_sigma_counts": prediction_systematic_sigma[local_bin],
                "prediction_systematic_lower_counts": prediction_lower[local_bin],
                "prediction_systematic_upper_counts": prediction_upper[local_bin],
                "derived_signal_counts": total[local_bin] - background[local_bin],
            })
    figure.suptitle("MicroBooNE BNB: public four-channel input reproduction\n(last bin is the released overflow bin)")
    figure.tight_layout()
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(arguments.output, dpi=180, bbox_inches="tight")
    audit_path = arguments.output.with_suffix(".csv")
    pd.DataFrame(audit_rows).to_csv(
        audit_path,
        index=False,
        float_format="%.17g",
    )
    spectrum_source = DEFAULT_SPECTRUM_PATH
    metadata = {
        "plot_semantics": {
            "data": "HEPData Data",
            "background": "HEPData Background, plotted separately",
            "signal_plus_background": "HEPData Signal + Background, used directly; Background is not added again",
            "prediction_uncertainty_band": "plus/minus sqrt of the released systematic covariance diagonal; correlations are used by chi2 but cannot be represented by independent per-bin bars",
            "data_error_bars": "released asymmetric data statistical errors only",
            "derived_signal": "Signal + Background minus Background; audit table only",
        },
        "binning": "25 bins of width 0.1 GeV from 0 to 2.5 GeV, followed by the released overflow bin displayed to 3.0 GeV",
        "spectrum_source": str(spectrum_source),
        "spectrum_source_sha256": sha256(spectrum_source.read_bytes()).hexdigest(),
        "audit_table": str(audit_path),
        "audit_table_sha256": sha256(audit_path.read_bytes()).hexdigest(),
    }
    arguments.output.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
