"""Create a 26x60 BNB-shaped Reco adapter without touching fit-core templates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sterile_fit.experiments.microboone.bnb.adapters.archival_reco_60_to_bnb26 import (
    ARCHIVAL_ENERGY_EDGES_GEV,
    rebin_archival_response_to_bnb26,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-directory",
        type=Path,
        default=Path("data/experiments/microboone/bnb/derived/archival_2022_reco_given_true"),
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("data/experiments/microboone/bnb/derived/archival_2022_reco_bnb26_given_true"),
    )
    arguments = parser.parse_args()
    arguments.output_directory.mkdir(parents=True, exist_ok=True)
    true_energy = (ARCHIVAL_ENERGY_EDGES_GEV[:-1] + ARCHIVAL_ENERGY_EDGES_GEV[1:]) / 2.0
    pd.DataFrame({
        "true_bin": np.arange(60, dtype=int),
        "true_energy_GeV": true_energy,
    }).to_csv(arguments.output_directory / "true_energy_GeV.csv", index=False, float_format="%.17g")
    for channel in ("nue_cc_fc", "nue_cc_pc", "numu_cc_fc", "numu_cc_pc"):
        source = pd.read_csv(arguments.input_directory / f"{channel}_reco_given_true.csv")
        if list(source.columns)[0] != "reco_bin" or source.shape != (60, 61):
            raise ValueError(f"unexpected visible response table for {channel}")
        rebinned = rebin_archival_response_to_bnb26(source.iloc[:, 1:].to_numpy(dtype=float))
        table = pd.DataFrame(rebinned, columns=[f"true_bin_{index:03d}" for index in range(60)])
        table.insert(0, "reco_bin", np.arange(26, dtype=int))
        table.to_csv(
            arguments.output_directory / f"{channel}_reco_given_true.csv",
            index=False,
            float_format="%.17g",
        )
    metadata = {
        "format": "conditional_probability_bnb26_reco_bin_given_true_bin",
        "shape": [26, 60],
        "true_energy_binning": "0 to 3 GeV in 60 bins of 0.05 GeV",
        "reconstructed_energy_binning": "25 bins of 0.1 GeV below 2.5 GeV plus [2.5,3.0] GeV overflow",
        "normalization": "each nonzero true-energy column sums to one; zero columns remain zero",
    }
    (arguments.output_directory / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(arguments.output_directory)


if __name__ == "__main__":
    main()
