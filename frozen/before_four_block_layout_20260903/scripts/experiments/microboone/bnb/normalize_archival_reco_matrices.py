"""Normalize and document the archived 60x60 Reco|true matrices."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sterile_fit.experiments.microboone.bnb.archival_response import load_archival_reco_given_true


FILES = {
    "nue_cc_fc": "HEPData-ins1953539-v3-nu_eCC_FC_Energy_Resolution.yaml",
    "nue_cc_pc": "HEPData-ins1953539-v3-nu_eCC_PC_Energy_Resolution.yaml",
    "numu_cc_fc": "HEPData-ins1953539-v3-nu_muCC_FC_Energy_Resolution.yaml",
    "numu_cc_pc": "HEPData-ins1953539-v3-nu_muCC_PC_Energy_Resolution.yaml",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-directory", type=Path, default=Path("data/experiments/microboone/bnb/raw_response"))
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("data/experiments/microboone/bnb/derived/archival_2022_reco_given_true"),
    )
    arguments = parser.parse_args()
    arguments.output_directory.mkdir(parents=True, exist_ok=True)
    for name, filename in FILES.items():
        response = load_archival_reco_given_true(arguments.input_directory / filename)
        columns = [f"true_bin_{index:03d}" for index in range(60)]
        table = pd.DataFrame(response.reco_given_true, columns=columns)
        table.insert(0, "reco_bin", np.arange(60, dtype=int))
        table.to_csv(
            arguments.output_directory / f"{name}_reco_given_true.csv",
            index=False,
            float_format="%.17g",
        )
        pd.DataFrame({
            "true_bin": np.arange(60, dtype=int),
            "raw_column_sum": response.raw_column_sums,
            "is_nonzero_column": np.isin(np.arange(60), response.valid_true_indices),
        }).to_csv(
            arguments.output_directory / f"{name}_column_diagnostics.csv",
            index=False,
            float_format="%.17g",
        )
    metadata = {
        "format": "conditional_probability_reco_bin_given_true_bin",
        "shape": [60, 60],
        "normalization": "each nonzero true-energy column sums to one; zero columns remain zero",
        "scope": "archival 2022 response; used only as the declared migration prior",
    }
    (arguments.output_directory / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(arguments.output_directory)


if __name__ == "__main__":
    main()
