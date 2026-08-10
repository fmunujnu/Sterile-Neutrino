"""Normalize and document the archived 60x60 Reco|true matrices."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from sterile_fit.archival_response import load_archival_reco_given_true


FILES = {
    "nue_cc_fc": "HEPData-ins1953539-v3-nu_eCC_FC_Energy_Resolution.yaml",
    "nue_cc_pc": "HEPData-ins1953539-v3-nu_eCC_PC_Energy_Resolution.yaml",
    "numu_cc_fc": "HEPData-ins1953539-v3-nu_muCC_FC_Energy_Resolution.yaml",
    "numu_cc_pc": "HEPData-ins1953539-v3-nu_muCC_PC_Energy_Resolution.yaml",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-directory", type=Path, default=Path("data/raw/hepdata_microboone_2022_response"))
    parser.add_argument("--output", type=Path, default=Path("data/derived/archival_2022_reco_given_true_index.npz"))
    arguments = parser.parse_args()
    payload: dict[str, np.ndarray] = {
        "format": np.asarray("conditional_probability_reco_index_given_true_index"),
        "scope": np.asarray("archival_2022_response_only_not_active_2025_template"),
    }
    for name, filename in FILES.items():
        response = load_archival_reco_given_true(arguments.input_directory / filename)
        payload[f"{name}_reco_given_true"] = response.reco_given_true
        payload[f"{name}_raw_column_sums"] = response.raw_column_sums
        payload[f"{name}_valid_true_indices"] = response.valid_true_indices
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(arguments.output, **payload)
    print(arguments.output)


if __name__ == "__main__":
    main()
