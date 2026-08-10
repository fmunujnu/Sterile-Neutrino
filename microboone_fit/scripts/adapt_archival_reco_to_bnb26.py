"""Create a 26x60 BNB-shaped Reco adapter without touching fit-core templates."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from sterile_fit.adapters.archival_reco_60_to_bnb26 import (
    ARCHIVAL_ENERGY_EDGES_GEV,
    rebin_archival_response_to_bnb26,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/derived/archival_2022_reco_given_true_index.npz"))
    parser.add_argument("--output", type=Path, default=Path("data/derived/archival_2022_reco_bnb26_given_true.npz"))
    arguments = parser.parse_args()
    with np.load(arguments.input, allow_pickle=False) as archive:
        payload: dict[str, np.ndarray] = {
            "true_energy_GeV": (ARCHIVAL_ENERGY_EDGES_GEV[:-1] + ARCHIVAL_ENERGY_EDGES_GEV[1:]) / 2.0,
            "true_energy_binning": np.asarray("user_confirmed_0_to_3_GeV_in_0p05_GeV_bins"),
            "reco_binning": np.asarray("25x[0.0,2.5)GeV_0p1_bins_plus_overflow_[2.5,3.0]GeV"),
            "scope": np.asarray("archival_response_adapter_not_a_detector_folded_2025_template"),
        }
        for channel in ("nue_cc_fc", "nue_cc_pc", "numu_cc_fc", "numu_cc_pc"):
            payload[f"{channel}_reco_bnb26_given_true"] = rebin_archival_response_to_bnb26(
                archive[f"{channel}_reco_given_true"]
            )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(arguments.output, **payload)
    print(arguments.output)


if __name__ == "__main__":
    main()
