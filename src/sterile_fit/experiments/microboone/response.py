"""experiments/microboone/response.py: regrouped existing implementations; see docs/ARCHITECTURE.md."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import yaml
from numpy.typing import NDArray
import argparse
import json
import pandas as pd
# Reader for archived 60x60 MicroBooNE response matrices.


@dataclass(frozen=True, slots=True)
class ArchivedRecoGivenTrueMatrix:
    """A normalized conditional response R[reco index, true index]."""

    reco_given_true: NDArray[np.float64]
    raw_column_sums: NDArray[np.float64]
    valid_true_indices: NDArray[np.int64]


def load_archival_reco_given_true(path: Path) -> ArchivedRecoGivenTrueMatrix:
    """Read HEPData 114862's indexed response and normalize non-empty columns.

    Each non-empty column is normalized independently, because the table labels
    the two coordinates as true-energy and reconstructed-energy bin indices.
    Normalizing rows would reverse the conditional probability.
    """
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    dependent = document["dependent_variables"][0]["values"]
    coordinates = {entry["header"]["name"]: entry["values"] for entry in document["independent_variables"]}
    true_indices = np.asarray([entry["value"] for entry in coordinates["true neutrino energy bin index"]], dtype=int)
    reco_indices = np.asarray([entry["value"] for entry in coordinates["reco neutrino energy bin index"]], dtype=int)
    values = np.asarray([entry["value"] for entry in dependent], dtype=float)
    if values.size != 60 * 60 or true_indices.shape != values.shape or reco_indices.shape != values.shape:
        raise ValueError("archival response must provide one value for every 60x60 index coordinate")
    if np.any(true_indices < 0) or np.any(true_indices >= 60) or np.any(reco_indices < 0) or np.any(reco_indices >= 60):
        raise ValueError("archival response indices must lie in [0, 59]")
    pairs = np.stack((reco_indices, true_indices), axis=1)
    if np.unique(pairs, axis=0).shape[0] != 60 * 60:
        raise ValueError("archival response contains duplicate or missing coordinates")
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("archival response probabilities must be finite and non-negative")
    matrix = np.zeros((60, 60), dtype=float)
    matrix[reco_indices, true_indices] = values
    raw_column_sums = matrix.sum(axis=0)
    valid = raw_column_sums > 0.0
    matrix[:, valid] /= raw_column_sums[valid]
    return ArchivedRecoGivenTrueMatrix(matrix, raw_column_sums, np.flatnonzero(valid))


# Explicit rebinning adapter for archived 60-bin MicroBooNE responses.


ARCHIVAL_ENERGY_EDGES_GEV = np.linspace(0.0, 3.0, 61)
# User-confirmed 0--3 GeV / 0.05 GeV binning; YAML itself stores only indices.


def bnb26_reco_aggregation() -> NDArray[np.float64]:
    """Map 60 reconstructed bins to 25 bins below 2.5 GeV plus one overflow.

    Output bins 0--24 each sum two adjacent 0.05 GeV input bins.
    Output bin 25 collects every input bin with reconstructed energy >=2.5 GeV.
    """
    aggregation = np.zeros((26, 60), dtype=float)
    output_indices = np.arange(25)
    aggregation[output_indices, 2 * output_indices] = 1.0
    aggregation[output_indices, 2 * output_indices + 1] = 1.0
    aggregation[25, 50:] = 1.0
    return aggregation


def rebin_archival_response_to_bnb26(reco_given_true: NDArray[np.float64]) -> NDArray[np.float64]:
    """Rebin only the reconstructed axis while preserving each true-bin column.

    A zero true-energy column remains zero, by linearity. This adapter does
    not invent source rates, efficiencies, backgrounds, or any flavour process.
    """
    response = np.asarray(reco_given_true, dtype=float)
    if response.shape != (60, 60):
        raise ValueError("archival response must have shape (60, 60)")
    if not np.all(np.isfinite(response)) or np.any(response < 0.0):
        raise ValueError("archival response must be finite and non-negative")
    rebinned = bnb26_reco_aggregation() @ response
    if not np.allclose(rebinned.sum(axis=0), response.sum(axis=0), rtol=1e-12, atol=1e-12):
        raise RuntimeError("rebinning failed to preserve true-bin response sums")
    return rebinned


# Normalize and document the archived 60x60 Reco|true matrices.


NORMALIZE_FILES = {
    "nue_cc_fc": "HEPData-ins1953539-v3-nu_eCC_FC_Energy_Resolution.yaml",
    "nue_cc_pc": "HEPData-ins1953539-v3-nu_eCC_PC_Energy_Resolution.yaml",
    "numu_cc_fc": "HEPData-ins1953539-v3-nu_muCC_FC_Energy_Resolution.yaml",
    "numu_cc_pc": "HEPData-ins1953539-v3-nu_muCC_PC_Energy_Resolution.yaml",
}


def prepare_normalized_response() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-directory", type=Path, default=Path("data/experiments/microboone/bnb/raw_response"))
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("data/experiments/microboone/bnb/derived/archival_2022_reco_given_true"),
    )
    arguments = parser.parse_args()
    arguments.output_directory.mkdir(parents=True, exist_ok=True)
    for name, filename in NORMALIZE_FILES.items():
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


# Create a 26x60 BNB-shaped Reco adapter without touching fit-core templates.


def prepare_bnb26_response() -> None:
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



