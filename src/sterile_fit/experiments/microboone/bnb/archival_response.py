"""Reader for archived 60x60 MicroBooNE response matrices.

These files are diagnostic provenance inputs only.  They are not the active
2025 BNB template because their true-energy bin edges and source-flavour event
rates are not supplied by this release.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from numpy.typing import NDArray


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
