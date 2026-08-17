"""Explicit rebinning adapter for archived 60-bin MicroBooNE responses."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


ARCHIVAL_ENERGY_EDGES_GEV = np.linspace(0.0, 3.0, 61)
"""User-confirmed 0--3 GeV / 0.05 GeV binning; YAML itself stores only indices."""


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
