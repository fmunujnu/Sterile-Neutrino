"""Likelihoods are independent of model and detector-kernel implementations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .covariance import solve_quadratic_form


@dataclass(frozen=True, slots=True)
class GaussianBinnedLikelihood:
    observed_counts: NDArray[np.float64]
    covariance: NDArray[np.float64]

    def __post_init__(self) -> None:
        observed = np.asarray(self.observed_counts, dtype=float)
        if observed.ndim != 1 or not np.all(np.isfinite(observed)) or np.any(observed < 0.0):
            raise ValueError("observed_counts must be a finite non-negative one-dimensional count vector")

    def chi2(self, predicted_counts: NDArray[np.float64]) -> float:
        prediction = np.asarray(predicted_counts, dtype=float)
        if prediction.shape != self.observed_counts.shape:
            raise ValueError("prediction shape does not match observed data")
        if not np.all(np.isfinite(prediction)) or np.any(prediction < 0.0):
            raise ValueError("prediction must contain finite non-negative expected counts")
        return solve_quadratic_form(self.observed_counts - prediction, self.covariance)
