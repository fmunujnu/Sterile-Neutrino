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


@dataclass(frozen=True, slots=True)
class PredictionScaledGaussianLikelihood:
    """Gaussian chi2 with a prediction-dependent covariance approximation.

    The released systematic covariance is absolute counts squared at the
    nominal prediction.  Following the paper's statement that fractional
    systematic uncertainty remains constant as the oscillated prediction
    changes, each element is rescaled by ``P_i/P0_i * P_j/P0_j``.  Pearson
    data statistics, ``diag(P)``, are then evaluated at the current prediction.

    The release does not separate non-neutrino and out-of-fiducial covariance
    components, so this whole-spectrum scaling is an explicit public-data
    approximation rather than the collaboration's exact component update.
    """

    observed_counts: NDArray[np.float64]
    reference_prediction_counts: NDArray[np.float64]
    reference_systematic_covariance: NDArray[np.float64]

    def __post_init__(self) -> None:
        observed = np.asarray(self.observed_counts, dtype=float)
        reference = np.asarray(self.reference_prediction_counts, dtype=float)
        systematic = np.asarray(self.reference_systematic_covariance, dtype=float)
        if observed.ndim != 1 or observed.size == 0 or reference.shape != observed.shape:
            raise ValueError("observed and reference predictions must be equal, non-empty vectors")
        if systematic.shape != (observed.size, observed.size):
            raise ValueError("reference systematic covariance shape must match the count vectors")
        if not np.all(np.isfinite(observed)) or np.any(observed < 0.0):
            raise ValueError("observed counts must be finite and non-negative")
        if not np.all(np.isfinite(reference)) or np.any(reference <= 0.0):
            raise ValueError("reference prediction must be finite and strictly positive for covariance scaling")
        if not np.allclose(systematic, systematic.T, rtol=1e-10, atol=1e-12):
            raise ValueError("reference systematic covariance must be symmetric")

    def covariance_for_prediction(self, predicted_counts: NDArray[np.float64]) -> NDArray[np.float64]:
        prediction = np.asarray(predicted_counts, dtype=float)
        if prediction.shape != self.reference_prediction_counts.shape or not np.all(np.isfinite(prediction)) or np.any(prediction <= 0.0):
            raise ValueError("prediction must be a finite strictly-positive 104-bin vector")
        ratio = prediction / self.reference_prediction_counts
        systematic = self.reference_systematic_covariance * ratio[:, None] * ratio[None, :]
        return systematic + np.diag(prediction)

    def chi2(self, predicted_counts: NDArray[np.float64]) -> float:
        prediction = np.asarray(predicted_counts, dtype=float)
        covariance = self.covariance_for_prediction(prediction)
        return solve_quadratic_form(self.observed_counts - prediction, covariance)
