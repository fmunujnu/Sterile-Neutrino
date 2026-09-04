"""Deterministic Gaussian approximation to the paper's pointwise CLs test.

This module deliberately does not generate pseudo-experiments.  It obtains the
first two moments of ``T = chi2_4nu - chi2_3nu`` analytically under each fixed
hypothesis and approximates each test-statistic distribution by a normal
distribution.  It is therefore an interim CLs calibration, not the paper's
Toy-MC calibration.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, isfinite
from typing import Iterable

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import cho_factor, cho_solve
from scipy.stats import norm


FloatVector = NDArray[np.float64]
FloatMatrix = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class GaussianHypothesis:
    """Mean and covariance for one fixed prediction hypothesis."""

    mean: FloatVector
    covariance: FloatMatrix

    def __post_init__(self) -> None:
        mean = np.asarray(self.mean, dtype=float)
        covariance = np.asarray(self.covariance, dtype=float)
        if mean.ndim != 1 or mean.size == 0 or not np.all(np.isfinite(mean)):
            raise ValueError("hypothesis mean must be a finite non-empty vector")
        if covariance.shape != (mean.size, mean.size):
            raise ValueError("hypothesis covariance shape must match its mean")
        if not np.all(np.isfinite(covariance)) or not np.allclose(
            covariance, covariance.T, rtol=1e-10, atol=1e-12
        ):
            raise ValueError("hypothesis covariance must be finite and symmetric")
        # Validate positive definiteness without changing or regularizing input.
        cho_factor(covariance, lower=True, check_finite=True)


@dataclass(frozen=True, slots=True)
class AsymptoticClsResult:
    """One-sided p-values and their CLs ratio for an observed test statistic."""

    test_statistic: float
    p_value_4nu: float
    p_value_3nu: float
    cls: float
    mean_under_4nu: float
    standard_deviation_under_4nu: float
    mean_under_3nu: float
    standard_deviation_under_3nu: float


def _quadratic_difference_moments(
    generated_under: GaussianHypothesis,
    null_3nu: GaussianHypothesis,
    tested_4nu: GaussianHypothesis,
) -> tuple[float, float]:
    """Return mean and variance of chi2_4nu-chi2_3nu under one hypothesis."""
    if not (
        generated_under.mean.shape == null_3nu.mean.shape == tested_4nu.mean.shape
    ):
        raise ValueError("all hypothesis means must have the same shape")

    generated_factor = cho_factor(generated_under.covariance, lower=True, check_finite=True)
    null_factor = cho_factor(null_3nu.covariance, lower=True, check_finite=True)
    tested_factor = cho_factor(tested_4nu.covariance, lower=True, check_finite=True)
    generated_cholesky = np.tril(generated_factor[0])

    offset_4nu = generated_under.mean - tested_4nu.mean
    offset_3nu = generated_under.mean - null_3nu.mean
    solved_offset_4nu = cho_solve(tested_factor, offset_4nu, check_finite=True)
    solved_offset_3nu = cho_solve(null_factor, offset_3nu, check_finite=True)
    constant = float(offset_4nu @ solved_offset_4nu - offset_3nu @ solved_offset_3nu)

    weighted_cholesky_4nu = cho_solve(tested_factor, generated_cholesky, check_finite=True)
    weighted_cholesky_3nu = cho_solve(null_factor, generated_cholesky, check_finite=True)
    whitened_quadratic = generated_cholesky.T @ (
        weighted_cholesky_4nu - weighted_cholesky_3nu
    )
    whitened_quadratic = 0.5 * (whitened_quadratic + whitened_quadratic.T)

    linear = 2.0 * (solved_offset_4nu - solved_offset_3nu)
    whitened_linear = generated_cholesky.T @ linear
    mean = constant + float(np.trace(whitened_quadratic))
    variance = 2.0 * float(np.sum(whitened_quadratic * whitened_quadratic)) + float(
        whitened_linear @ whitened_linear
    )
    if not isfinite(mean) or not isfinite(variance) or variance <= 0.0:
        raise ValueError("analytic test-statistic moments are non-finite or degenerate")
    return mean, variance


def asymptotic_cls(
    observed_test_statistic: float,
    hypothesis_pairs: Iterable[tuple[GaussianHypothesis, GaussianHypothesis]],
) -> AsymptoticClsResult:
    """Approximate pointwise CLs without Toy MC.

    Each pair is ``(null_3nu, tested_4nu)`` for an independent likelihood
    contribution.  Means and variances add because the registered analysis
    forbids separately summing correlated contributions.
    """
    if not isfinite(observed_test_statistic):
        raise ValueError("observed test statistic must be finite")
    pairs = tuple(hypothesis_pairs)
    if not pairs:
        raise ValueError("at least one hypothesis pair is required")

    mean_3nu = variance_3nu = mean_4nu = variance_4nu = 0.0
    for null_3nu, tested_4nu in pairs:
        component_mean_3nu, component_variance_3nu = _quadratic_difference_moments(
            null_3nu, null_3nu, tested_4nu
        )
        component_mean_4nu, component_variance_4nu = _quadratic_difference_moments(
            tested_4nu, null_3nu, tested_4nu
        )
        mean_3nu += component_mean_3nu
        variance_3nu += component_variance_3nu
        mean_4nu += component_mean_4nu
        variance_4nu += component_variance_4nu

    sigma_3nu = float(np.sqrt(variance_3nu))
    sigma_4nu = float(np.sqrt(variance_4nu))
    z_3nu = (observed_test_statistic - mean_3nu) / sigma_3nu
    z_4nu = (observed_test_statistic - mean_4nu) / sigma_4nu
    log_p_3nu = float(norm.logsf(z_3nu))
    log_p_4nu = float(norm.logsf(z_4nu))
    p_3nu = float(norm.sf(z_3nu))
    p_4nu = float(norm.sf(z_4nu))
    cls = min(1.0, exp(min(0.0, log_p_4nu - log_p_3nu)))
    return AsymptoticClsResult(
        test_statistic=float(observed_test_statistic),
        p_value_4nu=p_4nu,
        p_value_3nu=p_3nu,
        cls=cls,
        mean_under_4nu=mean_4nu,
        standard_deviation_under_4nu=sigma_4nu,
        mean_under_3nu=mean_3nu,
        standard_deviation_under_3nu=sigma_3nu,
    )
