"""Empirical pointwise CLs calibration with profile fits in every toy.

The physics prediction and covariance prescription are supplied by the active
analysis.  This module only generates pseudo-data, repeats the caller's test
statistic calculation, and counts right-tail probabilities.  It never replaces
the empirical distributions by a Gaussian or chi-square approximation.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from math import isfinite, sqrt
from typing import Callable, Iterable, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import cholesky, solve_triangular

from .asymptotic_cls import GaussianHypothesis


FloatVector = NDArray[np.float64]
ToyDataset = tuple[FloatVector, ...]
ToyTestStatistic = Callable[[ToyDataset], float]
FixedHypothesisChi2 = Callable[[ToyDataset], float]


@dataclass(frozen=True, slots=True)
class ToyClsResult:
    """Empirical CLs result and auditable Monte Carlo diagnostics."""

    observed_test_statistic: float
    number_of_toys_per_hypothesis: int
    right_tail_count_under_4nu: int
    right_tail_count_under_3nu: int
    p_value_4nu: float
    p_value_3nu: float
    cls: float
    p_value_4nu_standard_error: float
    p_value_3nu_standard_error: float
    cls_standard_error_delta_method: float
    test_statistics_under_4nu: FloatVector
    test_statistics_under_3nu: FloatVector


def _draw_gaussian_toys(
    hypotheses: Sequence[GaussianHypothesis],
    lower_cholesky_factors: Sequence[NDArray[np.float64]],
    number_of_toys: int,
    random_generators: Sequence[np.random.Generator],
) -> tuple[NDArray[np.float64], ...]:
    """Draw independent registered contributions using their full covariance."""
    draws: list[NDArray[np.float64]] = []
    for hypothesis, lower, generator in zip(
        hypotheses, lower_cholesky_factors, random_generators, strict=True
    ):
        standard_normal = generator.standard_normal(
            (number_of_toys, hypothesis.mean.size)
        )
        draws.append(hypothesis.mean[None, :] + standard_normal @ lower.T)
    return tuple(draws)


def fixed_hypothesis_chi2(
    toy_dataset: ToyDataset,
    hypotheses: Sequence[GaussianHypothesis],
) -> float:
    """Evaluate a sum of independent Gaussian quadratic forms."""
    if len(toy_dataset) != len(hypotheses):
        raise ValueError("toy dataset and hypothesis contribution counts differ")
    total = 0.0
    for observation, hypothesis in zip(toy_dataset, hypotheses, strict=True):
        residual = np.asarray(observation, dtype=float) - hypothesis.mean
        if residual.shape != hypothesis.mean.shape or not np.all(np.isfinite(residual)):
            raise ValueError("toy observation shape or values are invalid")
        lower = cholesky(hypothesis.covariance, lower=True, check_finite=False)
        whitened = solve_triangular(lower, residual, lower=True, check_finite=False)
        total += float(whitened @ whitened)
    return total


def prepare_fixed_hypothesis_chi2(
    hypotheses: Iterable[GaussianHypothesis],
) -> FixedHypothesisChi2:
    """Precompute fixed-hypothesis Cholesky factors for repeated Toy fits."""
    prepared_hypotheses = tuple(hypotheses)
    if not prepared_hypotheses:
        raise ValueError("at least one fixed Gaussian hypothesis is required")
    lower_factors = tuple(
        cholesky(item.covariance, lower=True, check_finite=False)
        for item in prepared_hypotheses
    )

    def evaluate(toy_dataset: ToyDataset) -> float:
        if len(toy_dataset) != len(prepared_hypotheses):
            raise ValueError("toy dataset and hypothesis contribution counts differ")
        total = 0.0
        for observation, hypothesis, lower in zip(
            toy_dataset, prepared_hypotheses, lower_factors, strict=True
        ):
            residual = np.asarray(observation, dtype=float) - hypothesis.mean
            if residual.shape != hypothesis.mean.shape or not np.all(np.isfinite(residual)):
                raise ValueError("toy observation shape or values are invalid")
            whitened = solve_triangular(
                lower, residual, lower=True, check_finite=False
            )
            total += float(whitened @ whitened)
        return total

    return evaluate


def _evaluate_toys(
    draws: tuple[NDArray[np.float64], ...],
    test_statistic: ToyTestStatistic,
    *,
    workers: int,
) -> FloatVector:
    number_of_toys = draws[0].shape[0]

    def evaluate(index: int) -> float:
        value = float(test_statistic(tuple(component[index] for component in draws)))
        if not isfinite(value):
            raise RuntimeError("toy test statistic returned a non-finite value")
        return value

    if workers == 1:
        values = [evaluate(index) for index in range(number_of_toys)]
    else:
        # Threads keep analysis callables and detector kernels in one process.
        # executor.map preserves input order, so the result is seed-reproducible.
        with ThreadPoolExecutor(max_workers=workers) as executor:
            values = list(executor.map(evaluate, range(number_of_toys)))
    return np.asarray(values, dtype=float)


def _empirical_right_tail(
    values: FloatVector,
    threshold: float,
) -> tuple[int, float, float]:
    """Count P(T >= threshold) with the finite-ensemble plus-one correction."""
    count = int(np.count_nonzero(values >= threshold))
    denominator = values.size + 1
    probability = (count + 1) / denominator
    standard_error = sqrt(probability * (1.0 - probability) / denominator)
    return count, probability, standard_error


def toy_cls(
    observed_test_statistic: float,
    null_3nu_hypotheses: Iterable[GaussianHypothesis],
    tested_4nu_hypotheses: Iterable[GaussianHypothesis],
    test_statistic: ToyTestStatistic,
    *,
    number_of_toys: int,
    seed: int,
    workers: int = 1,
    batch_size: int = 256,
) -> ToyClsResult:
    """Calibrate pointwise CLs from two empirical profiled-toy distributions.

    ``test_statistic`` must repeat the complete fit used on real data.  In the
    MicroBooNE scans this means profiling the unplotted physical mixing
    parameter for every toy while keeping the two displayed coordinates fixed.
    The tested-hypothesis toy generator is the observed-data profiled 4nu point
    (the documented plug-in nuisance prescription).
    """
    if not isfinite(observed_test_statistic):
        raise ValueError("observed test statistic must be finite")
    if number_of_toys < 2:
        raise ValueError("number_of_toys must be at least 2 per hypothesis")
    if workers < 1:
        raise ValueError("workers must be at least 1")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    null_hypotheses = tuple(null_3nu_hypotheses)
    tested_hypotheses = tuple(tested_4nu_hypotheses)
    if not null_hypotheses or len(null_hypotheses) != len(tested_hypotheses):
        raise ValueError("equal non-empty 3nu and 4nu hypothesis lists are required")
    for null, tested in zip(null_hypotheses, tested_hypotheses, strict=True):
        if null.mean.shape != tested.mean.shape:
            raise ValueError("paired 3nu and 4nu hypotheses must have equal dimensions")

    # Separate deterministic streams make results independent of evaluation order.
    seed_sequence = np.random.SeedSequence(seed)
    seed_3nu, seed_4nu = seed_sequence.spawn(2)
    null_lower_factors = tuple(
        cholesky(item.covariance, lower=True, check_finite=False)
        for item in null_hypotheses
    )
    tested_lower_factors = tuple(
        cholesky(item.covariance, lower=True, check_finite=False)
        for item in tested_hypotheses
    )

    def draw_and_evaluate(
        hypotheses: Sequence[GaussianHypothesis],
        lower_factors: Sequence[NDArray[np.float64]],
        child_seed: np.random.SeedSequence,
    ) -> FloatVector:
        generators = tuple(
            np.random.default_rng(component_seed)
            for component_seed in child_seed.spawn(len(hypotheses))
        )
        values = np.empty(number_of_toys, dtype=float)
        for start in range(0, number_of_toys, batch_size):
            stop = min(start + batch_size, number_of_toys)
            draws = _draw_gaussian_toys(
                hypotheses, lower_factors, stop - start, generators
            )
            values[start:stop] = _evaluate_toys(
                draws, test_statistic, workers=workers
            )
        return values

    values_3nu = draw_and_evaluate(
        null_hypotheses, null_lower_factors, seed_3nu
    )
    values_4nu = draw_and_evaluate(
        tested_hypotheses, tested_lower_factors, seed_4nu
    )

    count_4nu, p_4nu, error_4nu = _empirical_right_tail(
        values_4nu, observed_test_statistic
    )
    count_3nu, p_3nu, error_3nu = _empirical_right_tail(
        values_3nu, observed_test_statistic
    )
    cls = min(1.0, p_4nu / p_3nu)
    relative_variance = (error_4nu / p_4nu) ** 2 + (error_3nu / p_3nu) ** 2
    cls_error = cls * sqrt(relative_variance)
    return ToyClsResult(
        observed_test_statistic=float(observed_test_statistic),
        number_of_toys_per_hypothesis=number_of_toys,
        right_tail_count_under_4nu=count_4nu,
        right_tail_count_under_3nu=count_3nu,
        p_value_4nu=p_4nu,
        p_value_3nu=p_3nu,
        cls=cls,
        p_value_4nu_standard_error=error_4nu,
        p_value_3nu_standard_error=error_3nu,
        cls_standard_error_delta_method=cls_error,
        test_statistics_under_4nu=values_4nu,
        test_statistics_under_3nu=values_3nu,
    )
