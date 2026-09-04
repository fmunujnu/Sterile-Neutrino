"""core/calibration.py: regrouped existing implementations; see docs/ARCHITECTURE.md."""
from __future__ import annotations

from dataclasses import dataclass
from math import exp, isfinite
from typing import Iterable
import numpy as np
from numpy.typing import NDArray
from scipy.linalg import cho_factor, cho_solve
from scipy.stats import norm
from concurrent.futures import ThreadPoolExecutor
from math import isfinite, sqrt
from typing import Callable, Iterable, Sequence
from scipy.linalg import cholesky, solve_triangular
# Deterministic Gaussian approximation to the paper's pointwise CLs test.


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


# Empirical pointwise CLs calibration; scan parameters are fixed after the data profile.


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


def prepare_fixed_test_statistic(null_hypotheses, tested_hypotheses):
    """Freeze both predictions AND covariances at the observed-data profile point.

    The caller profiles observed data first. No optimization, parameter change,
    or data-dependent covariance update is performed inside a pseudo-experiment.
    This is the project prescription, not a confirmed collaboration implementation.
    """
    null_chi2 = prepare_fixed_hypothesis_chi2(null_hypotheses)
    tested_chi2 = prepare_fixed_hypothesis_chi2(tested_hypotheses)
    return lambda dataset: tested_chi2(dataset) - null_chi2(dataset)


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
    """Calibrate two empirical distributions of the supplied statistic.

    Production callers first profile observed data, then use
    prepare_fixed_test_statistic: the same fixed hypotheses generate and score
    Toys. This generic sampler does not itself fit or change any parameters.
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

