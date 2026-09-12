"""core/calibration.py: regrouped existing implementations; see docs/ARCHITECTURE.md."""
from __future__ import annotations

from dataclasses import dataclass
from math import exp, isfinite
from typing import Iterable
import numpy as np
from numpy.typing import NDArray
from scipy.linalg import cho_factor, cho_solve, eigh
from scipy.integrate import simpson
from scipy.stats import norm
from concurrent.futures import ThreadPoolExecutor
from math import isfinite, sqrt
from typing import Callable, Iterable, Sequence
from scipy.linalg import cholesky, solve_triangular
# Deterministic fixed-hypothesis quadratic-form calibration of pointwise CLs.


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


class QuadraticDifferenceLaw:
    """Law of T = c + sum(lambda_i*z_i**2 + b_i*z_i), z_i~N(0,1)."""

    def __init__(self, constant: float, eigenvalues, linear) -> None:
        self.constant = float(constant)
        self.eigenvalues = np.asarray(eigenvalues, dtype=float)
        self.linear = np.asarray(linear, dtype=float)
        if self.eigenvalues.shape != self.linear.shape:
            raise ValueError("quadratic and linear coefficient dimensions differ")
        self.mean = self.constant + float(np.sum(self.eigenvalues))
        variance = 2.0 * float(np.sum(self.eigenvalues**2)) + float(
            self.linear @ self.linear
        )
        self.standard_deviation = float(np.sqrt(variance))
        if not isfinite(self.standard_deviation) or self.standard_deviation <= 0.0:
            raise ValueError("quadratic-form law is non-finite or degenerate")

    def _standardized_characteristic_function(self, frequencies: FloatVector):
        frequencies = np.asarray(frequencies, dtype=float)
        result = np.empty(frequencies.size, dtype=complex)
        quadratic = self.eigenvalues / self.standard_deviation
        linear = self.linear / self.standard_deviation
        for start in range(0, frequencies.size, 256):
            frequency = frequencies[start : start + 256, None]
            denominator = 1.0 - 2.0j * frequency * quadratic
            log_characteristic = (
                -1.0j * frequency * quadratic
                - 0.5 * np.log(denominator)
                - 0.5 * frequency**2 * linear**2 / denominator
            ).sum(axis=1)
            result[start : start + 256] = np.exp(log_characteristic)
        return result

    def _truncation_bound(self, cutoff: float) -> float:
        coefficients = np.sort(
            np.abs(self.eigenvalues / self.standard_deviation)
        )[::-1]
        coefficients = coefficients[coefficients > 0.0]
        if coefficients.size:
            count = np.arange(1, coefficients.size + 1)
            logarithms = (
                -0.5 * np.cumsum(np.log(2.0 * coefficients))
                - 0.5 * count * np.log(cutoff)
                - np.log(count / 2.0)
                - np.log(np.pi)
            )
            return float(np.exp(np.min(logarithms)))
        return float(np.exp(-cutoff**2 / 2.0) / (np.pi * cutoff**2))

    def survival_probabilities(self, values) -> FloatVector:
        """Gil-Pelaez inversion with truncation and mesh-refinement checks."""
        standardized_values = (
            np.atleast_1d(np.asarray(values, dtype=float)) - self.mean
        ) / self.standard_deviation
        cutoff = 32.0
        while self._truncation_bound(cutoff) > 1.0e-9:
            cutoff *= 2.0
            if cutoff > 4096.0:
                raise ArithmeticError("quadratic characteristic function decays too slowly")
        spacing = min(
            0.025,
            np.pi / (12.0 * (1.0 + float(np.max(np.abs(standardized_values))))),
        )
        intervals = int(np.ceil(cutoff / spacing / 2.0)) * 2
        previous = None
        for _ in range(4):
            frequencies = np.linspace(0.0, cutoff, intervals + 1)
            characteristic = self._standardized_characteristic_function(frequencies)
            survival = np.empty(standardized_values.size, dtype=float)
            for start in range(0, standardized_values.size, 32):
                selected = standardized_values[start : start + 32]
                product = (
                    np.exp(-1.0j * frequencies[:, None] * selected[None, :])
                    * characteristic[:, None]
                )
                integrand = np.empty(product.shape, dtype=float)
                integrand[1:] = product.imag[1:] / frequencies[1:, None]
                integrand[0] = -selected
                survival[start : start + 32] = (
                    0.5 + simpson(integrand, x=frequencies, axis=0) / np.pi
                )
            if previous is not None and np.max(np.abs(survival - previous)) < 2.0e-7:
                if np.min(survival) < -2.0e-7 or np.max(survival) > 1.0 + 2.0e-7:
                    raise ArithmeticError("quadratic inversion produced an invalid probability")
                return np.clip(survival, 0.0, 1.0)
            previous = survival
            intervals *= 2
        raise ArithmeticError("quadratic inversion mesh did not converge")

    def survival_probability(self, value: float) -> float:
        return float(self.survival_probabilities([value])[0])


def _quadratic_difference_law(
    hypothesis_pairs: Iterable[tuple[GaussianHypothesis, GaussianHypothesis]],
    *,
    generated_under_tested: bool,
) -> QuadraticDifferenceLaw:
    constant = 0.0
    eigenvalues: list[float] = []
    linear_terms: list[float] = []
    for null_3nu, tested_4nu in hypothesis_pairs:
        generated = tested_4nu if generated_under_tested else null_3nu
        generated_cholesky = cholesky(
            generated.covariance, lower=True, check_finite=True
        )
        null_factor = cho_factor(null_3nu.covariance, lower=True, check_finite=True)
        tested_factor = cho_factor(tested_4nu.covariance, lower=True, check_finite=True)
        offset_3nu = generated.mean - null_3nu.mean
        offset_4nu = generated.mean - tested_4nu.mean
        solved_3nu = cho_solve(null_factor, offset_3nu, check_finite=True)
        solved_4nu = cho_solve(tested_factor, offset_4nu, check_finite=True)
        constant += float(offset_4nu @ solved_4nu - offset_3nu @ solved_3nu)
        matrix = generated_cholesky.T @ (
            cho_solve(tested_factor, generated_cholesky, check_finite=True)
            - cho_solve(null_factor, generated_cholesky, check_finite=True)
        )
        matrix = 0.5 * (matrix + matrix.T)
        weights, rotation = eigh(matrix, check_finite=True)
        eigenvalues.extend(weights)
        linear_terms.extend(
            rotation.T
            @ (2.0 * generated_cholesky.T @ (solved_4nu - solved_3nu))
        )
    return QuadraticDifferenceLaw(constant, eigenvalues, linear_terms)


def quadratic_cls(
    observed_test_statistic: float,
    hypothesis_pairs: Iterable[tuple[GaussianHypothesis, GaussianHypothesis]],
) -> AsymptoticClsResult:
    """Compute fixed-hypothesis CLs from the generalized quadratic-form law."""
    if not isfinite(observed_test_statistic):
        raise ValueError("observed test statistic must be finite")
    pairs = tuple(hypothesis_pairs)
    if not pairs:
        raise ValueError("at least one hypothesis pair is required")
    law_3nu = _quadratic_difference_law(pairs, generated_under_tested=False)
    law_4nu = _quadratic_difference_law(pairs, generated_under_tested=True)
    p_3nu = law_3nu.survival_probability(observed_test_statistic)
    p_4nu = law_4nu.survival_probability(observed_test_statistic)
    if p_3nu <= 0.0:
        cls = 1.0 if p_4nu <= 0.0 else float("inf")
    else:
        cls = min(1.0, p_4nu / p_3nu)
    return AsymptoticClsResult(
        test_statistic=float(observed_test_statistic),
        p_value_4nu=p_4nu,
        p_value_3nu=p_3nu,
        cls=cls,
        mean_under_4nu=law_4nu.mean,
        standard_deviation_under_4nu=law_4nu.standard_deviation,
        mean_under_3nu=law_3nu.mean,
        standard_deviation_under_3nu=law_3nu.standard_deviation,
    )


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
class PreparedFixedTestStatistic:
    """Cached fixed hypotheses with scalar and exactly batched evaluation."""

    null_hypotheses: tuple[GaussianHypothesis, ...]
    tested_hypotheses: tuple[GaussianHypothesis, ...]
    null_lower_factors: tuple[FloatMatrix, ...]
    tested_lower_factors: tuple[FloatMatrix, ...]

    def __call__(self, toy_dataset: ToyDataset) -> float:
        if len(toy_dataset) != len(self.null_hypotheses):
            raise ValueError("toy dataset and hypothesis contribution counts differ")
        total = 0.0
        for observation, null, tested, null_lower, tested_lower in zip(
            toy_dataset,
            self.null_hypotheses,
            self.tested_hypotheses,
            self.null_lower_factors,
            self.tested_lower_factors,
            strict=True,
        ):
            values = np.asarray(observation, dtype=float)
            tested_white = solve_triangular(
                tested_lower, values - tested.mean, lower=True, check_finite=False
            )
            null_white = solve_triangular(
                null_lower, values - null.mean, lower=True, check_finite=False
            )
            total += float(tested_white @ tested_white - null_white @ null_white)
        return total

    def evaluate_batch(self, draws: tuple[NDArray[np.float64], ...]) -> FloatVector:
        """Evaluate all rows using cached factors and multi-right-hand-side solves."""
        if len(draws) != len(self.null_hypotheses):
            raise ValueError("toy draws and hypothesis contribution counts differ")
        result = np.zeros(draws[0].shape[0], dtype=float)
        for values, null, tested, null_lower, tested_lower in zip(
            draws,
            self.null_hypotheses,
            self.tested_hypotheses,
            self.null_lower_factors,
            self.tested_lower_factors,
            strict=True,
        ):
            values = np.asarray(values, dtype=float)
            if values.ndim != 2 or values.shape[1] != null.mean.size:
                raise ValueError("batched toy draws have the wrong shape")
            tested_white = solve_triangular(
                tested_lower,
                (values - tested.mean[None, :]).T,
                lower=True,
                check_finite=False,
            )
            null_white = solve_triangular(
                null_lower,
                (values - null.mean[None, :]).T,
                lower=True,
                check_finite=False,
            )
            result += np.sum(tested_white * tested_white, axis=0)
            result -= np.sum(null_white * null_white, axis=0)
        if not np.all(np.isfinite(result)):
            raise RuntimeError("batched toy test statistic returned non-finite values")
        return result


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

    if isinstance(test_statistic, PreparedFixedTestStatistic):
        return test_statistic.evaluate_batch(draws)

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
    null_hypotheses = tuple(null_hypotheses)
    tested_hypotheses = tuple(tested_hypotheses)
    if not null_hypotheses or len(null_hypotheses) != len(tested_hypotheses):
        raise ValueError("equal non-empty 3nu and 4nu hypothesis lists are required")
    return PreparedFixedTestStatistic(
        null_hypotheses=null_hypotheses,
        tested_hypotheses=tested_hypotheses,
        null_lower_factors=tuple(
            cholesky(item.covariance, lower=True, check_finite=False)
            for item in null_hypotheses
        ),
        tested_lower_factors=tuple(
            cholesky(item.covariance, lower=True, check_finite=False)
            for item in tested_hypotheses
        ),
    )


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

    Production fixed-hypothesis callers pass ``PreparedFixedTestStatistic`` so
    every batch reuses the point's matrices and is evaluated as one exact
    multi-right-hand-side quadratic-form solve. Generic callables retain the
    scalar path for diagnostics.
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
    if isinstance(test_statistic, PreparedFixedTestStatistic):
        null_lower_factors = test_statistic.null_lower_factors
        tested_lower_factors = test_statistic.tested_lower_factors
    else:
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

