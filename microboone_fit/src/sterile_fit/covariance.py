"""Covariance handling for the active BNB likelihood."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


def symmetric_error_variance(
    statistical_error_up: NDArray[np.float64],
    statistical_error_down: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Diagonal variance from released asymmetric errors.

    This is a declared approximation, not a substitute for the collaboration's
    exact CNP construction. Profile results using it must retain that label.
    """
    up = np.asarray(statistical_error_up, dtype=float)
    down = np.asarray(statistical_error_down, dtype=float)
    if up.shape != down.shape or np.any(up < 0.0) or np.any(down < 0.0):
        raise ValueError("statistical errors must be non-negative vectors of equal shape")
    return ((up + down) / 2.0) ** 2


def total_covariance_with_released_statistical_errors(
    systematic_covariance: NDArray[np.float64],
    statistical_error_up: NDArray[np.float64],
    statistical_error_down: NDArray[np.float64],
) -> NDArray[np.float64]:
    covariance = np.asarray(systematic_covariance, dtype=float).copy()
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError("systematic covariance must be square")
    variance = symmetric_error_variance(statistical_error_up, statistical_error_down)
    if variance.shape[0] != covariance.shape[0]:
        raise ValueError("statistical errors do not match covariance dimension")
    return covariance + np.diag(variance)


@dataclass(frozen=True, slots=True)
class DeclaredTotalCovariance:
    """A total covariance whose statistical prescription is explicitly recorded."""

    covariance: NDArray[np.float64]
    statistical_treatment: str
    parameter_dependence: str
    reference_prediction_sha256: str
    provenance: str

    def __post_init__(self) -> None:
        if self.covariance.shape != (104, 104):
            raise ValueError("active BNB total covariance must have shape (104, 104)")
        if not self.statistical_treatment.strip():
            raise ValueError("statistical_treatment must be a non-empty declaration")
        if self.parameter_dependence != "fixed_at_reference":
            raise ValueError(
                "active BNB likelihood only supports a declared covariance fixed at the reference; "
                "a parameter-dependent covariance requires a dedicated likelihood implementation"
            )
        if len(self.reference_prediction_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.reference_prediction_sha256.lower()
        ):
            raise ValueError("reference_prediction_sha256 must be a SHA-256 hexadecimal digest")
        if not self.provenance.strip():
            raise ValueError("provenance must describe how the covariance was produced")
        if not np.allclose(self.covariance, self.covariance.T, rtol=1e-10, atol=1e-12):
            raise ValueError("total covariance must be symmetric")


def load_declared_total_covariance(path: Path) -> DeclaredTotalCovariance:
    """Load an externally prepared total covariance for a profile run.

    The archive must contain its reference-prediction hash and provenance in
    addition to the covariance and statistical prescription. This binds a
    fixed covariance to the spectrum against which it was constructed.
    """
    with np.load(path, allow_pickle=False) as archive:
        required = {
            "covariance",
            "statistical_treatment",
            "parameter_dependence",
            "reference_prediction_sha256",
            "provenance",
        }
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"total covariance archive is missing: {sorted(missing)}")
        covariance = np.asarray(archive["covariance"], dtype=float)
        treatment = str(np.asarray(archive["statistical_treatment"]).item())
        dependence = str(np.asarray(archive["parameter_dependence"]).item())
        reference_hash = str(np.asarray(archive["reference_prediction_sha256"]).item())
        provenance = str(np.asarray(archive["provenance"]).item())
    return DeclaredTotalCovariance(
        covariance=covariance,
        statistical_treatment=treatment,
        parameter_dependence=dependence,
        reference_prediction_sha256=reference_hash,
        provenance=provenance,
    )


def prediction_sha256(predicted_counts: NDArray[np.float64]) -> str:
    """Digest an exact float64 reference spectrum in declared bin order."""
    prediction = np.asarray(predicted_counts, dtype="<f8")
    if prediction.shape != (104,) or not np.all(np.isfinite(prediction)) or np.any(prediction < 0.0):
        raise ValueError("reference prediction must be a finite non-negative 104-bin vector")
    return sha256(prediction.tobytes()).hexdigest()


def combined_neyman_pearson_variance(
    observed_counts: NDArray[np.float64], predicted_counts: NDArray[np.float64]
) -> NDArray[np.float64]:
    """CNP data-statistical diagonal: 3 / (1/M + 2/P), bin by bin."""
    observed = np.asarray(observed_counts, dtype=float)
    predicted = np.asarray(predicted_counts, dtype=float)
    if observed.shape != predicted.shape or observed.ndim != 1:
        raise ValueError("CNP observed and predicted counts must be equal-shape vectors")
    if not np.all(np.isfinite(observed)) or not np.all(np.isfinite(predicted)):
        raise ValueError("CNP counts must be finite")
    if np.any(observed < 0.0) or np.any(predicted < 0.0):
        raise ValueError("CNP counts must be non-negative")
    with np.errstate(divide="ignore", invalid="ignore"):
        denominator = np.divide(1.0, observed) + np.divide(2.0, predicted)
        variance = np.divide(3.0, denominator, out=np.zeros_like(denominator), where=denominator > 0.0)
    return variance


def pearson_statistical_variance(reference_prediction_counts: NDArray[np.float64]) -> NDArray[np.float64]:
    """Pearson data-statistical diagonal evaluated at a fixed expected spectrum."""
    prediction = np.asarray(reference_prediction_counts, dtype=float)
    if prediction.shape != (104,) or not np.all(np.isfinite(prediction)) or np.any(prediction < 0.0):
        raise ValueError("Pearson reference prediction must be a finite non-negative 104-bin vector")
    return prediction.copy()


def cnp_total_covariance_at_reference(
    systematic_covariance: NDArray[np.float64],
    observed_counts: NDArray[np.float64],
    reference_prediction_counts: NDArray[np.float64],
    *,
    provenance: str,
) -> DeclaredTotalCovariance:
    """Add released systematics to CNP data statistics at one fixed reference."""
    systematic = np.asarray(systematic_covariance, dtype=float)
    if systematic.shape != (104, 104):
        raise ValueError("systematic covariance must have shape (104, 104)")
    variance = combined_neyman_pearson_variance(observed_counts, reference_prediction_counts)
    return DeclaredTotalCovariance(
        covariance=systematic + np.diag(variance),
        statistical_treatment="CNP data statistic: variance_i = 3/(1/M_i + 2/P_i)",
        parameter_dependence="fixed_at_reference",
        reference_prediction_sha256=prediction_sha256(reference_prediction_counts),
        provenance=provenance,
    )


def pearson_total_covariance_at_reference(
    systematic_covariance: NDArray[np.float64],
    reference_prediction_counts: NDArray[np.float64],
    *,
    provenance: str,
) -> DeclaredTotalCovariance:
    """Add Pearson data statistics to released systematics at one fixed reference."""
    systematic = np.asarray(systematic_covariance, dtype=float)
    if systematic.shape != (104, 104):
        raise ValueError("systematic covariance must have shape (104, 104)")
    variance = pearson_statistical_variance(reference_prediction_counts)
    return DeclaredTotalCovariance(
        covariance=systematic + np.diag(variance),
        statistical_treatment="Pearson data statistic: variance_i = P_i",
        parameter_dependence="fixed_at_reference",
        reference_prediction_sha256=prediction_sha256(reference_prediction_counts),
        provenance=provenance,
    )


def solve_quadratic_form(residual: NDArray[np.float64], covariance: NDArray[np.float64]) -> float:
    """Compute rᵀC⁻¹r without explicit inversion or silent regularization."""
    vector = np.asarray(residual, dtype=float)
    matrix = np.asarray(covariance, dtype=float)
    if matrix.shape != (vector.size, vector.size):
        raise ValueError("covariance shape does not match residual")
    if not np.allclose(matrix, matrix.T, rtol=1e-10, atol=1e-12):
        raise ValueError("covariance must be symmetric")
    try:
        cholesky = np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError as error:
        raise ValueError("total covariance is not positive definite; diagnose it explicitly before fitting") from error
    whitened = np.linalg.solve(cholesky, vector)
    return float(whitened @ whitened)
