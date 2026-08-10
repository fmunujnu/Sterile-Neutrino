"""Covariance handling for the active BNB likelihood."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


BNB_COVARIANCE_SHAPE = [104, 104]
BNB_COVARIANCE_ORDER = "nue_cc_fc,nue_cc_pc,numu_cc_fc,numu_cc_pc; 26 reconstructed bins each"


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
                "the stored covariance artifact must describe the fixed reference matrix; "
                "parameter-dependent covariance is evaluated by the dedicated scan likelihood"
            )
        if len(self.reference_prediction_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.reference_prediction_sha256.lower()
        ):
            raise ValueError("reference_prediction_sha256 must be a SHA-256 hexadecimal digest")
        if not self.provenance.strip():
            raise ValueError("provenance must describe how the covariance was produced")
        if not np.all(np.isfinite(self.covariance)):
            raise ValueError("total covariance must contain only finite values")
        if not np.allclose(self.covariance, self.covariance.T, rtol=1e-10, atol=1e-12):
            raise ValueError("total covariance must be symmetric")


def load_declared_total_covariance(path: Path) -> DeclaredTotalCovariance:
    """Load a human-readable covariance CSV and adjacent JSON metadata.

    For ``covariance.csv`` the metadata file is
    ``covariance.metadata.json``.  No binary archive is accepted.
    """
    path = Path(path)
    if path.suffix.lower() != ".csv":
        raise ValueError("total covariance must be supplied as a visible .csv file")
    metadata_path = path.with_suffix(".metadata.json")
    if not path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f"covariance CSV and metadata JSON are required: {path}, {metadata_path}")
    covariance = np.loadtxt(path, delimiter=",", dtype=float)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    required = {
        "shape",
        "row_order",
        "column_order",
        "statistical_treatment",
        "parameter_dependence",
        "reference_prediction_sha256",
        "provenance",
    }
    missing = required.difference(metadata)
    if missing:
        raise ValueError(f"total covariance metadata is missing: {sorted(missing)}")
    if metadata["shape"] != BNB_COVARIANCE_SHAPE:
        raise ValueError(f"total covariance metadata shape must be {BNB_COVARIANCE_SHAPE}")
    if metadata["row_order"] != BNB_COVARIANCE_ORDER:
        raise ValueError("total covariance row_order does not match the active BNB channel/bin order")
    if metadata["column_order"] != "same as row_order":
        raise ValueError("total covariance column_order must be declared as identical to row_order")
    return DeclaredTotalCovariance(
        covariance=covariance,
        statistical_treatment=str(metadata["statistical_treatment"]),
        parameter_dependence=str(metadata["parameter_dependence"]),
        reference_prediction_sha256=str(metadata["reference_prediction_sha256"]),
        provenance=str(metadata["provenance"]),
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
