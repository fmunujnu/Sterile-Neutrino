"""LSND final-2001 public-record adapter; it intentionally has no likelihood.

The collaboration's final likelihood needs event-level four-variable PDFs and
background variations that are not released as machine-readable inputs.  This
module may only expose printed scalar facts and the exact common 3+1
appearance-probability mapping; it must never infer disappearance information.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import numpy as np
from numpy.typing import NDArray

from sterile_fit.core.three_plus_one import (
    ThreePlusOneParameters, short_baseline_appearance_probability,
)
from sterile_fit.paths import REPOSITORY_ROOT

PUBLISHED = REPOSITORY_ROOT / "data/experiments/lsnd/published/final_2001_summary.csv"


@dataclass(frozen=True, slots=True)
class LSNDFinalPublicSummary:
    dar_excess_events: float
    dar_probability_percent: float
    best_fit_delta_m2_eV2: float
    best_fit_sin2_2theta_mue: float
    center_baseline_m: float


def load_published_summary(path: Path = PUBLISHED) -> LSNDFinalPublicSummary:
    """Read the scalar paper transcription, rejecting a mistaken grid input."""
    with path.open(encoding="utf-8", newline="") as stream:
        rows = {row["quantity"]: row for row in csv.DictReader(stream)}
    required = {
        "dar_antinu_mu_to_antinu_e_excess", "dar_oscillation_probability_percent",
        "four_dimensional_best_fit_amplitude", "four_dimensional_best_fit_delta_m2",
        "dar_source_to_detector_center_distance",
    }
    if set(rows) < required:
        raise ValueError("LSND public summary is incomplete")
    result = LSNDFinalPublicSummary(
        dar_excess_events=float(rows["dar_antinu_mu_to_antinu_e_excess"]["value"]),
        dar_probability_percent=float(rows["dar_oscillation_probability_percent"]["value"]),
        best_fit_delta_m2_eV2=float(rows["four_dimensional_best_fit_delta_m2"]["value"]),
        best_fit_sin2_2theta_mue=float(rows["four_dimensional_best_fit_amplitude"]["value"]),
        center_baseline_m=float(rows["dar_source_to_detector_center_distance"]["value"]),
    )
    if result.best_fit_delta_m2_eV2 <= 0 or not 0 <= result.best_fit_sin2_2theta_mue <= 1:
        raise ValueError("LSND printed best-fit coordinate is outside appearance bounds")
    return result


def three_plus_one_parameters_from_appearance_amplitude(
    delta_m2_41_eV2: float, sin2_2theta_mue: float,
) -> ThreePlusOneParameters:
    """Select an appearance-only representative with exact requested amplitude.

    With ``sin²(theta14)=1/2`` and ``sin²(theta24)=A``, the active core gives
    ``4 |Ue4|² |Umu4|² = 4*(1/2)*(1-1/2)*A = A``. LSND's published
    two-flavour fit cannot choose another point on this degeneracy, so callers
    may not reinterpret it as a disappearance prediction.
    """
    if not 0.0 <= sin2_2theta_mue <= 1.0:
        raise ValueError("sin2_2theta_mue must be in [0, 1]")
    return ThreePlusOneParameters(delta_m2_41_eV2, 0.5, sin2_2theta_mue)


def three_plus_one_dar_appearance_probability(
    energy_MeV: NDArray[np.float64], baseline_m: NDArray[np.float64] | float,
    delta_m2_eV2: float, sin2_2theta_mue: float,
) -> NDArray[np.float64]:
    """Call the common core with explicit LSND paper units (MeV, m)."""
    return short_baseline_appearance_probability(
        three_plus_one_parameters_from_appearance_amplitude(delta_m2_eV2, sin2_2theta_mue),
        np.asarray(energy_MeV, dtype=float) / 1000.0,
        np.asarray(baseline_m, dtype=float) / 1000.0,
    )


def core_mapping_rows(summary: LSNDFinalPublicSummary) -> list[dict[str, float]]:
    """A small deterministic unit/convention check, not an LSND event prediction."""
    energies = np.array([20.0, 30.0, 40.0, 50.0, 52.8])
    probabilities = three_plus_one_dar_appearance_probability(
        energies, summary.center_baseline_m, summary.best_fit_delta_m2_eV2,
        summary.best_fit_sin2_2theta_mue,
    )
    return [
        {"energy_MeV": float(energy), "baseline_m": summary.center_baseline_m,
         "delta_m2_eV2": summary.best_fit_delta_m2_eV2,
         "sin2_2theta_mue": summary.best_fit_sin2_2theta_mue,
         "appearance_probability": float(probability)}
        for energy, probability in zip(energies, probabilities, strict=True)
    ]
