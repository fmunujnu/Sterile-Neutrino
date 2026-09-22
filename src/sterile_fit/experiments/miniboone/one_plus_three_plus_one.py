"""MiniBooNE appearance-only 1+3+1 scans.

The public MiniBooNE release supplies full-transmutation event tables with
reconstructed energy, true energy, true baseline and event weight.  That is
enough to replace the released two-flavour appearance probability event by
event.  It is not enough to oscillate the published backgrounds or muon
control samples, so those components retain the exact treatment used by the
validated local 3+1 reconstruction.  The active first-stage scan profiles a
declared discrete two-mass grid and the appearance CP phase at every outer
mixing-amplitude point.  Fixed-mass slices remain an optional diagnostic.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import acos, pi, sqrt
from typing import Iterable

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.optimize import minimize_scalar

from sterile_fit.core.one_plus_three_plus_one import (
    OSCILLATION_AMPLITUDE_COEFFICIENT,
    OnePlusThreePlusOneParameters,
)
from sterile_fit.experiments.miniboone.official_2020 import (
    MiniBooNE2020Data,
    chi_square_from_signals,
    gaussian_negative_two_log_likelihood_from_signals,
)


@dataclass(frozen=True, slots=True)
class AppearanceSignalTemplates:
    """Six reconstructed-bin templates spanning the two-state probability."""

    state4: NDArray[np.float64]
    state5: NDArray[np.float64]
    cross_real: NDArray[np.float64]
    cross_imag: NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class FixedMassTemplates:
    delta_m2_41_absolute_eV2: float
    delta_m2_51_eV2: float
    neutrino: AppearanceSignalTemplates
    antineutrino: AppearanceSignalTemplates


def symmetric_parameters_from_appearance_amplitudes(
    delta_m2_41_absolute_eV2: float,
    delta_m2_51_eV2: float,
    amplitude4: float,
    amplitude5: float,
    cp_phase_mue_rad: float,
) -> OnePlusThreePlusOneParameters:
    """Return a physical representative of an appearance-only degeneracy.

    ``A_i = 4 |U_ei|^2 |U_mui|^2``.  MiniBooNE appearance events cannot
    identify the e/mu factorisation, so the minimum-row-norm symmetric
    representative is used.  The core parameter object still enforces row
    normalisation and five-neutrino unitary embeddability.
    """
    for name, value in (("amplitude4", amplitude4), ("amplitude5", amplitude5)):
        if not np.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be finite and in [0, 1]")
    x4 = 0.5 * sqrt(float(amplitude4))
    x5 = 0.5 * sqrt(float(amplitude5))
    return OnePlusThreePlusOneParameters(
        float(delta_m2_41_absolute_eV2),
        float(delta_m2_51_eV2),
        x4,
        x4,
        x5,
        x5,
        float(cp_phase_mue_rad),
    )


def event_appearance_probability(
    true_energy_MeV: NDArray[np.float64],
    baseline_cm: NDArray[np.float64],
    parameters: OnePlusThreePlusOneParameters,
    *,
    antineutrino: bool,
) -> NDArray[np.float64]:
    """Vectorised mu-to-e probability for MiniBooNE's eventwise baselines."""
    energy_GeV = np.asarray(true_energy_MeV, dtype=float) / 1000.0
    baseline_km = np.asarray(baseline_cm, dtype=float) / 100000.0
    if energy_GeV.shape != baseline_km.shape or energy_GeV.ndim != 1:
        raise ValueError("true energy and baseline must be equal-length vectors")
    if np.any(energy_GeV <= 0.0) or np.any(baseline_km <= 0.0):
        raise ValueError("true energy and baseline must be strictly positive")
    phases = np.exp(
        -1j
        * OSCILLATION_AMPLITUDE_COEFFICIENT
        * np.column_stack((
            np.full_like(energy_GeV, parameters.delta_m2_41_eV2),
            np.full_like(energy_GeV, parameters.delta_m2_51_eV2),
        ))
        * baseline_km[:, None]
        / energy_GeV[:, None]
    ) - 1.0
    coefficient4 = sqrt(parameters.abs_Ue4_squared * parameters.abs_Umu4_squared)
    phase_sign = 1.0 if antineutrino else -1.0
    coefficient5 = (
        sqrt(parameters.abs_Ue5_squared * parameters.abs_Umu5_squared)
        * np.exp(1j * phase_sign * parameters.cp_phase_mue_rad)
    )
    probability = np.abs(coefficient4 * phases[:, 0] + coefficient5 * phases[:, 1]) ** 2
    tolerance = 5e-12
    if np.any(probability > 1.0 + tolerance):
        raise FloatingPointError("1+3+1 event probability exceeds one")
    return np.clip(probability, 0.0, 1.0)


def _histogram_basis(
    events: NDArray[np.float64],
    edges_MeV: NDArray[np.float64],
    delta_m2_41_absolute_eV2: float,
    delta_m2_51_eV2: float,
) -> AppearanceSignalTemplates:
    energy_GeV = events[:, 1] / 1000.0
    baseline_km = events[:, 2] / 100000.0
    phases = np.exp(
        -1j
        * OSCILLATION_AMPLITUDE_COEFFICIENT
        * np.column_stack((
            np.full_like(energy_GeV, -float(delta_m2_41_absolute_eV2)),
            np.full_like(energy_GeV, float(delta_m2_51_eV2)),
        ))
        * baseline_km[:, None]
        / energy_GeV[:, None]
    ) - 1.0
    weight = events[:, 3] / len(events)

    def histogram(values: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.histogram(events[:, 0], bins=edges_MeV, weights=weight * values)[0]

    cross = 0.5 * phases[:, 0] * np.conjugate(phases[:, 1])
    return AppearanceSignalTemplates(
        state4=histogram(0.25 * np.abs(phases[:, 0]) ** 2),
        state5=histogram(0.25 * np.abs(phases[:, 1]) ** 2),
        cross_real=histogram(cross.real),
        cross_imag=histogram(cross.imag),
    )


def build_fixed_mass_templates(
    data: MiniBooNE2020Data,
    delta_m2_41_absolute_eV2: float,
    delta_m2_51_eV2: float,
) -> FixedMassTemplates:
    if delta_m2_41_absolute_eV2 <= 0.0 or delta_m2_51_eV2 <= 0.0:
        raise ValueError("both mass-splitting magnitudes must be positive")
    return FixedMassTemplates(
        float(delta_m2_41_absolute_eV2),
        float(delta_m2_51_eV2),
        _histogram_basis(
            data.nu_full_transmutation,
            data.electron_edges_MeV,
            delta_m2_41_absolute_eV2,
            delta_m2_51_eV2,
        ),
        _histogram_basis(
            data.nubar_full_transmutation,
            data.electron_edges_MeV,
            delta_m2_41_absolute_eV2,
            delta_m2_51_eV2,
        ),
    )


def _combine_templates(
    templates: AppearanceSignalTemplates,
    amplitude4: float,
    amplitude5: float,
    cp_phase_mue_rad: float,
    *,
    antineutrino: bool,
) -> NDArray[np.float64]:
    root_product = sqrt(float(amplitude4) * float(amplitude5))
    imaginary_sign = 1.0 if antineutrino else -1.0
    signal = (
        amplitude4 * templates.state4
        + amplitude5 * templates.state5
        + root_product
        * (
            np.cos(cp_phase_mue_rad) * templates.cross_real
            + imaginary_sign * np.sin(cp_phase_mue_rad) * templates.cross_imag
        )
    )
    if np.min(signal) < -1e-9:
        raise FloatingPointError("physical appearance templates produced a negative bin")
    return np.clip(signal, 0.0, None)


def signals_from_templates(
    templates: FixedMassTemplates,
    amplitude4: float,
    amplitude5: float,
    cp_phase_mue_rad: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    # This call is the physical-boundary check; likelihood evaluation itself
    # uses only the appearance-identifiable amplitudes and phase.
    symmetric_parameters_from_appearance_amplitudes(
        templates.delta_m2_41_absolute_eV2,
        templates.delta_m2_51_eV2,
        amplitude4,
        amplitude5,
        cp_phase_mue_rad,
    )
    return (
        _combine_templates(
            templates.neutrino, amplitude4, amplitude5, cp_phase_mue_rad,
            antineutrino=False,
        ),
        _combine_templates(
            templates.antineutrino, amplitude4, amplitude5, cp_phase_mue_rad,
            antineutrino=True,
        ),
    )


def _minimum_physical_absolute_phase(amplitude4: float, amplitude5: float) -> float:
    if amplitude4 == 0.0 or amplitude5 == 0.0:
        return 0.0
    root4 = sqrt(float(amplitude4))
    root5 = sqrt(float(amplitude5))
    limit = 1.0 + 2.0 * (1.0 - root4 - root5) / (root4 * root5)
    if limit >= 1.0:
        return 0.0
    if limit < -1.0 - 1e-12:
        raise ValueError("appearance amplitudes have no unitary symmetric representative")
    return float(acos(np.clip(limit, -1.0, 1.0)))


def profile_cp_phase(
    data: MiniBooNE2020Data,
    templates: FixedMassTemplates,
    amplitude4: float,
    amplitude5: float,
    *,
    phase_grid_points: int = 25,
    statistic: str = "gaussian-nll",
) -> tuple[float, float, NDArray[np.float64], NDArray[np.float64]]:
    """Continuously profile the CP phase after a deterministic coarse search."""
    if phase_grid_points < 9:
        raise ValueError("phase_grid_points must be at least 9")
    minimum_absolute = _minimum_physical_absolute_phase(amplitude4, amplitude5)
    segments = (
        [(-pi, pi)] if minimum_absolute == 0.0
        else [(-pi, -minimum_absolute), (minimum_absolute, pi)]
    )

    statistic_function = {
        "gaussian-nll": gaussian_negative_two_log_likelihood_from_signals,
        "chi-square": chi_square_from_signals,
    }.get(statistic)
    if statistic_function is None:
        raise ValueError("statistic must be 'gaussian-nll' or 'chi-square'")

    def objective(phase: float) -> float:
        wrapped = float((phase + pi) % (2.0 * pi) - pi)
        signal_nu, signal_nubar = signals_from_templates(
            templates, amplitude4, amplitude5, wrapped
        )
        return statistic_function(data, signal_nu, signal_nubar)

    candidates: list[tuple[float, float]] = []
    per_segment = max(5, phase_grid_points // len(segments))
    for low, high in segments:
        grid = np.linspace(low, high, per_segment)
        values = np.asarray([objective(float(value)) for value in grid])
        for index, (phase, value) in enumerate(zip(grid, values)):
            candidates.append((float(value), float(phase)))
            if 0 < index < len(grid) - 1 and value <= values[index - 1] and value <= values[index + 1]:
                refined = minimize_scalar(
                    objective,
                    bounds=(float(grid[index - 1]), float(grid[index + 1])),
                    method="bounded",
                    options={"xatol": 1e-6},
                )
                candidates.append((float(refined.fun), float(refined.x)))
    best_value, best_phase = min(candidates, key=lambda pair: pair[0])
    best_phase = float((best_phase + pi) % (2.0 * pi) - pi)
    signal_nu, signal_nubar = signals_from_templates(
        templates, amplitude4, amplitude5, best_phase
    )
    return best_phase, best_value, signal_nu, signal_nubar


def _physical_phase_grid(
    amplitude4: float,
    amplitude5: float,
    points: int,
) -> NDArray[np.float64]:
    if points < 5:
        raise ValueError("phase profile grid must contain at least five points")
    minimum_absolute = _minimum_physical_absolute_phase(amplitude4, amplitude5)
    if minimum_absolute == 0.0:
        return np.linspace(-pi, pi, points)
    negative_points = points // 2 + 1
    positive_points = points - negative_points + 1
    return np.unique(np.r_[
        np.linspace(-pi, -minimum_absolute, negative_points),
        np.linspace(minimum_absolute, pi, positive_points),
    ])


def build_mass_template_bank(
    data: MiniBooNE2020Data,
    mass_values_eV2: NDArray[np.float64],
) -> dict[tuple[float, float], FixedMassTemplates]:
    """Precompute all event histograms once for a discrete mass profile grid."""
    masses = np.asarray(mass_values_eV2, dtype=float)
    if masses.ndim != 1 or len(masses) < 2 or np.any(masses <= 0.0):
        raise ValueError("mass profile grid must contain at least two positive values")
    bank: dict[tuple[float, float], FixedMassTemplates] = {}
    total = len(masses) ** 2
    for index, q41 in enumerate(masses):
        for q51 in masses:
            bank[(float(q41), float(q51))] = build_fixed_mass_templates(
                data, float(q41), float(q51)
            )
        print(f"MiniBooNE 1+3+1 mass templates: {(index + 1) * len(masses)}/{total}")
    return bank


def profile_mass_grid_and_cp_phase(
    data: MiniBooNE2020Data,
    template_bank: dict[tuple[float, float], FixedMassTemplates],
    amplitude4: float,
    amplitude5: float,
    *,
    coarse_phase_points: int = 5,
    refined_mass_candidates: int = 10,
    refined_phase_grid_points: int = 13,
) -> tuple[float, float, float, float, NDArray[np.float64], NDArray[np.float64]]:
    """Profile a discrete two-mass grid, then continuously refine its best phases."""
    if refined_mass_candidates < 1:
        raise ValueError("refined_mass_candidates must be positive")
    phases = _physical_phase_grid(amplitude4, amplitude5, coarse_phase_points)
    coarse: list[tuple[float, float, tuple[float, float]]] = []
    for mass_pair, templates in template_bank.items():
        best_value = np.inf
        best_phase = 0.0
        for phase in phases:
            signal_nu = _combine_templates(
                templates.neutrino, amplitude4, amplitude5, float(phase),
                antineutrino=False,
            )
            signal_nubar = _combine_templates(
                templates.antineutrino, amplitude4, amplitude5, float(phase),
                antineutrino=True,
            )
            value = chi_square_from_signals(data, signal_nu, signal_nubar)
            if value < best_value:
                best_value = value
                best_phase = float(phase)
        coarse.append((best_value, best_phase, mass_pair))

    candidates = sorted(coarse, key=lambda row: row[0])[:refined_mass_candidates]
    refined: list[
        tuple[float, float, float, float, NDArray[np.float64], NDArray[np.float64]]
    ] = []
    for _, _, (q41, q51) in candidates:
        phase, value, signal_nu, signal_nubar = profile_cp_phase(
            data,
            template_bank[(q41, q51)],
            amplitude4,
            amplitude5,
            phase_grid_points=refined_phase_grid_points,
            statistic="chi-square",
        )
        refined.append((q41, q51, phase, value, signal_nu, signal_nubar))
    return min(refined, key=lambda row: row[3])


def scan_profiled_mixing_plane(
    data: MiniBooNE2020Data,
    amplitudes: NDArray[np.float64],
    mass_values_eV2: NDArray[np.float64],
    *,
    coarse_phase_points: int = 5,
    refined_mass_candidates: int = 10,
    refined_phase_grid_points: int = 13,
    report_every: int = 25,
) -> pd.DataFrame:
    """Scan A4-A5 while profiling both mass splittings and the CP phase."""
    amplitudes = np.asarray(amplitudes, dtype=float)
    if amplitudes.ndim != 1 or len(amplitudes) < 2:
        raise ValueError("amplitudes must be a one-dimensional grid with at least two points")
    if np.any(amplitudes <= 0.0) or np.any(amplitudes > 1.0):
        raise ValueError("amplitude grid must lie in (0, 1]")
    bank = build_mass_template_bank(data, mass_values_eV2)
    rows = []
    total = len(amplitudes) ** 2
    completed = 0
    for amplitude4 in amplitudes:
        for amplitude5 in amplitudes:
            q41, q51, phase, chi2, signal_nu, signal_nubar = profile_mass_grid_and_cp_phase(
                data,
                bank,
                float(amplitude4),
                float(amplitude5),
                coarse_phase_points=coarse_phase_points,
                refined_mass_candidates=refined_mass_candidates,
                refined_phase_grid_points=refined_phase_grid_points,
            )
            rows.append((
                float(amplitude4), float(amplitude5), q41, q51, phase, chi2,
                float(signal_nu.sum()), float(signal_nubar.sum()),
            ))
            completed += 1
            if report_every and (completed % report_every == 0 or completed == total):
                print(f"MiniBooNE 1+3+1 profiled mixing plane: {completed}/{total} points")
    minimum = min(row[5] for row in rows)
    return pd.DataFrame(
        [(*row, row[5] - minimum) for row in rows],
        columns=(
            "appearance_amplitude_state4",
            "appearance_amplitude_state5",
            "profiled_delta_m2_41_absolute_eV2",
            "profiled_delta_m2_51_eV2",
            "profiled_cp_phase_mue_rad",
            "chi_square",
            "nu_signal_total",
            "nubar_signal_total",
            "delta_chi_square",
        ),
    )


def scan_fixed_mass_product_plane(
    data: MiniBooNE2020Data,
    mixing_products: NDArray[np.float64],
    *,
    delta_m2_41_absolute_eV2: float = 0.9,
    delta_m2_51_eV2: float = 0.5,
    cp_phase_mue_rad: float = 0.0,
    report_every: int = 500,
) -> pd.DataFrame:
    """Scan the two appearance products at fixed masses and CP phase.

    The axes are ``p_i = |U_ei U_mui|`` and therefore the internal
    appearance amplitudes are exactly ``A_i = 4 p_i**2``.  MiniBooNE's
    released appearance signal has no dependence on how either product is
    factorised into its electron and muon matrix elements.  Those individual
    factors are consequently flat, non-identifiable coordinates: profiling
    them is analytically trivial and cannot change this chi-square surface.
    """
    products = np.asarray(mixing_products, dtype=float)
    if products.ndim != 1 or len(products) < 2:
        raise ValueError("mixing products must be a one-dimensional grid")
    if np.any(products <= 0.0) or np.any(products > 0.5):
        raise ValueError("mixing products must lie in (0, 0.5]")
    templates = build_fixed_mass_templates(
        data, delta_m2_41_absolute_eV2, delta_m2_51_eV2
    )
    rows: list[tuple[float, ...]] = []
    total = len(products) ** 2
    for index, product4 in enumerate(products):
        for product5 in products:
            amplitude4 = 4.0 * float(product4) ** 2
            amplitude5 = 4.0 * float(product5) ** 2
            signal_nu, signal_nubar = signals_from_templates(
                templates, amplitude4, amplitude5, cp_phase_mue_rad
            )
            chi2 = chi_square_from_signals(data, signal_nu, signal_nubar)
            rows.append((
                float(product4), float(product5), amplitude4, amplitude5,
                float(delta_m2_41_absolute_eV2), float(delta_m2_51_eV2),
                float(cp_phase_mue_rad), float(chi2),
                float(signal_nu.sum()), float(signal_nubar.sum()),
            ))
        completed = (index + 1) * len(products)
        if report_every and (completed % report_every < len(products) or completed == total):
            print(f"MiniBooNE fixed-mass 1+3+1 product plane: {completed}/{total} points")
    minimum = min(row[7] for row in rows)
    return pd.DataFrame(
        [(*row, row[7] - minimum) for row in rows],
        columns=(
            "absolute_Ue4_Umu4",
            "absolute_Ue5_Umu5",
            "appearance_amplitude_state4",
            "appearance_amplitude_state5",
            "delta_m2_41_absolute_eV2",
            "delta_m2_51_eV2",
            "cp_phase_mue_rad",
            "chi_square",
            "nu_signal_total",
            "nubar_signal_total",
            "delta_chi_square",
        ),
    )


def scan_fixed_mass_slices(
    data: MiniBooNE2020Data,
    mass_pairs_eV2: Iterable[tuple[float, float]],
    amplitudes: NDArray[np.float64],
    *,
    phase_grid_points: int = 25,
    report_every: int = 500,
) -> pd.DataFrame:
    """Scan A4-A5 planes and profile only the appearance CP phase."""
    amplitudes = np.asarray(amplitudes, dtype=float)
    if amplitudes.ndim != 1 or len(amplitudes) < 2:
        raise ValueError("amplitudes must be a one-dimensional grid with at least two points")
    if np.any(amplitudes <= 0.0) or np.any(amplitudes > 1.0):
        raise ValueError("amplitude grid must lie in (0, 1]")
    mass_pairs = tuple((float(q41), float(q51)) for q41, q51 in mass_pairs_eV2)
    rows: list[tuple[float, ...]] = []
    total = len(mass_pairs) * len(amplitudes) ** 2
    completed = 0
    for q41, q51 in mass_pairs:
        templates = build_fixed_mass_templates(data, q41, q51)
        slice_start = len(rows)
        for amplitude4 in amplitudes:
            for amplitude5 in amplitudes:
                phase, nll, signal_nu, signal_nubar = profile_cp_phase(
                    data,
                    templates,
                    float(amplitude4),
                    float(amplitude5),
                    phase_grid_points=phase_grid_points,
                )
                rows.append((
                    q41, q51, float(amplitude4), float(amplitude5), phase,
                    nll, float(signal_nu.sum()), float(signal_nubar.sum()),
                ))
                completed += 1
                if report_every and (completed % report_every == 0 or completed == total):
                    print(f"MiniBooNE 1+3+1: {completed}/{total} points")
        slice_values = [row[5] for row in rows[slice_start:]]
        minimum = min(slice_values)
        for index in range(slice_start, len(rows)):
            rows[index] = (*rows[index], rows[index][5] - minimum)
    return pd.DataFrame(rows, columns=(
        "delta_m2_41_absolute_eV2",
        "delta_m2_51_eV2",
        "appearance_amplitude_state4",
        "appearance_amplitude_state5",
        "profiled_cp_phase_mue_rad",
        "negative_2_log_likelihood",
        "nu_signal_total",
        "nubar_signal_total",
        "delta_negative_2_log_likelihood_within_mass_slice",
    ))
