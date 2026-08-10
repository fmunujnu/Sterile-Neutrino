"""Numerical fitting in the named 3+1 parameter coordinates."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Callable, Iterable, Mapping

import numpy as np
from scipy.optimize import differential_evolution

from .parameters import ThreePlusOneParameters
Objective = Callable[[ThreePlusOneParameters], float]
PARAMETER_NAMES = ("delta_m2_41_eV2", "sin2_theta14", "sin2_theta24")
PROFILE_MIXING_BOUNDS = {
    "sin2_theta14": (0.0, 0.5),
    "sin2_theta24": (0.0, 1.0),
}


@dataclass(frozen=True, slots=True)
class FitPoint:
    parameters: ThreePlusOneParameters
    chi2: float


@dataclass(frozen=True, slots=True)
class ProfileResult:
    fixed_parameters: Mapping[str, float]
    best_fit: FitPoint
    optimizer_message: str


@dataclass(frozen=True, slots=True)
class AppearanceAmplitudeProfileResult:
    """Profile result at fixed Δm²41 and exact sin²(2θμe).

    The two stored scan coordinates are not members of
    :class:`ThreePlusOneParameters`; the best-fit point records the physical
    ``sin2_theta14`` and derived, constrained ``sin2_theta24``.
    """

    delta_m2_41_eV2: float
    sin2_2theta_mue: float
    best_fit: FitPoint
    optimizer_message: str


def _validate_fixed_parameters(fixed_parameters: Mapping[str, float]) -> None:
    unknown = set(fixed_parameters).difference(PARAMETER_NAMES)
    if unknown:
        raise ValueError(f"unknown 3+1 profile parameter names: {sorted(unknown)}")
    values = {
        "delta_m2_41_eV2": fixed_parameters.get("delta_m2_41_eV2", 1.0),
        "sin2_theta14": fixed_parameters.get("sin2_theta14", 0.0),
        "sin2_theta24": fixed_parameters.get("sin2_theta24", 0.0),
    }
    ThreePlusOneParameters(**values)
    if values["sin2_theta14"] > PROFILE_MIXING_BOUNDS["sin2_theta14"][1]:
        raise ValueError("3+1 profile uses the conventional small-mixing branch sin2_theta14 <= 0.5")


def profile_three_plus_one(
    objective: Objective,
    fixed_parameters: Mapping[str, float],
    *,
    delta_m2_bounds_eV2: tuple[float, float] = (1e-2, 1e2),
    seed: int = 42,
) -> ProfileResult:
    """Compute min χ² over every non-fixed physical 3+1 parameter.

    A profile point is not a fixed-grid evaluation: all parameters absent from
    `fixed_parameters` are globally minimized with bounded differential
    evolution. Δm² is explored in log10 space; mixing variables are explored
    directly as sin²(theta), so their names and bounds remain physical.
    """
    _validate_fixed_parameters(fixed_parameters)
    lower, upper = delta_m2_bounds_eV2
    if lower <= 0.0 or upper <= lower:
        raise ValueError("delta_m2_bounds_eV2 must be positive and increasing")
    fixed = {name: float(value) for name, value in fixed_parameters.items()}
    free_names = [name for name in PARAMETER_NAMES if name not in fixed]

    def unpack(vector: np.ndarray) -> ThreePlusOneParameters:
        values = dict(fixed)
        for name, value in zip(free_names, vector, strict=True):
            values[name] = float(10.0 ** value) if name == "delta_m2_41_eV2" else float(value)
        return ThreePlusOneParameters(**values)

    if not free_names:
        point = ThreePlusOneParameters(**fixed)
        return ProfileResult(fixed, FitPoint(point, float(objective(point))), "no free parameters")

    bounds = [
        (np.log10(lower), np.log10(upper)) if name == "delta_m2_41_eV2" else PROFILE_MIXING_BOUNDS[name]
        for name in free_names
    ]
    result = differential_evolution(
        lambda vector: objective(unpack(np.asarray(vector, dtype=float))),
        bounds=bounds,
        seed=seed,
        polish=True,
        workers=1,
        updating="immediate",
    )
    if not result.success or not np.isfinite(result.fun):
        raise RuntimeError(f"profile optimization failed: {result.message}")
    return ProfileResult(fixed, FitPoint(unpack(np.asarray(result.x, dtype=float)), float(result.fun)), result.message)


def prefit_three_plus_one(objective: Objective, *, seed: int = 42) -> FitPoint:
    """Global 3+1 minimum, expressed as a profile with zero fixed parameters."""
    return profile_three_plus_one(objective, {}, seed=seed).best_fit


def profile_s14_s24_at_fixed_sin2_2theta_mue(
    objective: Objective,
    *,
    delta_m2_41_eV2: float,
    sin2_2theta_mue: float,
    seed: int = 42,
) -> AppearanceAmplitudeProfileResult:
    """Profile the physical 3+1 mixing freedom at fixed appearance amplitude.

    For the active rotation convention,

    ``sin²(2θμe) = 4 s14 (1-s14) s24``,

    where ``s14 = sin²θ14`` and ``s24 = sin²θ24``.  Thus a non-zero fixed
    appearance amplitude leaves one physical degree of freedom, ``s14``;
    ``s24`` is derived rather than independently minimized.  The optimiser is
    restricted to ``0 <= s14 <= 0.5`` and ``0 <= s24 <= 1``.  The s14 bound
    selects the conventional branch used when expressing the fit through
    sin²(2θee); the discarded branch duplicates that disappearance amplitude.
    This is a profile likelihood, not the historical arbitrary slice
    ``θ14 = θ24``.
    """
    # Reuse the public parameter validation for the mass-squared coordinate.
    ThreePlusOneParameters(delta_m2_41_eV2, 0.0, 0.0)
    if not np.isfinite(sin2_2theta_mue) or not 0.0 <= sin2_2theta_mue <= 1.0:
        raise ValueError("sin2_2theta_mue must be in [0, 1]")

    def minimise_on_branch(
        make_parameters: Callable[[float], ThreePlusOneParameters],
        bounds: tuple[float, float],
    ) -> tuple[FitPoint, str]:
        result = differential_evolution(
            lambda vector: objective(make_parameters(float(vector[0]))),
            bounds=[bounds],
            seed=seed,
            polish=True,
            workers=1,
            updating="immediate",
        )
        if not result.success or not np.isfinite(result.fun):
            raise RuntimeError(f"appearance-amplitude profile optimization failed: {result.message}")
        return FitPoint(make_parameters(float(result.x[0])), float(result.fun)), result.message

    if sin2_2theta_mue == 0.0:
        # The zero-amplitude boundary is a union of the two branches admitted
        # by the conventional s14 <= 0.5 parameterization.
        candidates = (
            minimise_on_branch(
                lambda s14: ThreePlusOneParameters(delta_m2_41_eV2, s14, 0.0),
                PROFILE_MIXING_BOUNDS["sin2_theta14"],
            ),
            minimise_on_branch(
                lambda s24: ThreePlusOneParameters(delta_m2_41_eV2, 0.0, s24),
                PROFILE_MIXING_BOUNDS["sin2_theta24"],
            ),
        )
        best_point, message = min(candidates, key=lambda item: item[0].chi2)
        return AppearanceAmplitudeProfileResult(delta_m2_41_eV2, sin2_2theta_mue, best_point, f"zero-amplitude boundary; {message}")

    if sin2_2theta_mue == 1.0:
        point = ThreePlusOneParameters(delta_m2_41_eV2, 0.5, 1.0)
        return AppearanceAmplitudeProfileResult(
            delta_m2_41_eV2,
            sin2_2theta_mue,
            FitPoint(point, float(objective(point))),
            "unique unit-amplitude boundary point",
        )

    # s24 <= 1 implies 4*s14*(1-s14) >= sin2_2theta_mue.
    root = np.sqrt(1.0 - sin2_2theta_mue)
    s14_lower = (1.0 - root) / 2.0
    s14_upper = PROFILE_MIXING_BOUNDS["sin2_theta14"][1]

    def constrained_parameters(s14: float) -> ThreePlusOneParameters:
        denominator = 4.0 * s14 * (1.0 - s14)
        s24 = sin2_2theta_mue / denominator
        # Floating-point roundoff can exceed the endpoint by a few ulps.
        return ThreePlusOneParameters(delta_m2_41_eV2, s14, float(np.clip(s24, 0.0, 1.0)))

    result = differential_evolution(
        lambda vector: objective(constrained_parameters(float(vector[0]))),
        bounds=[(s14_lower, s14_upper)],
        seed=seed,
        polish=True,
        workers=1,
        updating="immediate",
    )
    if not result.success or not np.isfinite(result.fun):
        raise RuntimeError(f"appearance-amplitude profile optimization failed: {result.message}")
    best_fit = FitPoint(constrained_parameters(float(result.x[0])), float(result.fun))
    return AppearanceAmplitudeProfileResult(delta_m2_41_eV2, sin2_2theta_mue, best_fit, result.message)


def profile_appearance_amplitude_grid(
    objective: Objective,
    delta_m2_values_eV2: Iterable[float],
    sin2_2theta_mue_values: Iterable[float],
    *,
    seed: int = 42,
) -> list[AppearanceAmplitudeProfileResult]:
    """Profile every point of the exact sin²(2θμe)--Δm²41 scan plane."""
    delta_values = tuple(float(value) for value in delta_m2_values_eV2)
    amplitude_values = tuple(float(value) for value in sin2_2theta_mue_values)
    if not delta_values or not amplitude_values:
        raise ValueError("both appearance-amplitude scan axes must be non-empty")
    return [
        profile_s14_s24_at_fixed_sin2_2theta_mue(
            objective,
            delta_m2_41_eV2=delta_m2,
            sin2_2theta_mue=amplitude,
            seed=seed + index,
        )
        for index, (delta_m2, amplitude) in enumerate(product(delta_values, amplitude_values))
    ]


def profile_grid(
    objective: Objective,
    scan_axes: Mapping[str, Iterable[float]],
    *,
    seed: int = 42,
) -> list[ProfileResult]:
    """Profile each Cartesian scan point over all remaining 3+1 parameters."""
    _validate_fixed_parameters({name: 1.0 if name == "delta_m2_41_eV2" else 0.0 for name in scan_axes})
    axis_names = tuple(scan_axes)
    axis_values = tuple(tuple(float(value) for value in values) for values in scan_axes.values())
    if not axis_names or any(not values for values in axis_values):
        raise ValueError("scan_axes must contain at least one non-empty named axis")
    results: list[ProfileResult] = []
    for index, values in enumerate(product(*axis_values)):
        fixed = dict(zip(axis_names, values, strict=True))
        results.append(profile_three_plus_one(objective, fixed, seed=seed + index))
    return results
