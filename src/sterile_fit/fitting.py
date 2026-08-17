"""Numerical fitting in the named 3+1 parameter coordinates."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Callable, Iterable, Mapping

import numpy as np
from scipy.optimize import differential_evolution, minimize_scalar

from .parameters import ThreePlusOneParameters
Objective = Callable[[ThreePlusOneParameters], float]
PARAMETER_NAMES = ("delta_m2_41_eV2", "sin2_theta14", "sin2_theta24")
PROFILE_MIXING_BOUNDS = {
    "sin2_theta14": (0.0, 1.0),
    "sin2_theta24": (0.0, 1.0),
}


def _profile_bounded_scalar(
    objective: Callable[[float], float],
    bounds: tuple[float, float],
    *,
    grid_points: int = 33,
) -> tuple[float, float, str]:
    """Deterministically locate and polish every sampled one-dimensional basin."""
    lower, upper = bounds
    if not np.isfinite(lower) or not np.isfinite(upper) or upper < lower:
        raise ValueError("scalar profile bounds must be finite and increasing")
    if upper == lower:
        value = float(objective(lower))
        return lower, value, "unique scalar boundary point"
    coordinates = np.linspace(lower, upper, grid_points)
    values = np.asarray([objective(float(value)) for value in coordinates], dtype=float)
    if not np.all(np.isfinite(values)):
        raise RuntimeError("scalar profile objective returned a non-finite value")
    candidates: list[tuple[float, float]] = [
        (float(coordinates[0]), float(values[0])),
        (float(coordinates[-1]), float(values[-1])),
    ]
    local_indices = [
        index
        for index in range(1, grid_points - 1)
        if values[index] <= values[index - 1] and values[index] <= values[index + 1]
    ]
    if not local_indices:
        local_indices = [int(np.argmin(values[1:-1])) + 1]
    for index in local_indices:
        result = minimize_scalar(
            objective,
            bounds=(float(coordinates[index - 1]), float(coordinates[index + 1])),
            method="bounded",
            options={"xatol": 1e-10},
        )
        if result.success and np.isfinite(result.fun):
            candidates.append((float(result.x), float(result.fun)))
    best_coordinate, best_value = min(candidates, key=lambda item: item[1])
    return best_coordinate, best_value, f"deterministic {grid_points}-point basin search plus bounded polishing"


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


@dataclass(frozen=True, slots=True)
class ElectronDisappearanceProfileResult:
    """Profile result at fixed Δm²41 and exact sin²(2θee)."""

    delta_m2_41_eV2: float
    sin2_2theta_ee: float
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
    """Best 3+1 fit including lower-dimensional zero-appearance boundaries.

    A continuous global optimiser is not guaranteed to land exactly on
    ``s24=0`` or ``s14=0`` even when the physical minimum lies there.  Profile
    the full volume and both zero-appearance boundary surfaces explicitly, then
    return the lowest candidate.  Multiple seeds are still required by callers
    because the mass-squared direction is oscillatory.
    """
    candidates = (
        profile_three_plus_one(objective, {}, seed=seed).best_fit,
        profile_three_plus_one(objective, {"sin2_theta24": 0.0}, seed=seed + 10_000).best_fit,
        profile_three_plus_one(objective, {"sin2_theta14": 0.0}, seed=seed + 20_000).best_fit,
    )
    return min(candidates, key=lambda point: point.chi2)


def profile_s14_s24_at_fixed_sin2_2theta_mue(
    objective: Objective,
    *,
    delta_m2_41_eV2: float,
    sin2_2theta_mue: float,
) -> AppearanceAmplitudeProfileResult:
    """Profile the physical 3+1 mixing freedom at fixed appearance amplitude.

    For the active rotation convention,

    ``sin²(2θμe) = 4 s14 (1-s14) s24``,

    where ``s14 = sin²θ14`` and ``s24 = sin²θ24``.  Thus a non-zero fixed
    appearance amplitude leaves one physical degree of freedom, ``s14``;
    ``s24`` is derived rather than independently minimized.  The optimiser is
    searched over the complete unitary domain ``0 <= s14 <= 1`` and
    ``0 <= s24 <= 1``.  Although ``s14`` and ``1-s14`` give the same electron
    disappearance amplitude, they do not in general give the same
    ``|U_mu4|²=(1-s14)s24`` at fixed appearance amplitude.  Dropping the
    large-``s14`` branch would therefore not be a complete profile.
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
        coordinate, chi2, message = _profile_bounded_scalar(
            lambda value: objective(make_parameters(value)),
            bounds,
        )
        return FitPoint(make_parameters(coordinate), chi2), message

    if sin2_2theta_mue == 0.0:
        # The zero-amplitude boundary is the union s24=0 or s14 in {0,1}.
        # Searching s24=0 over the full s14 interval includes both endpoints;
        # the s14=0 branch is also searched over all s24.  At s14=1 the
        # short-baseline active probabilities are independent of s24.
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
    s14_upper = (1.0 + root) / 2.0

    def constrained_parameters(s14: float) -> ThreePlusOneParameters:
        denominator = 4.0 * s14 * (1.0 - s14)
        s24 = sin2_2theta_mue / denominator
        # Floating-point roundoff can exceed the endpoint by a few ulps.
        return ThreePlusOneParameters(delta_m2_41_eV2, s14, float(np.clip(s24, 0.0, 1.0)))

    coordinate, chi2, message = _profile_bounded_scalar(
        lambda value: objective(constrained_parameters(value)),
        (s14_lower, s14_upper),
    )
    best_fit = FitPoint(constrained_parameters(coordinate), chi2)
    return AppearanceAmplitudeProfileResult(delta_m2_41_eV2, sin2_2theta_mue, best_fit, message)


def profile_appearance_amplitude_grid(
    objective: Objective,
    delta_m2_values_eV2: Iterable[float],
    sin2_2theta_mue_values: Iterable[float],
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
        )
        for delta_m2, amplitude in product(delta_values, amplitude_values)
    ]


def profile_s14_s24_at_fixed_sin2_2theta_ee(
    objective: Objective,
    *,
    delta_m2_41_eV2: float,
    sin2_2theta_ee: float,
) -> ElectronDisappearanceProfileResult:
    """Profile sin²θ24 on every physical branch at fixed electron disappearance.

    ``sin²(2θee)=4*s14*(1-s14)`` generally has the two solutions
    ``s14=(1±sqrt(1-Aee))/2``.  They are not discarded or identified because
    their different ``|Umu4|²=(1-s14)*s24`` values can produce different event
    predictions.  Each branch is independently profiled over the complete
    unitary ``0 <= s24 <= 1`` interval and the lower chi-square is retained.
    """
    ThreePlusOneParameters(delta_m2_41_eV2, 0.0, 0.0)
    if not np.isfinite(sin2_2theta_ee) or not 0.0 <= sin2_2theta_ee <= 1.0:
        raise ValueError("sin2_2theta_ee must be in [0, 1]")
    root = np.sqrt(1.0 - sin2_2theta_ee)
    branches = ((1.0 - root) / 2.0, (1.0 + root) / 2.0)
    candidates: list[tuple[FitPoint, str]] = []
    for s14 in dict.fromkeys(float(value) for value in branches):
        make_parameters = lambda s24, s14=s14: ThreePlusOneParameters(
            delta_m2_41_eV2, s14, s24
        )
        coordinate, chi2, message = _profile_bounded_scalar(
            lambda s24: objective(make_parameters(s24)),
            PROFILE_MIXING_BOUNDS["sin2_theta24"],
        )
        candidates.append((FitPoint(make_parameters(coordinate), chi2), message))
    best_fit, message = min(candidates, key=lambda item: item[0].chi2)
    return ElectronDisappearanceProfileResult(
        delta_m2_41_eV2,
        sin2_2theta_ee,
        best_fit,
        f"all {len(candidates)} physical s14 branch(es); {message}",
    )


def profile_electron_disappearance_grid(
    objective: Objective,
    delta_m2_values_eV2: Iterable[float],
    sin2_2theta_ee_values: Iterable[float],
) -> list[ElectronDisappearanceProfileResult]:
    """Profile every point of the exact sin²(2θee)--Δm²41 scan plane."""
    delta_values = tuple(float(value) for value in delta_m2_values_eV2)
    amplitude_values = tuple(float(value) for value in sin2_2theta_ee_values)
    if not delta_values or not amplitude_values:
        raise ValueError("both electron-disappearance scan axes must be non-empty")
    return [
        profile_s14_s24_at_fixed_sin2_2theta_ee(
            objective,
            delta_m2_41_eV2=delta_m2,
            sin2_2theta_ee=amplitude,
        )
        for delta_m2, amplitude in product(delta_values, amplitude_values)
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
