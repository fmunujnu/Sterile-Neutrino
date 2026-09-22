"""core/profile_one_plus_three_plus_one.py: regrouped existing implementations; see docs/ARCHITECTURE.md."""
from __future__ import annotations

from dataclasses import dataclass
from math import pi
from typing import Callable, Mapping
import numpy as np
from scipy.optimize import NonlinearConstraint, differential_evolution
from sterile_fit.core.one_plus_three_plus_one import MIXING_PARAMETER_NAMES, PARAMETER_NAMES, OnePlusThreePlusOneParameters
from sterile_fit.core.profile_specification import ProfileSpecification
# Constrained profile fits for the parallel 1+3+1 parameterisation.


Objective = Callable[[OnePlusThreePlusOneParameters], float]
MASS_PARAMETER_NAMES = (
    "delta_m2_41_absolute_eV2",
    "delta_m2_51_eV2",
)


@dataclass(frozen=True, slots=True)
class OnePlusThreePlusOneFitPoint:
    parameters: OnePlusThreePlusOneParameters
    chi2: float


@dataclass(frozen=True, slots=True)
class OnePlusThreePlusOneProfileResult:
    fixed_parameters: Mapping[str, float]
    best_fit: OnePlusThreePlusOneFitPoint
    optimizer_message: str


def _default_complete_point(fixed: Mapping[str, float]) -> dict[str, float]:
    values = {
        "delta_m2_41_absolute_eV2": 1.0,
        "delta_m2_51_eV2": 1.0,
        "abs_Ue4_squared": 0.0,
        "abs_Umu4_squared": 0.0,
        "abs_Ue5_squared": 0.0,
        "abs_Umu5_squared": 0.0,
        "cp_phase_mue_rad": 0.0,
    }
    values.update({name: float(value) for name, value in fixed.items()})
    return values


def _validate_fixed_parameters(fixed_parameters: Mapping[str, float]) -> None:
    unknown = set(fixed_parameters).difference(PARAMETER_NAMES)
    if unknown:
        raise ValueError(f"unknown 1+3+1 profile parameter names: {sorted(unknown)}")
    OnePlusThreePlusOneParameters(**_default_complete_point(fixed_parameters))


def profile_one_plus_three_plus_one(
    objective: Objective,
    fixed_parameters: Mapping[str, float] | ProfileSpecification,
    *,
    mass_magnitude_bounds_eV2: tuple[float, float] = (1e-2, 1e2),
    seed: int = 42,
    maxiter: int = 400,
    popsize: int = 15,
    tolerance: float = 1e-8,
    polish: bool = True,
) -> OnePlusThreePlusOneProfileResult:
    """Minimise over every non-fixed coordinate with exact physical constraints.

    Both mass magnitudes are searched in log10 space.  The four matrix-element
    moduli and CP phase are profiled directly.  The optimiser is constrained by
    both row normalisations and the unitary-row embeddability inequality; an
    invalid parameter point is never passed to the physics objective.
    """
    specification = fixed_parameters if isinstance(fixed_parameters, ProfileSpecification) else None
    declared_fixed = specification.fixed_values if specification is not None else fixed_parameters
    _validate_fixed_parameters(declared_fixed)
    lower_mass, upper_mass = (float(value) for value in mass_magnitude_bounds_eV2)
    if lower_mass <= 0.0 or upper_mass <= lower_mass:
        raise ValueError("mass_magnitude_bounds_eV2 must be positive and increasing")
    if maxiter < 1 or popsize < 1 or tolerance <= 0.0:
        raise ValueError("optimizer controls must be strictly positive")

    fixed = {name: float(value) for name, value in declared_fixed.items()}
    if specification is None:
        free_names = tuple(name for name in PARAMETER_NAMES if name not in fixed)
        declared_bounds = {
            name: (
                mass_magnitude_bounds_eV2 if name in MASS_PARAMETER_NAMES
                else (0.0, 1.0) if name in MIXING_PARAMETER_NAMES
                else (-pi, pi)
            )
            for name in free_names
        }
    else:
        specification = ProfileSpecification.create(
            PARAMETER_NAMES,
            fixed_values=specification.fixed_values,
            profiled_bounds=specification.profiled_bounds,
        )
        free_names = tuple(name for name in PARAMETER_NAMES if name in specification.profiled_bounds)
        declared_bounds = dict(specification.profiled_bounds)

    def raw_values(vector: np.ndarray) -> dict[str, float]:
        values = dict(fixed)
        for name, coordinate in zip(free_names, vector, strict=True):
            value = float(coordinate)
            values[name] = 10.0**value if name in MASS_PARAMETER_NAMES else value
        return values

    def unpack(vector: np.ndarray) -> OnePlusThreePlusOneParameters:
        return OnePlusThreePlusOneParameters(**raw_values(vector))

    if not free_names:
        point = OnePlusThreePlusOneParameters(**fixed)
        chi2 = float(objective(point))
        if not np.isfinite(chi2):
            raise RuntimeError("profile objective returned a non-finite value")
        return OnePlusThreePlusOneProfileResult(
            fixed, OnePlusThreePlusOneFitPoint(point, chi2), "no free parameters"
        )

    bounds: list[tuple[float, float]] = []
    for name in free_names:
        bound_lower, bound_upper = declared_bounds[name]
        if name in MASS_PARAMETER_NAMES:
            if bound_lower <= 0.0 or bound_upper <= bound_lower:
                raise ValueError(f"profiled {name} bounds must be positive and increasing")
            bounds.append((float(np.log10(bound_lower)), float(np.log10(bound_upper))))
        elif name in MIXING_PARAMETER_NAMES:
            if bound_lower < 0.0 or bound_upper > 1.0:
                raise ValueError(f"profiled {name} bounds must stay inside [0, 1]")
            bounds.append((bound_lower, bound_upper))
        else:
            if bound_lower < -pi or bound_upper > pi:
                raise ValueError("profiled cp_phase_mue_rad bounds must stay inside [-pi, pi]")
            bounds.append((bound_lower, bound_upper))

    def feasibility(vector: np.ndarray) -> np.ndarray:
        values = raw_values(np.asarray(vector, dtype=float))
        xe4 = values["abs_Ue4_squared"]
        xmu4 = values["abs_Umu4_squared"]
        xe5 = values["abs_Ue5_squared"]
        xmu5 = values["abs_Umu5_squared"]
        phase = values["cp_phase_mue_rad"]
        overlap = (
            xe4 * xmu4
            + xe5 * xmu5
            + 2.0 * np.sqrt(max(0.0, xe4 * xmu4 * xe5 * xmu5)) * np.cos(phase)
        )
        embedding_margin = (1.0 - xe4 - xe5) * (1.0 - xmu4 - xmu5) - overlap
        return np.array(
            [1.0 - xe4 - xe5, 1.0 - xmu4 - xmu5, embedding_margin],
            dtype=float,
        )

    constraint = NonlinearConstraint(feasibility, 0.0, np.inf)

    def evaluate(vector: np.ndarray) -> float:
        try:
            point = unpack(np.asarray(vector, dtype=float))
        except ValueError:
            return 1e100
        value = float(objective(point))
        return value if np.isfinite(value) else 1e100

    result = differential_evolution(
        evaluate,
        bounds=bounds,
        constraints=(constraint,),
        seed=seed,
        maxiter=maxiter,
        popsize=popsize,
        tol=tolerance,
        polish=polish,
        workers=1,
        updating="immediate",
    )
    if not result.success or not np.isfinite(result.fun) or result.fun >= 1e99:
        raise RuntimeError(f"1+3+1 profile optimization failed: {result.message}")
    best = unpack(np.asarray(result.x, dtype=float))
    return OnePlusThreePlusOneProfileResult(
        fixed,
        OnePlusThreePlusOneFitPoint(best, float(result.fun)),
        str(result.message),
    )


def profile_at_fixed_mass_pair(
    objective: Objective,
    *,
    delta_m2_41_absolute_eV2: float,
    delta_m2_51_eV2: float,
    seed: int = 42,
    maxiter: int = 400,
    popsize: int = 15,
    tolerance: float = 1e-8,
    polish: bool = True,
) -> OnePlusThreePlusOneProfileResult:
    """Profile all measurable e/mu mixings at one 1+3+1 mass pair.

    Exact lower-dimensional surfaces are evaluated separately because a
    continuous global optimiser need not land exactly on a decoupled-state or
    no-sterile-mixing boundary.
    """
    masses = {
        "delta_m2_41_absolute_eV2": float(delta_m2_41_absolute_eV2),
        "delta_m2_51_eV2": float(delta_m2_51_eV2),
    }
    common = {
        "mass_magnitude_bounds_eV2": (
            min(delta_m2_41_absolute_eV2, delta_m2_51_eV2) / 10.0,
            max(delta_m2_41_absolute_eV2, delta_m2_51_eV2) * 10.0,
        ),
        "maxiter": maxiter,
        "popsize": popsize,
        "tolerance": tolerance,
        "polish": polish,
    }
    full_profile = ProfileSpecification.create(
        PARAMETER_NAMES,
        fixed_values=masses,
        profiled_bounds={
            "abs_Ue4_squared": (0.0, 1.0),
            "abs_Umu4_squared": (0.0, 1.0),
            "abs_Ue5_squared": (0.0, 1.0),
            "abs_Umu5_squared": (0.0, 1.0),
            "cp_phase_mue_rad": (-pi, pi),
        },
    )
    state4_decoupled = ProfileSpecification.create(
        PARAMETER_NAMES,
        fixed_values={
            **masses,
            "abs_Ue4_squared": 0.0,
            "abs_Umu4_squared": 0.0,
            "cp_phase_mue_rad": 0.0,
        },
        profiled_bounds={
            "abs_Ue5_squared": (0.0, 1.0),
            "abs_Umu5_squared": (0.0, 1.0),
        },
    )
    state5_decoupled = ProfileSpecification.create(
        PARAMETER_NAMES,
        fixed_values={
            **masses,
            "abs_Ue5_squared": 0.0,
            "abs_Umu5_squared": 0.0,
            "cp_phase_mue_rad": 0.0,
        },
        profiled_bounds={
            "abs_Ue4_squared": (0.0, 1.0),
            "abs_Umu4_squared": (0.0, 1.0),
        },
    )
    candidates = [
        profile_one_plus_three_plus_one(objective, full_profile, seed=seed, **common),
        profile_one_plus_three_plus_one(
            objective,
            state4_decoupled,
            seed=seed + 10_000,
            **common,
        ),
        profile_one_plus_three_plus_one(
            objective,
            state5_decoupled,
            seed=seed + 20_000,
            **common,
        ),
    ]
    null = OnePlusThreePlusOneParameters(
        masses["delta_m2_41_absolute_eV2"],
        masses["delta_m2_51_eV2"],
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    )
    candidates.append(
        OnePlusThreePlusOneProfileResult(
            masses,
            OnePlusThreePlusOneFitPoint(null, float(objective(null))),
            "explicit no-sterile-mixing boundary",
        )
    )
    best = min(candidates, key=lambda item: item.best_fit.chi2)
    return OnePlusThreePlusOneProfileResult(
        masses,
        best.best_fit,
        f"best of full volume, state-4 decoupled, state-5 decoupled and null; {best.optimizer_message}",
    )


# Standalone global prefit archived; constrained profile is retained.

