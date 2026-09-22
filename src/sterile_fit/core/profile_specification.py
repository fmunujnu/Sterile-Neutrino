"""Explicit declarations of fixed and profiled model coordinates.

This module contains no optimiser and no physics.  It only prevents a caller
from accidentally relying on the old convention that every omitted parameter
is silently profiled.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Iterable, Mapping

import numpy as np


@dataclass(frozen=True, slots=True)
class ProfileSpecification:
    """Partition every model coordinate into fixed values or profile bounds."""

    fixed_values: Mapping[str, float]
    profiled_bounds: Mapping[str, tuple[float, float]]

    @classmethod
    def create(
        cls,
        parameter_names: Iterable[str],
        *,
        fixed_values: Mapping[str, float],
        profiled_bounds: Mapping[str, tuple[float, float]],
    ) -> "ProfileSpecification":
        names = tuple(parameter_names)
        known = set(names)
        fixed = {name: float(value) for name, value in fixed_values.items()}
        profiled = {
            name: (float(bounds[0]), float(bounds[1]))
            for name, bounds in profiled_bounds.items()
        }
        unknown = (set(fixed) | set(profiled)) - known
        overlap = set(fixed) & set(profiled)
        missing = known - set(fixed) - set(profiled)
        if unknown:
            raise ValueError(f"unknown profile parameter names: {sorted(unknown)}")
        if overlap:
            raise ValueError(f"parameters cannot be both fixed and profiled: {sorted(overlap)}")
        if missing:
            raise ValueError(f"every model parameter must be declared; missing: {sorted(missing)}")
        if any(not np.isfinite(value) for value in fixed.values()):
            raise ValueError("fixed profile values must be finite")
        for name, (lower, upper) in profiled.items():
            if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
                raise ValueError(f"profile bounds for {name} must be finite and increasing")
        return cls(MappingProxyType(fixed), MappingProxyType(profiled))
