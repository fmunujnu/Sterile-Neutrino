"""Unambiguous physical parameter definitions.

External code must use mass-squared splittings in eV^2 and sin^2(theta).
Angles in radians and sin(theta) are internal derived quantities only.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import asin, isfinite, sqrt


def _validate_unit_interval(name: str, value: float) -> None:
    if not isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {value!r}")


@dataclass(frozen=True, slots=True)
class ThreePlusOneParameters:
    """Three-parameter 3+1 oscillation point.

    `delta_m2_41_eV2` is Δm²₄₁ itself, never its square root.
    `sin2_theta14` and `sin2_theta24` mean sin²(θ14) and sin²(θ24).
    """

    delta_m2_41_eV2: float
    sin2_theta14: float
    sin2_theta24: float

    def __post_init__(self) -> None:
        if not isfinite(self.delta_m2_41_eV2) or self.delta_m2_41_eV2 <= 0.0:
            raise ValueError("delta_m2_41_eV2 must be strictly positive")
        _validate_unit_interval("sin2_theta14", self.sin2_theta14)
        _validate_unit_interval("sin2_theta24", self.sin2_theta24)

    @property
    def theta14_rad(self) -> float:
        return asin(sqrt(self.sin2_theta14))

    @property
    def theta24_rad(self) -> float:
        return asin(sqrt(self.sin2_theta24))

    @property
    def sin_theta14(self) -> float:
        return sqrt(self.sin2_theta14)

    @property
    def sin_theta24(self) -> float:
        return sqrt(self.sin2_theta24)

    @property
    def sin2_2theta_mue_exact(self) -> float:
        """Exact 4|Ue4|²|Umu4|² for this rotation convention."""
        ue4_sq = self.sin2_theta14
        umu4_sq = (1.0 - self.sin2_theta14) * self.sin2_theta24
        return 4.0 * ue4_sq * umu4_sq
