"""core/three_plus_one.py: regrouped existing implementations; see docs/ARCHITECTURE.md."""
from __future__ import annotations

from dataclasses import dataclass
from math import asin, isfinite, sqrt
from typing import Protocol
import numpy as np
from numpy.typing import NDArray
# Unambiguous physical parameter definitions.


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

    @property
    def sin2_2theta_ee_exact(self) -> float:
        """Exact electron-flavour disappearance amplitude 4|Ue4|²(1-|Ue4|²)."""
        ue4_sq = self.sin2_theta14
        return 4.0 * ue4_sq * (1.0 - ue4_sq)


# Interfaces shared by present and future oscillation models.


class VacuumOscillationModel(Protocol):
    """A model that returns P(initial flavour -> final flavour)."""

    def probability(
        self,
        initial_flavour: int,
        final_flavour: int,
        energy_GeV: NDArray[np.float64],
        baseline_km: float,
        *,
        antineutrino: bool = False,
    ) -> NDArray[np.float64]: ...


# Vacuum 3+1 probabilities with explicit units and parameter meanings.


# The amplitude exponent is -i * 2 * 1.267 * m²[eV²] L[km] / E[GeV].
# Its interference term therefore has the conventional sin²(1.267 Δm² L / E).
OSCILLATION_AMPLITUDE_COEFFICIENT = 2.0 * 1.267


def _rotation(size: int, first: int, second: int, angle_rad: float) -> NDArray[np.complex128]:
    matrix = np.eye(size, dtype=complex)
    sine, cosine = np.sin(angle_rad), np.cos(angle_rad)
    matrix[first, first] = cosine
    matrix[second, second] = cosine
    matrix[first, second] = sine
    matrix[second, first] = -sine
    return matrix


def _standard_pmns_embedded(size: int) -> NDArray[np.complex128]:
    theta12 = np.deg2rad(33.0)
    theta13 = np.deg2rad(8.6)
    theta23 = np.deg2rad(49.0)
    return _rotation(size, 1, 2, theta23) @ _rotation(size, 0, 2, theta13) @ _rotation(size, 0, 1, theta12)


class ThreePlusOneVacuumModel:
    """3+1 model with e=0, mu=1, tau=2 and sterile=3 flavour indices."""

    def __init__(self, parameters: ThreePlusOneParameters) -> None:
        self.parameters = parameters
        size = 4
        sterile_rotations = (
            _rotation(size, 1, 3, parameters.theta24_rad)
            @ _rotation(size, 0, 3, parameters.theta14_rad)
        )
        self.mixing_matrix = sterile_rotations @ _standard_pmns_embedded(size)
        # The published short-baseline equations neglect solar and atmospheric
        # phases, so nu1, nu2 and nu3 are exactly degenerate here.  Retaining
        # their splittings would introduce an effect absent from the target
        # analysis, especially in the lowest true-energy bins.
        self.mass_squared_eV2 = np.array([0.0, 0.0, 0.0, parameters.delta_m2_41_eV2], dtype=float)

    def probability(
        self,
        initial_flavour: int,
        final_flavour: int,
        energy_GeV: NDArray[np.float64],
        baseline_km: float,
        *,
        antineutrino: bool = False,
    ) -> NDArray[np.float64]:
        energies = np.asarray(energy_GeV, dtype=float)
        if energies.ndim != 1 or energies.size == 0 or not np.all(np.isfinite(energies)) or np.any(energies <= 0.0):
            raise ValueError("energy_GeV must be a non-empty finite, strictly positive vector")
        if not np.isfinite(baseline_km) or baseline_km <= 0.0:
            raise ValueError("baseline_km must be strictly positive")
        if not 0 <= initial_flavour < 4 or not 0 <= final_flavour < 4:
            raise ValueError("flavour indices must be in [0, 3]")

        phase = np.exp(
            -1j * OSCILLATION_AMPLITUDE_COEFFICIENT * self.mass_squared_eV2[None, :] * baseline_km / energies[:, None]
        )
        if antineutrino:
            coefficients = np.conj(self.mixing_matrix[final_flavour, :]) * self.mixing_matrix[initial_flavour, :]
        else:
            coefficients = self.mixing_matrix[final_flavour, :] * np.conj(self.mixing_matrix[initial_flavour, :])
        return np.abs(phase @ coefficients) ** 2


def short_baseline_appearance_probability(
    parameters: ThreePlusOneParameters,
    energy_GeV: NDArray[np.float64],
    baseline_km: NDArray[np.float64] | float,
) -> NDArray[np.float64]:
    """Exact 3+1 SBL P(mu->e), allowing an event-by-event baseline.

    With states 1--3 degenerate there is one oscillation frequency and no CP
    asymmetry, so the same expression applies to neutrinos and antineutrinos.
    """
    energies = np.asarray(energy_GeV, dtype=float)
    baselines = np.asarray(baseline_km, dtype=float)
    if energies.ndim != 1 or energies.size == 0 or np.any(energies <= 0.0):
        raise ValueError("energy_GeV must be a non-empty positive vector")
    if baselines.ndim not in (0, 1) or np.any(baselines <= 0.0):
        raise ValueError("baseline_km must be a positive scalar or vector")
    if baselines.ndim == 1 and baselines.shape != energies.shape:
        raise ValueError("an event-by-event baseline must match energy_GeV")
    phase = 1.267 * parameters.delta_m2_41_eV2 * baselines / energies
    return parameters.sin2_2theta_mue_exact * np.sin(phase) ** 2

