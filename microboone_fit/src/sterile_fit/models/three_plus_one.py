"""Vacuum 3+1 probabilities with explicit units and parameter meanings."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..parameters import ThreePlusOneParameters

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
        self.mass_squared_eV2 = np.array(
            [0.0, 7.5e-5, 2.5e-3, parameters.delta_m2_41_eV2], dtype=float
        )

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
