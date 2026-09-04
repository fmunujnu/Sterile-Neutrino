"""Effective e/mu short-baseline probabilities for a 1+3+1 spectrum."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .parameters import OnePlusThreePlusOneParameters


OSCILLATION_AMPLITUDE_COEFFICIENT = 2.0 * 1.267


class OnePlusThreePlusOneVacuumModel:
    """Two-isolated-state vacuum model used by the existing CC event kernels.

    The detector adapters currently request only e and mu initial/final
    flavours.  This class therefore evaluates those amplitudes exactly in the
    short-baseline limit without inventing unmeasured tau-sector angles.
    ``theta34`` and ``theta35`` are effectively zero in this first-stage CC
    analysis; a separate tau/NC extension is required before those parameters
    can be interpreted or profiled.
    """

    def __init__(self, parameters: OnePlusThreePlusOneParameters) -> None:
        self.parameters = parameters

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
        if (
            energies.ndim != 1
            or energies.size == 0
            or not np.all(np.isfinite(energies))
            or np.any(energies <= 0.0)
        ):
            raise ValueError("energy_GeV must be a non-empty finite, strictly positive vector")
        if not np.isfinite(baseline_km) or baseline_km <= 0.0:
            raise ValueError("baseline_km must be strictly positive")
        if initial_flavour not in (0, 1) or final_flavour not in (0, 1):
            raise ValueError("the initial 1+3+1 CC model supports only e=0 and mu=1")

        parameters = self.parameters
        phases = np.exp(
            -1j
            * OSCILLATION_AMPLITUDE_COEFFICIENT
            * np.array(
                [parameters.delta_m2_41_eV2, parameters.delta_m2_51_eV2],
                dtype=float,
            )[None, :]
            * float(baseline_km)
            / energies[:, None]
        )
        phase_differences = phases - 1.0

        if initial_flavour == final_flavour:
            if initial_flavour == 0:
                heavy_weights = np.array(
                    [parameters.abs_Ue4_squared, parameters.abs_Ue5_squared],
                    dtype=float,
                )
            else:
                heavy_weights = np.array(
                    [parameters.abs_Umu4_squared, parameters.abs_Umu5_squared],
                    dtype=float,
                )
            amplitude = 1.0 + phase_differences @ heavy_weights
        else:
            coefficient4 = np.sqrt(
                parameters.abs_Ue4_squared * parameters.abs_Umu4_squared
            )
            phase_sign = 1.0 if (initial_flavour, final_flavour) == (0, 1) else -1.0
            if antineutrino:
                phase_sign *= -1.0
            coefficient5 = np.sqrt(
                parameters.abs_Ue5_squared * parameters.abs_Umu5_squared
            ) * np.exp(1j * phase_sign * parameters.cp_phase_mue_rad)
            amplitude = (
                coefficient4 * phase_differences[:, 0]
                + coefficient5 * phase_differences[:, 1]
            )

        probability = np.abs(amplitude) ** 2
        tolerance = 5e-12
        if np.any(probability < -tolerance) or np.any(probability > 1.0 + tolerance):
            raise FloatingPointError("1+3+1 probability left the physical [0, 1] interval")
        return np.clip(probability, 0.0, 1.0)
