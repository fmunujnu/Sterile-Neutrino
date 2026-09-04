"""core/one_plus_three_plus_one.py: regrouped existing implementations; see docs/ARCHITECTURE.md."""
from __future__ import annotations

from dataclasses import dataclass
from math import cos, isfinite, pi, sqrt
import numpy as np
from numpy.typing import NDArray
# Physical short-baseline parameters for the initial 1+3+1 implementation.


MIXING_PARAMETER_NAMES = (
    "abs_Ue4_squared",
    "abs_Umu4_squared",
    "abs_Ue5_squared",
    "abs_Umu5_squared",
)
PARAMETER_NAMES = (
    "delta_m2_41_absolute_eV2",
    "delta_m2_51_eV2",
    *MIXING_PARAMETER_NAMES,
    "cp_phase_mue_rad",
)


@dataclass(frozen=True, slots=True)
class OnePlusThreePlusOneParameters:
    """Seven physical coordinates used by the effective SBL CC model.

    State 4 is below the three approximately degenerate active states and
    state 5 is above them.  Therefore ``delta_m2_41_absolute_eV2`` stores the
    positive magnitude of a negative ``Delta m^2_41``; the signed value
    is exposed by :attr:`delta_m2_41_eV2`.

    The four mixing coordinates are matrix-element moduli, not rotation-angle
    names.  The phase convention is explicitly
    ``cp_phase_mue_rad = arg(U_mu4* U_e4 U_mu5 U_e5*)``.  It is the single
    phase measurable by mu-to-e short-baseline appearance in this restricted
    model; antineutrinos use its negative.  The row-normalisation
    and row-orthogonality inequalities below are necessary and sufficient for
    the two specified heavy-state row fragments to be embeddable in a unitary
    five-neutrino matrix.

    This is the standard two-sterile SBL parameterisation used, for example,
    by Karagiorgi et al. (hep-ph/0609177) and Kopp et al. (arXiv:1303.3011),
    specialised to the 1+3+1 mass ordering.
    """

    delta_m2_41_absolute_eV2: float
    delta_m2_51_eV2: float
    abs_Ue4_squared: float
    abs_Umu4_squared: float
    abs_Ue5_squared: float
    abs_Umu5_squared: float
    cp_phase_mue_rad: float

    def __post_init__(self) -> None:
        for name in PARAMETER_NAMES:
            value = float(getattr(self, name))
            if not isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.delta_m2_41_absolute_eV2 <= 0.0:
            raise ValueError("delta_m2_41_absolute_eV2 must be strictly positive")
        if self.delta_m2_51_eV2 <= 0.0:
            raise ValueError("delta_m2_51_eV2 must be strictly positive")
        for name in MIXING_PARAMETER_NAMES:
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if not -pi <= self.cp_phase_mue_rad <= pi:
            raise ValueError("cp_phase_mue_rad must be in [-pi, pi]")
        if self.electron_heavy_row_norm > 1.0 + 1e-12:
            raise ValueError("abs_Ue4_squared + abs_Ue5_squared must not exceed 1")
        if self.muon_heavy_row_norm > 1.0 + 1e-12:
            raise ValueError("abs_Umu4_squared + abs_Umu5_squared must not exceed 1")
        if self.unitary_embedding_margin < -1e-12:
            raise ValueError(
                "the selected e/mu heavy-state elements and CP phase cannot be embedded "
                "in orthogonal rows of a unitary five-neutrino matrix"
            )

    @property
    def delta_m2_41_eV2(self) -> float:
        """Signed lower-state mass splitting, always negative in 1+3+1."""
        return -float(self.delta_m2_41_absolute_eV2)

    @property
    def delta_m2_54_eV2(self) -> float:
        """Positive splitting between the two isolated states."""
        return float(self.delta_m2_51_eV2 + self.delta_m2_41_absolute_eV2)

    @property
    def electron_heavy_row_norm(self) -> float:
        return float(self.abs_Ue4_squared + self.abs_Ue5_squared)

    @property
    def muon_heavy_row_norm(self) -> float:
        return float(self.abs_Umu4_squared + self.abs_Umu5_squared)

    @property
    def heavy_row_overlap_squared(self) -> float:
        """Squared heavy-sector e/mu row overlap at the selected CP phase."""
        first = self.abs_Ue4_squared * self.abs_Umu4_squared
        second = self.abs_Ue5_squared * self.abs_Umu5_squared
        interference = 2.0 * sqrt(max(0.0, first * second)) * cos(self.cp_phase_mue_rad)
        return float(first + second + interference)

    @property
    def unitary_embedding_margin(self) -> float:
        light_capacity = (
            (1.0 - self.electron_heavy_row_norm)
            * (1.0 - self.muon_heavy_row_norm)
        )
        return float(light_capacity - self.heavy_row_overlap_squared)

    @property
    def sin2_2theta_mue_state4(self) -> float:
        return float(4.0 * self.abs_Ue4_squared * self.abs_Umu4_squared)

    @property
    def sin2_2theta_mue_state5(self) -> float:
        return float(4.0 * self.abs_Ue5_squared * self.abs_Umu5_squared)

    @classmethod
    def three_neutrino_null(cls) -> "OnePlusThreePlusOneParameters":
        """Return the no-sterile-mixing boundary; both masses are irrelevant."""
        return cls(1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0)


# Effective e/mu short-baseline probabilities for a 1+3+1 spectrum.


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

