"""Auditable LSND DAR rate-only likelihood from final-paper public inputs.

This is intentionally less informative than the collaboration's four-variable
event likelihood.  It preserves the physics interface needed by a future
1+3+1 model: an arbitrary appearance probability is averaged with a declared
DAR flux/cross-section/baseline kernel, rather than inferred from a published
3+1 contour.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from sterile_fit.experiments.lsnd.final_2001 import (
    three_plus_one_dar_appearance_probability,
)


ProbabilityFunction = Callable[
    [NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]
]


@dataclass(frozen=True, slots=True)
class LSNDDARRateInputs:
    """Published measurement plus explicit public-kernel approximations."""

    observed_average_probability: float = 0.00264
    statistical_sigma: float = 0.00067
    systematic_sigma: float = 0.00045
    detector_center_baseline_m: float = 30.0
    detector_axial_length_m: float = 8.3
    positron_energy_threshold_MeV: float = 20.0
    muon_dar_endpoint_MeV: float = 52.8

    @property
    def total_sigma(self) -> float:
        return float(np.hypot(self.statistical_sigma, self.systematic_sigma))


def _public_dar_kernel(inputs: LSNDDARRateInputs) -> tuple[
    NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]
]:
    """Return E, L and normalized weights for a documented public approximation.

    The anti-muon-neutrino Michel spectrum is multiplied by leading IBD phase
    space.  Selection efficiency is treated as constant because LSND publishes
    only its average for this sample.  The 8.3 m detector axis is integrated
    uniformly with the point-source 1/L^2 flux factor.  Source extent,
    transverse geometry and reconstructed-energy migration are unavailable.
    """
    neutron_proton_mass_difference_MeV = 1.293332
    electron_mass_MeV = 0.510999
    minimum_neutrino_energy_MeV = (
        inputs.positron_energy_threshold_MeV
        + neutron_proton_mass_difference_MeV
    )
    energy = np.linspace(
        minimum_neutrino_energy_MeV, inputs.muon_dar_endpoint_MeV, 512
    )
    half_length = inputs.detector_axial_length_m / 2.0
    baseline = np.linspace(
        inputs.detector_center_baseline_m - half_length,
        inputs.detector_center_baseline_m + half_length,
        129,
    )
    x = energy / inputs.muon_dar_endpoint_MeV
    michel_antinu_mu = 2.0 * x**2 * (3.0 - 2.0 * x)
    positron_total_energy = energy - neutron_proton_mass_difference_MeV
    positron_momentum = np.sqrt(
        np.maximum(positron_total_energy**2 - electron_mass_MeV**2, 0.0)
    )
    ibd_phase_space = positron_total_energy * positron_momentum
    weight = (
        michel_antinu_mu[:, None]
        * ibd_phase_space[:, None]
        / baseline[None, :] ** 2
    )
    # The regular grids have constant spacing, so normalization by the sum is
    # the corresponding rectangular quadrature ratio; absolute factors cancel.
    weight /= weight.sum()
    return energy, baseline, weight


def average_appearance_probability(
    probability: ProbabilityFunction,
    inputs: LSNDDARRateInputs = LSNDDARRateInputs(),
) -> float:
    """Average any model probability over the same LSND public DAR kernel."""
    energy, baseline, weight = _public_dar_kernel(inputs)
    energy_grid, baseline_grid = np.meshgrid(energy, baseline, indexing="ij")
    values = np.asarray(
        probability(energy_grid.ravel(), baseline_grid.ravel()), dtype=float
    ).reshape(weight.shape)
    if values.shape != weight.shape or not np.all(np.isfinite(values)):
        raise ValueError("appearance probability returned an invalid array")
    if np.any(values < -1e-12) or np.any(values > 1.0 + 1e-12):
        raise ValueError("appearance probability must remain in [0, 1]")
    return float(np.sum(weight * values))


def three_plus_one_average_probability(
    delta_m2_41_eV2: float,
    sin2_2theta_mue: float,
    inputs: LSNDDARRateInputs = LSNDDARRateInputs(),
) -> float:
    """Average the unchanged common 3+1 core over the LSND approximation."""
    return average_appearance_probability(
        lambda energy, baseline: three_plus_one_dar_appearance_probability(
            energy, baseline, delta_m2_41_eV2, sin2_2theta_mue
        ),
        inputs,
    )


def rate_negative_two_log_likelihood(
    predicted_average_probability: float,
    inputs: LSNDDARRateInputs = LSNDDARRateInputs(),
) -> float:
    """Gaussian -2 log L for LSND's published average DAR probability."""
    pull = (
        predicted_average_probability - inputs.observed_average_probability
    ) / inputs.total_sigma
    return float(pull * pull)

