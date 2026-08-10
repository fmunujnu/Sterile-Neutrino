"""BNB four-channel oscillation templates.

This module is deliberately BNB-specific. It does not try to be a universal
experiment interface before the physical content of each experiment is known.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .models.base import VacuumOscillationModel


REQUIRED_TEMPLATE_KEYS = (
    "true_energy_GeV",
    "fixed_non_oscillatory_background_counts",
    "beam_nue_to_nue_cc_response_counts",
    "beam_numu_to_nue_cc_response_counts",
    "beam_nue_to_numu_cc_response_counts",
    "beam_numu_to_numu_cc_response_counts",
    "beam_nuebar_to_nuebar_cc_response_counts",
    "beam_numubar_to_nuebar_cc_response_counts",
    "beam_nuebar_to_numubar_cc_response_counts",
    "beam_numubar_to_numubar_cc_response_counts",
)


@dataclass(frozen=True, slots=True)
class BnbFourChannelOscillationTemplates:
    """Detector-folded, true-energy templates for the active BNB likelihood.

    `fixed_non_oscillatory_background_counts` contains only components that
    cannot be reweighted by an active-neutrino oscillation probability (for
    example, a separately documented cosmic or detector background). It must
    never be filled blindly from the public table headed "Background".

    Each beam response array has shape `(104, n_true_energy_bins)`. Its entry
    `[reconstructed_bin, true_energy_bin]` is the selected count before the
    oscillation probability for one specified source-to-final-flavour process.
    It already contains BNB flux, interaction model, detector efficiency,
    selection and reconstructed-energy migration. Therefore these ingredients
    must not be separately multiplied later.
    """

    true_energy_GeV: NDArray[np.float64]
    fixed_non_oscillatory_background_counts: NDArray[np.float64]
    beam_nue_to_nue_cc_response_counts: NDArray[np.float64]
    beam_numu_to_nue_cc_response_counts: NDArray[np.float64]
    beam_nue_to_numu_cc_response_counts: NDArray[np.float64]
    beam_numu_to_numu_cc_response_counts: NDArray[np.float64]
    beam_nuebar_to_nuebar_cc_response_counts: NDArray[np.float64]
    beam_numubar_to_nuebar_cc_response_counts: NDArray[np.float64]
    beam_nuebar_to_numubar_cc_response_counts: NDArray[np.float64]
    beam_numubar_to_numubar_cc_response_counts: NDArray[np.float64]

    def __post_init__(self) -> None:
        energy = np.asarray(self.true_energy_GeV, dtype=float)
        if energy.ndim != 1 or energy.size == 0 or np.any(energy <= 0.0):
            raise ValueError("true_energy_GeV must be a non-empty positive one-dimensional array")
        if not np.all(np.diff(energy) > 0.0):
            raise ValueError("true_energy_GeV must be strictly increasing")
        fixed_background = np.asarray(self.fixed_non_oscillatory_background_counts, dtype=float)
        if fixed_background.shape != (104,):
            raise ValueError("fixed_non_oscillatory_background_counts must have shape (104,)")
        if not np.all(np.isfinite(fixed_background)) or np.any(fixed_background < 0.0):
            raise ValueError("fixed_non_oscillatory_background_counts must be finite and non-negative")
        for name in REQUIRED_TEMPLATE_KEYS[2:]:
            response = np.asarray(getattr(self, name), dtype=float)
            if response.shape != (104, energy.size):
                raise ValueError(f"{name} must have shape (104, {energy.size})")
            if not np.all(np.isfinite(response)) or np.any(response < 0.0):
                raise ValueError(f"{name} must be finite and non-negative")

    @classmethod
    def from_npz(cls, path: Path) -> "BnbFourChannelOscillationTemplates":
        """Load only the documented `.npz` schema; reject hidden conventions."""
        with np.load(path, allow_pickle=False) as archive:
            missing = set(REQUIRED_TEMPLATE_KEYS).difference(archive.files)
            if missing:
                raise ValueError(f"template archive is missing required arrays: {sorted(missing)}")
            values = {key: np.asarray(archive[key], dtype=float) for key in REQUIRED_TEMPLATE_KEYS}
        return cls(**values)

    def predict_total_counts(self, model: VacuumOscillationModel, baseline_km: float) -> NDArray[np.float64]:
        """Return BNB prediction after applying all oscillation probabilities."""
        energy = self.true_energy_GeV
        probability = model.probability
        predicted = np.asarray(self.fixed_non_oscillatory_background_counts, dtype=float).copy()
        predicted += self.beam_nue_to_nue_cc_response_counts @ probability(0, 0, energy, baseline_km)
        predicted += self.beam_numu_to_nue_cc_response_counts @ probability(1, 0, energy, baseline_km)
        predicted += self.beam_nue_to_numu_cc_response_counts @ probability(0, 1, energy, baseline_km)
        predicted += self.beam_numu_to_numu_cc_response_counts @ probability(1, 1, energy, baseline_km)
        predicted += self.beam_nuebar_to_nuebar_cc_response_counts @ probability(0, 0, energy, baseline_km, antineutrino=True)
        predicted += self.beam_numubar_to_nuebar_cc_response_counts @ probability(1, 0, energy, baseline_km, antineutrino=True)
        predicted += self.beam_nuebar_to_numubar_cc_response_counts @ probability(0, 1, energy, baseline_km, antineutrino=True)
        predicted += self.beam_numubar_to_numubar_cc_response_counts @ probability(1, 1, energy, baseline_km, antineutrino=True)
        if not np.all(np.isfinite(predicted)) or np.any(predicted < 0.0):
            raise FloatingPointError("template prediction contains non-finite or negative counts")
        return predicted
