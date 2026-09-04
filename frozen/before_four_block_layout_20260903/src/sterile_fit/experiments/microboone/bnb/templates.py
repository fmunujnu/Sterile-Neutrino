"""BNB four-channel oscillation templates.

This module is deliberately BNB-specific. It does not try to be a universal
experiment interface before the physical content of each experiment is known.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from ....models.base import VacuumOscillationModel


RESPONSE_TEMPLATE_FIELDS = (
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

    `fixed_published_background_counts` is the released HEPData ``Background``
    category.  This analysis deliberately holds that whole category fixed; the
    name does not claim that every underlying physical component is intrinsically
    non-oscillatory.

    Each beam response array has shape `(104, n_true_energy_bins)`. Its entry
    `[reconstructed_bin, true_energy_bin]` is the selected count before the
    oscillation probability for one specified source-to-final-flavour process.
    It already contains BNB flux, interaction model, detector efficiency,
    selection and reconstructed-energy migration. Therefore these ingredients
    must not be separately multiplied later.
    """

    true_energy_GeV: NDArray[np.float64]
    fixed_published_background_counts: NDArray[np.float64]
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
        fixed_background = np.asarray(self.fixed_published_background_counts, dtype=float)
        if fixed_background.shape != (104,):
            raise ValueError("fixed_published_background_counts must have shape (104,)")
        if not np.all(np.isfinite(fixed_background)) or np.any(fixed_background < 0.0):
            raise ValueError("fixed_published_background_counts must be finite and non-negative")
        for name in RESPONSE_TEMPLATE_FIELDS:
            response = np.asarray(getattr(self, name), dtype=float)
            if response.shape != (104, energy.size):
                raise ValueError(f"{name} must have shape (104, {energy.size})")
            if not np.all(np.isfinite(response)) or np.any(response < 0.0):
                raise ValueError(f"{name} must be finite and non-negative")

    @classmethod
    def from_directory(cls, directory: Path) -> "BnbFourChannelOscillationTemplates":
        """Load a fully visible CSV/JSON template directory."""
        directory = Path(directory)
        required_paths = {
            "metadata.json": directory / "metadata.json",
            "true_energy_GeV.csv": directory / "true_energy_GeV.csv",
            "fixed_published_background_counts": directory / "fixed_published_background_counts.csv",
            **{name: directory / f"{name}.csv" for name in RESPONSE_TEMPLATE_FIELDS},
        }
        missing = [name for name, path in required_paths.items() if not path.is_file()]
        if missing:
            raise ValueError(f"template directory is missing visible files: {missing}")
        metadata = json.loads(required_paths["metadata.json"].read_text(encoding="utf-8"))
        if metadata.get("format") != "bnb_four_channel_text_templates_v1":
            raise ValueError("template metadata has an unknown format")
        energy_table = pd.read_csv(required_paths["true_energy_GeV.csv"])
        if list(energy_table.columns) != ["true_bin", "true_energy_GeV"]:
            raise ValueError("true_energy_GeV.csv must contain true_bin,true_energy_GeV")
        expected_bins = np.arange(len(energy_table), dtype=int)
        if not np.array_equal(energy_table["true_bin"].to_numpy(dtype=int), expected_bins):
            raise ValueError("true_energy_GeV.csv true_bin must be contiguous and zero-based")
        fixed_table = pd.read_csv(required_paths["fixed_published_background_counts"])
        if list(fixed_table.columns) != ["global_reco_bin", "fixed_published_background_counts"]:
            raise ValueError("fixed background CSV has unexpected columns")
        if not np.array_equal(fixed_table["global_reco_bin"].to_numpy(dtype=int), np.arange(104)):
            raise ValueError("fixed background global_reco_bin must be 0..103")
        values: dict[str, NDArray[np.float64]] = {
            "true_energy_GeV": energy_table["true_energy_GeV"].to_numpy(dtype=float),
            "fixed_published_background_counts": fixed_table[
                "fixed_published_background_counts"
            ].to_numpy(dtype=float),
        }
        for name in RESPONSE_TEMPLATE_FIELDS:
            table = pd.read_csv(required_paths[name])
            expected_columns = ["global_reco_bin", *[f"true_bin_{index:03d}" for index in expected_bins]]
            if list(table.columns) != expected_columns:
                raise ValueError(f"{name}.csv has unexpected columns")
            if not np.array_equal(table["global_reco_bin"].to_numpy(dtype=int), np.arange(104)):
                raise ValueError(f"{name}.csv global_reco_bin must be 0..103")
            values[name] = table.iloc[:, 1:].to_numpy(dtype=float)
        return cls(**values)

    def to_directory(self, directory: Path, *, metadata: dict[str, object]) -> None:
        """Write every physical input as inspectable CSV plus JSON metadata."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        energy = np.asarray(self.true_energy_GeV, dtype=float)
        pd.DataFrame({
            "true_bin": np.arange(energy.size, dtype=int),
            "true_energy_GeV": energy,
        }).to_csv(directory / "true_energy_GeV.csv", index=False, float_format="%.17g")
        pd.DataFrame({
            "global_reco_bin": np.arange(104, dtype=int),
            "fixed_published_background_counts": self.fixed_published_background_counts,
        }).to_csv(
            directory / "fixed_published_background_counts.csv",
            index=False,
            float_format="%.17g",
        )
        matrix_columns = [f"true_bin_{index:03d}" for index in range(energy.size)]
        for name in RESPONSE_TEMPLATE_FIELDS:
            table = pd.DataFrame(np.asarray(getattr(self, name), dtype=float), columns=matrix_columns)
            table.insert(0, "global_reco_bin", np.arange(104, dtype=int))
            table.to_csv(directory / f"{name}.csv", index=False, float_format="%.17g")
        document = {
            **metadata,
            "format": "bnb_four_channel_text_templates_v1",
            "shape": {"reconstructed_bins": 104, "true_energy_bins": int(energy.size)},
            "reconstructed_bin_order": "nue_cc_fc,nue_cc_pc,numu_cc_fc,numu_cc_pc; 26 bins each",
        }
        (directory / "metadata.json").write_text(
            json.dumps(document, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def predict_total_counts(self, model: VacuumOscillationModel, baseline_km: float) -> NDArray[np.float64]:
        """Return BNB prediction after applying all oscillation probabilities."""
        energy = self.true_energy_GeV
        probability = model.probability
        predicted = np.asarray(self.fixed_published_background_counts, dtype=float).copy()
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
