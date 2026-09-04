"""NuMI four-channel empirical event reweighting, isolated from BNB code."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from ....models.base import VacuumOscillationModel


PROCESS_FIELDS = (
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
class NumiFourChannelEmpiricalKernel:
    """Eight source-to-selected-flavour event kernels on 60 true-energy bins.

    These arrays are an algebraic reference-ratio construction, not measured
    cross sections or efficiencies.  The fixed published Background category is
    added exactly once and is not reweighted because its component templates are
    absent from the public release.
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
        if energy.shape != (60,) or np.any(energy <= 0.0) or not np.all(np.diff(energy) > 0.0):
            raise ValueError("true_energy_GeV must be the increasing 60-bin Reco grid")
        background = np.asarray(self.fixed_published_background_counts, dtype=float)
        if background.shape != (104,) or np.any(background < 0.0):
            raise ValueError("fixed_published_background_counts must be non-negative with shape (104,)")
        for name in PROCESS_FIELDS:
            values = np.asarray(getattr(self, name), dtype=float)
            if values.shape != (104, 60) or not np.all(np.isfinite(values)) or np.any(values < 0.0):
                raise ValueError(f"{name} must be finite and non-negative with shape (104, 60)")

    @classmethod
    def from_directory(cls, directory: Path) -> "NumiFourChannelEmpiricalKernel":
        """Load the visible BNB-style NuMI kernel contract."""
        directory = Path(directory)
        metadata_path = directory / "metadata.json"
        energy_path = directory / "true_energy_GeV.csv"
        background_path = directory / "fixed_published_background_counts.csv"
        required = [metadata_path, energy_path, background_path]
        required.extend(directory / f"{name}.csv" for name in PROCESS_FIELDS)
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise ValueError(f"NuMI kernel is missing visible files: {missing}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("format") != "numi_four_channel_empirical_kernel_v1":
            raise ValueError("NuMI kernel metadata format is not recognized")
        energy_table = pd.read_csv(energy_path)
        if list(energy_table.columns) != ["true_bin", "true_energy_GeV"]:
            raise ValueError("NuMI true-energy CSV has unexpected columns")
        if not np.array_equal(energy_table["true_bin"].to_numpy(dtype=int), np.arange(60)):
            raise ValueError("NuMI true-energy bins must be contiguous 0..59")
        background_table = pd.read_csv(background_path)
        if list(background_table.columns) != [
            "local_numi_reco_bin",
            "fixed_published_background_counts",
        ]:
            raise ValueError("NuMI fixed-background CSV has unexpected columns")
        if not np.array_equal(
            background_table["local_numi_reco_bin"].to_numpy(dtype=int), np.arange(104)
        ):
            raise ValueError("NuMI fixed-background bins must be contiguous 0..103")
        values: dict[str, NDArray[np.float64]] = {
            "true_energy_GeV": energy_table["true_energy_GeV"].to_numpy(dtype=float),
            "fixed_published_background_counts": background_table[
                "fixed_published_background_counts"
            ].to_numpy(dtype=float),
        }
        expected_columns = [
            "local_numi_reco_bin",
            *[f"true_bin_{index:03d}" for index in range(60)],
        ]
        for name in PROCESS_FIELDS:
            table = pd.read_csv(directory / f"{name}.csv")
            if list(table.columns) != expected_columns or table.shape != (104, 61):
                raise ValueError(f"{name}.csv has an unexpected visible schema")
            if not np.array_equal(
                table["local_numi_reco_bin"].to_numpy(dtype=int), np.arange(104)
            ):
                raise ValueError(f"{name}.csv local bins must be contiguous 0..103")
            values[name] = table.iloc[:, 1:].to_numpy(dtype=float)
        return cls(**values)

    def component_counts(
        self, model: VacuumOscillationModel, baseline_km: float
    ) -> dict[str, NDArray[np.float64]]:
        energy = self.true_energy_GeV
        probability = model.probability
        return {
            "nue_to_nue": self.beam_nue_to_nue_cc_response_counts
            @ probability(0, 0, energy, baseline_km),
            "numu_to_nue": self.beam_numu_to_nue_cc_response_counts
            @ probability(1, 0, energy, baseline_km),
            "nue_to_numu": self.beam_nue_to_numu_cc_response_counts
            @ probability(0, 1, energy, baseline_km),
            "numu_to_numu": self.beam_numu_to_numu_cc_response_counts
            @ probability(1, 1, energy, baseline_km),
            "nuebar_to_nuebar": self.beam_nuebar_to_nuebar_cc_response_counts
            @ probability(0, 0, energy, baseline_km, antineutrino=True),
            "numubar_to_nuebar": self.beam_numubar_to_nuebar_cc_response_counts
            @ probability(1, 0, energy, baseline_km, antineutrino=True),
            "nuebar_to_numubar": self.beam_nuebar_to_numubar_cc_response_counts
            @ probability(0, 1, energy, baseline_km, antineutrino=True),
            "numubar_to_numubar": self.beam_numubar_to_numubar_cc_response_counts
            @ probability(1, 1, energy, baseline_km, antineutrino=True),
        }

    def predict_total_counts(
        self, model: VacuumOscillationModel, baseline_km: float
    ) -> NDArray[np.float64]:
        components = self.component_counts(model, baseline_km)
        prediction = self.fixed_published_background_counts + sum(components.values())
        if not np.all(np.isfinite(prediction)) or np.any(prediction < 0.0):
            raise FloatingPointError("NuMI event prediction contains invalid counts")
        return prediction
