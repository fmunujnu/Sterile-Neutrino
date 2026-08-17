"""Build four exposure-averaged, oscillation-weighted NuMI flavour flux CSVs."""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd

from sterile_fit.models.three_plus_one import ThreePlusOneVacuumModel
from sterile_fit.parameters import ThreePlusOneParameters


ROOT = Path(__file__).resolve().parents[4]
INPUT_DIRECTORY = ROOT / "data" / "experiments" / "microboone" / "numi" / "inputs" / "flux_components"
OUTPUT_DIRECTORY = ROOT / "data" / "experiments" / "microboone" / "numi" / "derived" / "paper_figure3_weighted_flux"

DELTA_M2_41_EV2 = 1.2
SIN2_2THETA_MUE = 0.003
SIN2_THETA24_VALUES = (0.018, 0.0045)
NUMI_BASELINE_KM = 0.680
TOTAL_EXPOSURE_POT = 10.54e20
FHC_EXPOSURE_FRACTION = 0.308
RHC_EXPOSURE_FRACTION = 0.692
FLAVOURS = ("numu", "numubar", "nue", "nuebar")


def _parameters(sin2_theta24: float) -> ThreePlusOneParameters:
    """Convert the paper's effective angle on its stated small-theta14 branch."""

    sin2_2theta14 = SIN2_2THETA_MUE / sin2_theta24
    if not 0.0 <= sin2_2theta14 <= 1.0:
        raise ValueError("sin2(2theta_mue)/sin2(theta24) is outside [0, 1]")
    sin2_theta14 = (1.0 - np.sqrt(1.0 - sin2_2theta14)) / 2.0
    parameters = ThreePlusOneParameters(
        delta_m2_41_eV2=DELTA_M2_41_EV2,
        sin2_theta14=float(sin2_theta14),
        sin2_theta24=sin2_theta24,
    )
    if not np.isclose(parameters.sin2_2theta_mue_exact, SIN2_2THETA_MUE, rtol=1e-12, atol=1e-15):
        raise RuntimeError("paper effective-angle conversion failed its exact-amplitude check")
    return parameters


def _load_fluxes() -> tuple[np.ndarray, dict[str, dict[str, np.ndarray]], dict[str, str]]:
    fluxes: dict[str, dict[str, np.ndarray]] = {"fhc": {}, "rhc": {}}
    hashes: dict[str, str] = {}
    reference_edges: tuple[np.ndarray, np.ndarray] | None = None
    centers: np.ndarray | None = None
    for horn_mode in ("fhc", "rhc"):
        for flavour in FLAVOURS:
            path = INPUT_DIRECTORY / f"numi_{horn_mode}_{flavour}_flux.csv"
            table = pd.read_csv(path)
            required = {
                "energy_low_GeV",
                "energy_high_GeV",
                "energy_center_GeV",
                "flux_per_POT_per_cm2_per_100MeV",
                "is_censored",
            }
            if not required.issubset(table.columns) or table.shape[0] != 50:
                raise ValueError(f"unexpected NuMI flux input: {path}")
            low = table["energy_low_GeV"].to_numpy(dtype=float)
            high = table["energy_high_GeV"].to_numpy(dtype=float)
            current_centers = table["energy_center_GeV"].to_numpy(dtype=float)
            if reference_edges is None:
                reference_edges = low, high
                centers = current_centers
            elif not (
                np.allclose(low, reference_edges[0], atol=1e-12)
                and np.allclose(high, reference_edges[1], atol=1e-12)
                and np.allclose(current_centers, centers, atol=1e-12)
            ):
                raise ValueError("all eight NuMI flux inputs must use exactly the same grid")
            values = table["flux_per_POT_per_cm2_per_100MeV"].to_numpy(dtype=float)
            if not np.all(np.isfinite(values)) or np.any(values < 0.0):
                raise ValueError(f"invalid flux values: {path}")
            if table["is_censored"].astype(bool).any():
                raise ValueError(f"censored input cannot be used for oscillation weighting: {path}")
            fluxes[horn_mode][flavour] = values
            hashes[path.name] = sha256(path.read_bytes()).hexdigest().upper()
    assert centers is not None
    return centers, fluxes, hashes


def _oscillated_mode_flux(
    source: dict[str, np.ndarray],
    energy_GeV: np.ndarray,
    parameters: ThreePlusOneParameters,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    model = ThreePlusOneVacuumModel(parameters)
    probabilities = {
        "nue_to_nue": model.probability(0, 0, energy_GeV, NUMI_BASELINE_KM),
        "numu_to_nue": model.probability(1, 0, energy_GeV, NUMI_BASELINE_KM),
        "nue_to_numu": model.probability(0, 1, energy_GeV, NUMI_BASELINE_KM),
        "numu_to_numu": model.probability(1, 1, energy_GeV, NUMI_BASELINE_KM),
        "nuebar_to_nuebar": model.probability(
            0, 0, energy_GeV, NUMI_BASELINE_KM, antineutrino=True
        ),
        "numubar_to_nuebar": model.probability(
            1, 0, energy_GeV, NUMI_BASELINE_KM, antineutrino=True
        ),
        "nuebar_to_numubar": model.probability(
            0, 1, energy_GeV, NUMI_BASELINE_KM, antineutrino=True
        ),
        "numubar_to_numubar": model.probability(
            1, 1, energy_GeV, NUMI_BASELINE_KM, antineutrino=True
        ),
    }
    output = {
        "nue": source["nue"] * probabilities["nue_to_nue"]
        + source["numu"] * probabilities["numu_to_nue"],
        "numu": source["numu"] * probabilities["numu_to_numu"]
        + source["nue"] * probabilities["nue_to_numu"],
        "nuebar": source["nuebar"] * probabilities["nuebar_to_nuebar"]
        + source["numubar"] * probabilities["numubar_to_nuebar"],
        "numubar": source["numubar"] * probabilities["numubar_to_numubar"]
        + source["nuebar"] * probabilities["nuebar_to_numubar"],
    }
    return output, probabilities


def main() -> None:
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    energy, source_flux, input_hashes = _load_fluxes()
    low = energy - 0.05
    high = energy + 0.05
    no_oscillation = {
        flavour: FHC_EXPOSURE_FRACTION * source_flux["fhc"][flavour]
        + RHC_EXPOSURE_FRACTION * source_flux["rhc"][flavour]
        for flavour in FLAVOURS
    }

    weighted_by_point: dict[float, dict[str, np.ndarray]] = {}
    parameter_metadata = []
    for sin2_theta24 in SIN2_THETA24_VALUES:
        parameters = _parameters(sin2_theta24)
        fhc_output, _ = _oscillated_mode_flux(source_flux["fhc"], energy, parameters)
        rhc_output, _ = _oscillated_mode_flux(source_flux["rhc"], energy, parameters)
        weighted_by_point[sin2_theta24] = {
            flavour: FHC_EXPOSURE_FRACTION * fhc_output[flavour]
            + RHC_EXPOSURE_FRACTION * rhc_output[flavour]
            for flavour in FLAVOURS
        }
        parameter_metadata.append(
            {
                "delta_m2_41_eV2": parameters.delta_m2_41_eV2,
                "labelled_sin2_2theta_mue": SIN2_2THETA_MUE,
                "sin2_theta14": parameters.sin2_theta14,
                "sin2_theta24": parameters.sin2_theta24,
                "sin2_2theta_mue_exact": parameters.sin2_2theta_mue_exact,
            }
        )

    final_files = []
    for flavour in FLAVOURS:
        table = pd.DataFrame(
            {
                "energy_low_GeV": low,
                "energy_high_GeV": high,
                "energy_center_GeV": energy,
                "no_oscillation_exposure_weighted_flux_per_POT_per_cm2_per_100MeV": no_oscillation[flavour],
                "paper_sin2_theta24_0p018_flux_per_POT_per_cm2_per_100MeV": weighted_by_point[0.018][flavour],
                "paper_sin2_theta24_0p0045_flux_per_POT_per_cm2_per_100MeV": weighted_by_point[0.0045][flavour],
            }
        )
        if not np.all(np.isfinite(table.iloc[:, 3:].to_numpy(dtype=float))):
            raise RuntimeError(f"non-finite weighted flux for {flavour}")
        if np.any(table.iloc[:, 3:].to_numpy(dtype=float) < 0.0):
            raise RuntimeError(f"negative weighted flux for {flavour}")
        destination = OUTPUT_DIRECTORY / f"numi_exposure_weighted_{flavour}_flux.csv"
        visible_table = table.copy()
        visible_table["energy_low_GeV"] = visible_table["energy_low_GeV"].map(lambda value: f"{value:.1f}")
        visible_table["energy_high_GeV"] = visible_table["energy_high_GeV"].map(lambda value: f"{value:.1f}")
        visible_table["energy_center_GeV"] = visible_table["energy_center_GeV"].map(lambda value: f"{value:.2f}")
        visible_table.to_csv(destination, index=False, float_format="%.17g")
        final_files.append(str(destination))

    metadata = {
        "status": "diagnostic_flux_level_only",
        "final_file_count": len(final_files),
        "final_files": final_files,
        "input_flux_sha256": input_hashes,
        "flux_unit": "neutrinos / POT / cm^2 / 100 MeV",
        "energy_grid": {"minimum_GeV": 0.0, "maximum_GeV": 5.0, "bin_width_GeV": 0.1, "bins": 50},
        "exposure_weighting": {
            "total_NuMI_POT": TOTAL_EXPOSURE_POT,
            "FHC_fraction": FHC_EXPOSURE_FRACTION,
            "RHC_fraction": RHC_EXPOSURE_FRACTION,
            "formula": "combined flux per POT = 0.308 * FHC flux per POT + 0.692 * RHC flux per POT",
        },
        "oscillation_weighting": {
            "baseline_km": NUMI_BASELINE_KM,
            "baseline_treatment": "single target-to-detector baseline approximation",
            "parameter_points": parameter_metadata,
            "final_e_formula": "Phi_e * P(e->e) + Phi_mu * P(mu->e)",
            "final_mu_formula": "Phi_mu * P(mu->mu) + Phi_e * P(e->mu)",
            "antineutrino_formula": "same construction using antineutrino probabilities",
        },
        "paper_sources": {
            "parameter_note": "MICROBOONE-NOTE-1132-PUB Figure 3 discussion",
            "parameter_note_url": "https://microboone.fnal.gov/wp-content/uploads/MICROBOONE-NOTE-1132-PUB.pdf",
            "exposure_paper_doi": "10.1038/s41586-025-09757-7",
        },
        "scientific_boundary": (
            "The input fluxes are recovered from displayed PDF vector paths. The fixed 0.680 km baseline "
            "does not integrate over the NuMI decay-position distribution. These files are diagnostic flux-level "
            "weights, not detector-folded event predictions and not inputs that validate a NuMI likelihood."
        ),
    }
    (OUTPUT_DIRECTORY / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({"status": "pass", "files": final_files}, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
