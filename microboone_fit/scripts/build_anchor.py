"""Build a visible, exactly closing BNB reference-reweight kernel."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from sterile_fit.binning import BNB_FOUR_CHANNELS
from sterile_fit.models.three_plus_one import ThreePlusOneVacuumModel
from sterile_fit.parameters import ThreePlusOneParameters
from sterile_fit.published_inputs import load_bnb_four_channel_inputs
from sterile_fit.templates import BnbFourChannelOscillationTemplates


PROCESS_FIELDS = {
    ("nue", "nue"): "beam_nue_to_nue_cc_response_counts",
    ("numu", "nue"): "beam_numu_to_nue_cc_response_counts",
    ("nue", "numu"): "beam_nue_to_numu_cc_response_counts",
    ("numu", "numu"): "beam_numu_to_numu_cc_response_counts",
    ("nuebar", "nue"): "beam_nuebar_to_nuebar_cc_response_counts",
    ("numubar", "nue"): "beam_numubar_to_nuebar_cc_response_counts",
    ("nuebar", "numu"): "beam_nuebar_to_numubar_cc_response_counts",
    ("numubar", "numu"): "beam_numubar_to_numubar_cc_response_counts",
}


def _load_reference_setup(path: Path) -> tuple[ThreePlusOneParameters, float]:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    values = document["reference_parameters"]
    parameters = ThreePlusOneParameters(
        delta_m2_41_eV2=float(values["delta_m2_41_eV2"]),
        sin2_theta14=float(values["sin2_theta14"]),
        sin2_theta24=float(values["sin2_theta24"]),
    )
    baseline_km = float(document["baseline_km"])
    if not np.isfinite(baseline_km) or baseline_km <= 0.0:
        raise ValueError("baseline_km must be finite and positive")
    return parameters, baseline_km


def _load_visible_response(directory: Path, channel: str) -> np.ndarray:
    path = directory / f"{channel}_reco_given_true.csv"
    table = pd.read_csv(path)
    expected = ["reco_bin", *[f"true_bin_{index:03d}" for index in range(60)]]
    if list(table.columns) != expected or table.shape != (26, 61):
        raise ValueError(f"unexpected visible 26x60 response table: {path}")
    if not np.array_equal(table["reco_bin"].to_numpy(dtype=int), np.arange(26)):
        raise ValueError(f"reco_bin must be 0..25 in {path}")
    return table.iloc[:, 1:].to_numpy(dtype=float)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build K(reco,true,process) from the public reference prediction using visible inputs."
    )
    parser.add_argument("--reference-config", type=Path, default=Path("configs/bnb_3plus1_reference.yaml"))
    parser.add_argument("--flux", type=Path, default=Path("data/inputs/bnb_flux.csv"))
    parser.add_argument(
        "--response-directory",
        type=Path,
        default=Path("data/derived/archival_2022_reco_bnb26_given_true"),
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("data/anchor/bnb_reference_reweight_kernel"),
    )
    arguments = parser.parse_args()

    parameters, baseline_km = _load_reference_setup(arguments.reference_config)
    model = ThreePlusOneVacuumModel(parameters)
    inputs = load_bnb_four_channel_inputs()
    flux_table = pd.read_csv(arguments.flux)
    expected_flux_columns = [
        "true_bin",
        "true_energy_GeV",
        "numu_flux",
        "numubar_flux",
        "nue_flux",
        "nuebar_flux",
    ]
    if list(flux_table.columns) != expected_flux_columns or flux_table.shape != (60, 6):
        raise ValueError("bnb_flux.csv has an unexpected visible schema")
    energy = flux_table["true_energy_GeV"].to_numpy(dtype=float)
    flux = {
        "nue": flux_table["nue_flux"].to_numpy(dtype=float),
        "numu": flux_table["numu_flux"].to_numpy(dtype=float),
        "nuebar": flux_table["nuebar_flux"].to_numpy(dtype=float),
        "numubar": flux_table["numubar_flux"].to_numpy(dtype=float),
    }
    process_arrays = {name: np.zeros((104, 60), dtype=float) for name in PROCESS_FIELDS.values()}
    closure_rows: list[dict[str, object]] = []

    for channel in BNB_FOUR_CHANNELS:
        final_name = "nue" if channel.identifier.startswith("nue_") else "numu"
        final_index = 0 if final_name == "nue" else 1
        response = _load_visible_response(arguments.response_directory, channel.identifier)
        source_probability = {
            "nue": model.probability(0, final_index, energy, baseline_km=baseline_km),
            "numu": model.probability(1, final_index, energy, baseline_km=baseline_km),
            "nuebar": model.probability(0, final_index, energy, baseline_km=baseline_km, antineutrino=True),
            "numubar": model.probability(1, final_index, energy, baseline_km=baseline_km, antineutrino=True),
        }
        reference_true_weight = sum(flux[name] * source_probability[name] for name in flux)
        denominator = response @ reference_true_weight
        start, stop = channel.first_global_bin, channel.stop_global_bin
        published_signal = inputs.published_signal_counts[start:stop]
        if np.any((denominator <= 0.0) & (published_signal > 0.0)):
            raise ValueError(f"reference response has zero support for a nonzero {channel.identifier} bin")
        scale = np.divide(
            published_signal,
            denominator,
            out=np.zeros_like(published_signal),
            where=denominator > 0.0,
        )
        for source_name in flux:
            field = PROCESS_FIELDS[(source_name, final_name)]
            process_arrays[field][start:stop, :] = scale[:, None] * response * flux[source_name][None, :]

    templates = BnbFourChannelOscillationTemplates(
        true_energy_GeV=energy,
        fixed_published_background_counts=inputs.published_background_counts,
        **process_arrays,
    )
    reconstructed = templates.predict_total_counts(model, baseline_km=baseline_km)
    residual = reconstructed - inputs.published_total_prediction_counts
    if not np.allclose(reconstructed, inputs.published_total_prediction_counts, rtol=1e-12, atol=1e-10):
        largest = int(np.argmax(np.abs(residual)))
        raise RuntimeError(f"reference kernel failed exact closure at global reco bin {largest}")

    templates.to_directory(
        arguments.output_directory,
        metadata={
            "construction": "per-reconstructed-bin reference-ratio kernel",
            "baseline_km": baseline_km,
            "baseline_interpretation": "BNB target-to-detector distance; one-baseline approximation",
            "formula": "K_r,t,source = S_reference_r * R_r,t * flux_source_t / sum_t,source(R_r,t * flux_source_t * P_source_to_final(reference))",
            "reference_parameters": {
                "delta_m2_41_eV2": parameters.delta_m2_41_eV2,
                "sin2_theta14": parameters.sin2_theta14,
                "sin2_theta24": parameters.sin2_theta24,
                "sin2_2theta_mue_exact": parameters.sin2_2theta_mue_exact,
            },
            "fixed_background_definition": "HEPData published Background block, frozen during reweighting",
            "reference_total_definition": "HEPData published Signal + Background block",
            "assumptions": [
                "unknown cross section and efficiency are not separately identified; their reference-weighted effect is assumed to be absorbed by the chosen response-times-flux true-energy prior and one scale factor per reconstructed bin",
                "neutrino and antineutrino contributions to the same selected final-state channel share that effective ratio",
                "the 2022 normalized migration matrix is used as the declared true-energy prior",
                "the HEPData reference prediction is assigned to the configured 3+1 reference point",
            ],
        },
    )
    for channel in BNB_FOUR_CHANNELS:
        for local_bin, global_bin in enumerate(range(channel.first_global_bin, channel.stop_global_bin)):
            closure_rows.append({
                "channel": channel.identifier,
                "channel_reco_bin": local_bin,
                "global_reco_bin": global_bin,
                "published_data_counts": inputs.observed_counts[global_bin],
                "published_background_counts": inputs.published_background_counts[global_bin],
                "published_signal_counts": inputs.published_signal_counts[global_bin],
                "published_total_prediction_counts": inputs.published_total_prediction_counts[global_bin],
                "reconstructed_total_prediction_counts": reconstructed[global_bin],
                "closure_residual_counts": residual[global_bin],
            })
    pd.DataFrame(closure_rows).to_csv(
        arguments.output_directory / "reference_closure.csv",
        index=False,
        float_format="%.17g",
    )
    print(arguments.output_directory)


if __name__ == "__main__":
    main()
