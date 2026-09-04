"""Build isolated NuMI four-channel empirical kernels and per-bin events.

This script intentionally does not register a NuMI likelihood.  It reuses the
declared 26x60 response matrices and the BNB reference-ratio construction while
reading NuMI channel blocks 8--11 and NuMI fluxes in a separate adapter layer.
"""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd

from sterile_fit.experiments.microboone.numi.binning import NUMI_FOUR_CHANNELS
from sterile_fit.experiments.microboone.numi.event_prediction import (
    PROCESS_FIELDS,
    NumiFourChannelEmpiricalKernel,
)
from sterile_fit.experiments.microboone.numi.published_inputs import (
    DEFAULT_COVARIANCE_PATH,
    DEFAULT_SPECTRUM_PATH,
    load_numi_four_channel_inputs,
)
from sterile_fit.models.three_plus_one import ThreePlusOneVacuumModel
from sterile_fit.parameters import ThreePlusOneParameters


ROOT = Path(__file__).resolve().parents[4]
BNB_RESPONSE_DIRECTORY = (
    ROOT
    / "data"
    / "experiments"
    / "microboone"
    / "bnb"
    / "derived"
    / "archival_2022_reco_bnb26_given_true"
)
NUMI_FLUX_DIRECTORY = (
    ROOT
    / "data"
    / "experiments"
    / "microboone"
    / "numi"
    / "derived"
    / "paper_figure3_weighted_flux"
)
NUMI_DATA_DIRECTORY = (
    ROOT
    / "data"
    / "experiments"
    / "microboone"
    / "numi"
)
KERNEL_DIRECTORY = NUMI_DATA_DIRECTORY / "reweighting"
DERIVED_DIRECTORY = NUMI_DATA_DIRECTORY / "derived"
EVENT_OUTPUT = (
    ROOT
    / "outputs"
    / "spectra"
    / "microboone"
    / "numi"
    / "paper_parameter_event_counts.csv"
)

NUMI_BASELINE_KM = 0.680
DELTA_M2_41_EV2 = 1.2
SIN2_2THETA_MUE = 0.003
SIN2_THETA24_VALUES = (0.018, 0.0045)
FLAVOURS = ("nue", "numu", "nuebar", "numubar")
PROCESS_BY_SOURCE_AND_FINAL = {
    ("nue", "nue"): "beam_nue_to_nue_cc_response_counts",
    ("numu", "nue"): "beam_numu_to_nue_cc_response_counts",
    ("nue", "numu"): "beam_nue_to_numu_cc_response_counts",
    ("numu", "numu"): "beam_numu_to_numu_cc_response_counts",
    ("nuebar", "nue"): "beam_nuebar_to_nuebar_cc_response_counts",
    ("numubar", "nue"): "beam_numubar_to_nuebar_cc_response_counts",
    ("nuebar", "numu"): "beam_nuebar_to_numubar_cc_response_counts",
    ("numubar", "numu"): "beam_numubar_to_numubar_cc_response_counts",
}


def _sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest().upper()


def _load_response(channel_identifier: str) -> np.ndarray:
    path = BNB_RESPONSE_DIRECTORY / f"{channel_identifier}_reco_given_true.csv"
    table = pd.read_csv(path)
    expected_columns = ["reco_bin", *[f"true_bin_{index:03d}" for index in range(60)]]
    if list(table.columns) != expected_columns or table.shape != (26, 61):
        raise ValueError(f"unexpected 26x60 response schema: {path}")
    if not np.array_equal(table["reco_bin"].to_numpy(dtype=int), np.arange(26)):
        raise ValueError(f"reco_bin must be 0..25 in {path}")
    response = table.iloc[:, 1:].to_numpy(dtype=float)
    if np.any(response < 0.0) or not np.all(np.isfinite(response)):
        raise ValueError(f"invalid response values: {path}")
    column_sums = response.sum(axis=0)
    if not np.all(np.isclose(column_sums, 1.0, atol=1e-12) | np.isclose(column_sums, 0.0, atol=1e-12)):
        raise ValueError(f"response columns are neither normalized nor zero: {path}")
    return response


def _load_true_energy() -> np.ndarray:
    path = BNB_RESPONSE_DIRECTORY / "true_energy_GeV.csv"
    table = pd.read_csv(path)
    if list(table.columns) != ["true_bin", "true_energy_GeV"] or table.shape != (60, 2):
        raise ValueError("Reco true-energy grid must contain 60 visible bins")
    if not np.array_equal(table["true_bin"].to_numpy(dtype=int), np.arange(60)):
        raise ValueError("Reco true_bin must be 0..59")
    energy = table["true_energy_GeV"].to_numpy(dtype=float)
    if not np.allclose(energy, 0.025 + 0.05 * np.arange(60), atol=1e-12):
        raise ValueError("expected the declared 0--3 GeV, 0.05 GeV Reco grid")
    return energy


def _load_source_flux_on_reco_grid(true_energy_GeV: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    """Conservatively split each 0.1 GeV flux bin into two 0.05 GeV bins."""
    output: dict[str, np.ndarray] = {}
    hashes: dict[str, str] = {}
    column = "no_oscillation_exposure_weighted_flux_per_POT_per_cm2_per_100MeV"
    for flavour in FLAVOURS:
        path = NUMI_FLUX_DIRECTORY / f"numi_exposure_weighted_{flavour}_flux.csv"
        table = pd.read_csv(path)
        if table.shape[0] != 50 or column not in table.columns:
            raise ValueError(f"unexpected NuMI flux schema: {path}")
        low = table["energy_low_GeV"].to_numpy(dtype=float)
        high = table["energy_high_GeV"].to_numpy(dtype=float)
        values = table[column].to_numpy(dtype=float)
        selected = np.empty(60, dtype=float)
        for index, energy in enumerate(true_energy_GeV):
            matches = np.flatnonzero((low <= energy) & (energy < high))
            if matches.size != 1:
                raise ValueError(f"NuMI flux bin is not unique at E={energy:.6g} GeV")
            # Input is per 100 MeV; each Reco true bin is 50 MeV wide.
            selected[index] = 0.5 * values[matches[0]]
        if not np.all(np.isfinite(selected)) or np.any(selected < 0.0):
            raise ValueError(f"invalid resampled flux: {path}")
        output[flavour] = selected
        hashes[path.name] = _sha256(path)
    return output, hashes


def _paper_parameters(sin2_theta24: float) -> ThreePlusOneParameters:
    sin2_2theta14 = SIN2_2THETA_MUE / sin2_theta24
    sin2_theta14 = (1.0 - np.sqrt(1.0 - sin2_2theta14)) / 2.0
    parameters = ThreePlusOneParameters(
        delta_m2_41_eV2=DELTA_M2_41_EV2,
        sin2_theta14=float(sin2_theta14),
        sin2_theta24=sin2_theta24,
    )
    if not np.isclose(parameters.sin2_2theta_mue_exact, SIN2_2THETA_MUE, rtol=1e-12, atol=1e-15):
        raise RuntimeError("paper effective-angle conversion failed")
    return parameters


def _write_matrix(path: Path, matrix: np.ndarray, first_column: str) -> None:
    table = pd.DataFrame(matrix, columns=[f"true_bin_{index:03d}" for index in range(matrix.shape[1])])
    table.insert(0, first_column, np.arange(matrix.shape[0], dtype=int))
    table.to_csv(path, index=False, float_format="%.17g")


def main() -> None:
    KERNEL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    DERIVED_DIRECTORY.mkdir(parents=True, exist_ok=True)
    EVENT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    inputs = load_numi_four_channel_inputs()
    true_energy = _load_true_energy()
    source_flux, flux_hashes = _load_source_flux_on_reco_grid(true_energy)

    reference_parameters = ThreePlusOneParameters(
        delta_m2_41_eV2=DELTA_M2_41_EV2,
        sin2_theta14=0.0,
        sin2_theta24=0.0,
    )
    reference_model = ThreePlusOneVacuumModel(reference_parameters)
    process_arrays = {name: np.zeros((104, 60), dtype=float) for name in PROCESS_FIELDS}

    for local_channel_index, channel in enumerate(NUMI_FOUR_CHANNELS):
        final_name = "nue" if channel.identifier.startswith("nue_") else "numu"
        final_index = 0 if final_name == "nue" else 1
        response = _load_response(channel.identifier)
        source_probability = {
            "nue": reference_model.probability(0, final_index, true_energy, NUMI_BASELINE_KM),
            "numu": reference_model.probability(1, final_index, true_energy, NUMI_BASELINE_KM),
            "nuebar": reference_model.probability(
                0, final_index, true_energy, NUMI_BASELINE_KM, antineutrino=True
            ),
            "numubar": reference_model.probability(
                1, final_index, true_energy, NUMI_BASELINE_KM, antineutrino=True
            ),
        }
        reference_true_weight = sum(
            source_flux[source] * source_probability[source] for source in FLAVOURS
        )
        denominator = response @ reference_true_weight
        local_start = local_channel_index * 26
        local_stop = local_start + 26
        published_signal = inputs.published_signal_counts[local_start:local_stop]
        if np.any((denominator <= 0.0) & (published_signal > 0.0)):
            raise ValueError(f"zero response support for nonzero {channel.identifier} signal")
        scale = np.divide(
            published_signal,
            denominator,
            out=np.zeros_like(published_signal),
            where=denominator > 0.0,
        )
        for source in FLAVOURS:
            field = PROCESS_BY_SOURCE_AND_FINAL[(source, final_name)]
            process_arrays[field][local_start:local_stop, :] = (
                scale[:, None] * response * source_flux[source][None, :]
            )

    kernel = NumiFourChannelEmpiricalKernel(
        true_energy_GeV=true_energy,
        fixed_published_background_counts=inputs.published_background_counts,
        **process_arrays,
    )
    reference_prediction = kernel.predict_total_counts(reference_model, NUMI_BASELINE_KM)
    reference_residual = reference_prediction - inputs.published_total_prediction_counts
    if not np.allclose(reference_residual, 0.0, rtol=0.0, atol=1e-10):
        largest = int(np.argmax(np.abs(reference_residual)))
        raise RuntimeError(f"reference closure failed at local NuMI bin {largest}")

    paper_predictions: dict[float, np.ndarray] = {}
    paper_components: dict[float, dict[str, np.ndarray]] = {}
    parameter_records: list[dict[str, float]] = []
    for sin2_theta24 in SIN2_THETA24_VALUES:
        parameters = _paper_parameters(sin2_theta24)
        model = ThreePlusOneVacuumModel(parameters)
        paper_components[sin2_theta24] = kernel.component_counts(model, NUMI_BASELINE_KM)
        paper_predictions[sin2_theta24] = (
            inputs.published_background_counts + sum(paper_components[sin2_theta24].values())
        )
        parameter_records.append(
            {
                "delta_m2_41_eV2": parameters.delta_m2_41_eV2,
                "sin2_theta14": parameters.sin2_theta14,
                "sin2_theta24": parameters.sin2_theta24,
                "sin2_2theta_mue_exact": parameters.sin2_2theta_mue_exact,
            }
        )

    rows: list[dict[str, object]] = []
    for local_channel_index, channel in enumerate(NUMI_FOUR_CHANNELS):
        for channel_reco_bin in range(26):
            local_bin = local_channel_index * 26 + channel_reco_bin
            published_bin = channel.first_published_bin + channel_reco_bin
            row: dict[str, object] = {
                "channel": channel.identifier,
                "released_channel_ordinal": channel.released_channel_ordinal,
                "channel_reco_bin": channel_reco_bin,
                "local_numi_reco_bin": local_bin,
                "published_global_bin": published_bin,
                "reco_energy_low_GeV": 0.1 * channel_reco_bin if channel_reco_bin < 25 else 2.5,
                "reco_energy_high_GeV": 0.1 * (channel_reco_bin + 1) if channel_reco_bin < 25 else 3.0,
                "is_overflow_bin": channel_reco_bin == 25,
                "observed_counts": inputs.observed_counts[local_bin],
                "observed_statistical_error_up": inputs.observed_statistical_error_up[local_bin],
                "observed_statistical_error_down": inputs.observed_statistical_error_down[local_bin],
                "published_background_counts": inputs.published_background_counts[local_bin],
                "published_signal_counts": inputs.published_signal_counts[local_bin],
                "published_total_prediction_counts": inputs.published_total_prediction_counts[local_bin],
                "empirical_reference_total_counts": reference_prediction[local_bin],
                "empirical_reference_closure_residual_counts": reference_residual[local_bin],
            }
            for sin2_theta24 in SIN2_THETA24_VALUES:
                label = "0p018" if sin2_theta24 == 0.018 else "0p0045"
                row[f"paper_sin2_theta24_{label}_total_counts"] = paper_predictions[sin2_theta24][local_bin]
                for component_name, values in paper_components[sin2_theta24].items():
                    row[f"paper_sin2_theta24_{label}_{component_name}_counts"] = values[local_bin]
            rows.append(row)
    pd.DataFrame(rows).to_csv(
        EVENT_OUTPUT,
        index=False,
        float_format="%.17g",
    )

    covariance_columns = [f"local_numi_reco_bin_{index:03d}" for index in range(104)]
    covariance_table = pd.DataFrame(inputs.systematic_covariance, columns=covariance_columns)
    covariance_table.insert(0, "local_numi_reco_bin", np.arange(104, dtype=int))
    covariance_table.to_csv(
        DERIVED_DIRECTORY / "numi_four_channel_systematic_covariance.csv",
        index=False,
        float_format="%.17g",
    )
    block_rows = []
    for row_channel_index, row_channel in enumerate(NUMI_FOUR_CHANNELS):
        for column_channel_index, column_channel in enumerate(NUMI_FOUR_CHANNELS):
            block_rows.append(
                {
                    "block_row": row_channel_index,
                    "block_column": column_channel_index,
                    "row_channel": row_channel.identifier,
                    "column_channel": column_channel.identifier,
                    "local_row_start_inclusive": 26 * row_channel_index,
                    "local_row_stop_exclusive": 26 * (row_channel_index + 1),
                    "local_column_start_inclusive": 26 * column_channel_index,
                    "local_column_stop_exclusive": 26 * (column_channel_index + 1),
                    "numerical_block_shape": "26x26",
                }
            )
    pd.DataFrame(block_rows).to_csv(
        DERIVED_DIRECTORY / "numi_four_channel_covariance_block_map.csv", index=False
    )

    pd.DataFrame(
        {"true_bin": np.arange(60, dtype=int), "true_energy_GeV": true_energy}
    ).to_csv(KERNEL_DIRECTORY / "true_energy_GeV.csv", index=False, float_format="%.17g")
    pd.DataFrame({
        "local_numi_reco_bin": np.arange(104, dtype=int),
        "fixed_published_background_counts": inputs.published_background_counts,
    }).to_csv(
        KERNEL_DIRECTORY / "fixed_published_background_counts.csv",
        index=False,
        float_format="%.17g",
    )
    for name, values in process_arrays.items():
        _write_matrix(KERNEL_DIRECTORY / f"{name}.csv", values, "local_numi_reco_bin")

    closure_columns = [
        "channel",
        "released_channel_ordinal",
        "channel_reco_bin",
        "local_numi_reco_bin",
        "published_global_bin",
        "observed_counts",
        "published_background_counts",
        "published_signal_counts",
        "published_total_prediction_counts",
        "empirical_reference_total_counts",
        "empirical_reference_closure_residual_counts",
    ]
    pd.DataFrame(rows)[closure_columns].to_csv(
        KERNEL_DIRECTORY / "reference_closure.csv", index=False, float_format="%.17g"
    )

    metadata = {
        "status": "diagnostic_event_reweighting_only",
        "format": "numi_four_channel_empirical_kernel_v1",
        "selected_released_channel_ordinals_one_based": [8, 9, 10, 11],
        "selected_published_global_bins_inclusive": [182, 285],
        "channel_order": [channel.identifier for channel in NUMI_FOUR_CHANNELS],
        "covariance": {
            "numerical_shape": [104, 104],
            "logical_shape": "4x4 channel blocks, each block 26x26 reconstructed bins",
            "source": str(DEFAULT_COVARIANCE_PATH),
            "source_sha256": _sha256(DEFAULT_COVARIANCE_PATH),
        },
        "spectrum": {
            "source": str(DEFAULT_SPECTRUM_PATH),
            "source_sha256": _sha256(DEFAULT_SPECTRUM_PATH),
            "published_signal_definition": "Signal + Background minus Background",
        },
        "response": {
            "source_directory": str(BNB_RESPONSE_DIRECTORY),
            "shape_per_channel": [26, 60],
            "true_energy_range_GeV": [0.0, 3.0],
            "true_energy_bin_width_GeV": 0.05,
            "reuse_policy": "read-only reuse; no BNB file was modified",
        },
        "flux": {
            "source_directory": str(NUMI_FLUX_DIRECTORY),
            "source_sha256": flux_hashes,
            "adaptation": "each 0.1 GeV per-bin value is split equally into its two 0.05 GeV Reco-grid bins",
            "used_energy_range_GeV": [0.0, 3.0],
            "unused_flux_range_GeV": [3.0, 5.0],
        },
        "reference": {
            "baseline_km": NUMI_BASELINE_KM,
            "parameters": {
                "delta_m2_41_eV2": DELTA_M2_41_EV2,
                "sin2_theta14": 0.0,
                "sin2_theta24": 0.0,
            },
            "construction": "same per-reconstructed-bin reference-ratio algebra used by BNB",
            "closure": "exact against the selected HEPData Signal + Background vector by construction",
        },
        "paper_parameter_predictions": parameter_records,
        "scientific_limits": [
            "the HEPData total is an empirical anchor and is not asserted to be a published 3nu/null spectrum",
            "the same normalized 2022 BNB response is assumed for NuMI because no NuMI-specific response is available",
            "the response has no true-energy support above 3 GeV, so 3--5 GeV flux does not enter event reweighting",
            "unknown cross section and efficiency are absorbed into one empirical scale per reconstructed bin",
            "the aggregate published Background is frozen because component-level oscillatable templates are unavailable",
            "this artifact does not define a NuMI chi-square or a combined BNB+NuMI likelihood",
        ],
    }
    (KERNEL_DIRECTORY / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    covariance_metadata = {
        "status": "diagnostic_prepared_input",
        "matrix_shape": [104, 104],
        "logical_shape": "4x4 channel blocks, each block 26x26 reconstructed bins",
        "channel_order": [channel.identifier for channel in NUMI_FOUR_CHANNELS],
        "selected_published_global_bins_inclusive": [182, 285],
        "source": str(DEFAULT_COVARIANCE_PATH),
        "source_sha256": _sha256(DEFAULT_COVARIANCE_PATH),
        "likelihood_status": "disabled",
    }
    (DERIVED_DIRECTORY / "numi_four_channel_systematic_covariance.metadata.json").write_text(
        json.dumps(covariance_metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    event_metadata = {
        "status": "regenerable_spectrum_output",
        "source_kernel": str(KERNEL_DIRECTORY),
        "source_covariance": str(DERIVED_DIRECTORY / "numi_four_channel_systematic_covariance.csv"),
        "parameter_points": parameter_records,
        "event_bins": 104,
        "likelihood_or_chi2_included": False,
        "scientific_limits": metadata["scientific_limits"],
    }
    EVENT_OUTPUT.with_suffix(".metadata.json").write_text(
        json.dumps(event_metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": "pass",
                "event_bins": 104,
                "covariance_shape": [104, 104],
                "maximum_reference_closure_residual": float(np.max(np.abs(reference_residual))),
                "kernel_directory": str(KERNEL_DIRECTORY),
                "event_output": str(EVENT_OUTPUT),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
