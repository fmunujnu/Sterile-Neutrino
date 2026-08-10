"""One-command audit of visible BNB inputs, kernels, covariance and 3+1 physics."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from sterile_fit.models.three_plus_one import ThreePlusOneVacuumModel
from sterile_fit.covariance import load_declared_total_covariance
from sterile_fit.parameters import ThreePlusOneParameters
from sterile_fit.published_inputs import load_bnb_four_channel_inputs
from sterile_fit.workflows import build_strict_bnb_workflow


ROOT = Path(__file__).resolve().parents[1]
REFERENCE_CONFIG = ROOT / "configs" / "bnb_3nu_anchor.yaml"
KERNEL_DIRECTORY = ROOT / "data" / "anchor" / "bnb_reference_reweight_kernel"
COVARIANCE_PATH = ROOT / "data" / "derived" / "bnb_four_channel_total_covariance.csv"
RESPONSE_DIRECTORY = ROOT / "data" / "derived" / "archival_2022_reco_bnb26_given_true"


def _reference_setup() -> tuple[ThreePlusOneParameters, float]:
    document = yaml.safe_load(REFERENCE_CONFIG.read_text(encoding="utf-8"))
    values = document["reference_parameters"]
    return (
        ThreePlusOneParameters(**{name: float(value) for name, value in values.items()}),
        float(document["baseline_km"]),
    )


def main() -> None:
    parameters, baseline_km = _reference_setup()
    inputs = load_bnb_four_channel_inputs()
    workflow = build_strict_bnb_workflow(KERNEL_DIRECTORY, COVARIANCE_PATH, parameters, baseline_km)
    reference_prediction = workflow.predictor.predict_total_counts(parameters)
    if not np.allclose(reference_prediction, inputs.published_total_prediction_counts, rtol=1e-12, atol=1e-10):
        raise AssertionError("reference kernel does not reproduce published Signal + Background")
    # This call also proves that the covariance is finite, positive definite,
    # dimensionally compatible and usable by the Cholesky likelihood.
    reference_chi2 = workflow.likelihood.chi2(reference_prediction)
    if not np.isfinite(reference_chi2):
        raise AssertionError("reference chi2 is not finite")
    stored_reference_covariance = load_declared_total_covariance(COVARIANCE_PATH).covariance
    active_reference_covariance = workflow.likelihood.covariance_for_prediction(reference_prediction)
    if not np.allclose(active_reference_covariance, stored_reference_covariance, rtol=1e-12, atol=1e-10):
        raise AssertionError("active prediction-scaled covariance does not reproduce the declared reference matrix")
    audit_prediction = workflow.predictor.predict_total_counts(ThreePlusOneParameters(1.2, 0.1, 0.1))
    audit_covariance = workflow.likelihood.covariance_for_prediction(audit_prediction)
    if np.allclose(audit_covariance, active_reference_covariance, rtol=1e-10, atol=1e-8):
        raise AssertionError("active covariance remained fixed when the oscillated prediction changed")
    kernel_metadata = json.loads((KERNEL_DIRECTORY / "metadata.json").read_text(encoding="utf-8"))
    if kernel_metadata.get("baseline_km") != baseline_km:
        raise AssertionError("kernel metadata baseline does not match the active config")
    if kernel_metadata.get("reference_parameters", {}) != {
        "delta_m2_41_eV2": parameters.delta_m2_41_eV2,
        "sin2_theta14": parameters.sin2_theta14,
        "sin2_theta24": parameters.sin2_theta24,
        "sin2_2theta_mue_exact": parameters.sin2_2theta_mue_exact,
    }:
        raise AssertionError("kernel metadata reference parameters do not match the active config")
    for channel in ("nue_cc_fc", "nue_cc_pc", "numu_cc_fc", "numu_cc_pc"):
        table = pd.read_csv(RESPONSE_DIRECTORY / f"{channel}_reco_given_true.csv")
        response = table.iloc[:, 1:].to_numpy(dtype=float)
        if response.shape != (26, 60) or not np.all(np.isfinite(response)) or np.any(response < 0.0):
            raise AssertionError(f"{channel} response is not a finite non-negative 26x60 matrix")
        sums = response.sum(axis=0)
        if not np.all(np.isclose(sums, 0.0, atol=1e-12) | np.isclose(sums, 1.0, atol=1e-12)):
            raise AssertionError(f"{channel} response columns are not zero-or-one normalized")
    model = ThreePlusOneVacuumModel(parameters)
    energy = np.array([0.2, 0.7, 1.4], dtype=float)
    for initial in range(4):
        total = sum(model.probability(initial, final, energy, baseline_km=baseline_km) for final in range(4))
        if not np.allclose(total, 1.0, atol=1e-12):
            raise AssertionError(f"3+1 probability is not unitary for initial flavour {initial}")
    print("PASS visible public spectrum: 104 bins")
    print("PASS visible covariance: 104 x 104 CSV with JSON metadata")
    print("PASS visible Reco matrices: 4 x (26 x 60), every true column sums to zero or one")
    print("PASS visible reference kernel: published Signal + Background closes bin by bin")
    print("PASS bookkeeping: HEPData Background is added exactly once (its frozen treatment remains a declared approximation)")
    print("PASS scan covariance: reference matrix closes and changes with the current prediction")
    print("PASS 3+1 probability conservation")
    print(f"PASS declared BNB baseline: {baseline_km:.4f} km")
    print(f"PASS reference chi2 is finite: {reference_chi2:.8g}")


if __name__ == "__main__":
    main()
