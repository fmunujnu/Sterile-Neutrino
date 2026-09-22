"""Locate which public-data approximation layer creates the NuMI mass spikes.

This is a read-only diagnostic of the active four-channel NuMI approximation.
It never changes the production predictor, likelihood, configuration, or data.
The comparison is performed on one *fixed* (s14, s24) slice of the official
NuMI-only grid, so profile branch switching cannot explain any roughness.
"""

from __future__ import annotations

from dataclasses import fields, replace
import json
import mmap
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.linalg import cho_factor, cho_solve
import yaml


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from sterile_fit.core.three_plus_one import ThreePlusOneParameters  # noqa: E402
from sterile_fit.experiments.microboone.numi import (  # noqa: E402
    NumiEnergyBaselinePredictor,
    NumiFourChannelEmpiricalKernel,
    PROCESS_FIELDS,
    build_diagnostic_numi_workflow,
    build_energy_baseline_numi_workflow,
)


CONFIG = ROOT / "configs/experiments/microboone/numi/analysis.yaml"
GRID = ROOT / "data/experiments/microboone/shared/microboone_material_gridscan_numi_dm2_t14_t24_dchi2.txt"
EL_INPUT = ROOT / "data/experiments/microboone/numi/derived/public_dk2nu_energy_baseline/psi_exposure_weighted_four_flavours.csv"
OUTPUT = ROOT / "outputs/checks/microboone/numi_roughness_layer_audit"

# This exact public-grid slice was selected because both axes are away from
# boundaries and its 3--6 eV^2 segment visibly separates official/local shape.
S14_INDEX = 2000
S24_INDEX = 50


def quadratic(residual: np.ndarray, covariance: np.ndarray) -> float:
    factor = cho_factor(covariance, lower=True, check_finite=False)
    return float(residual @ cho_solve(factor, residual, check_finite=False))


def official_slice() -> pd.DataFrame:
    rows: list[tuple[float, float, float, float]] = []
    with GRID.open("rb") as handle:
        document = mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ)
        cursor = 0
        for mass_index in range(1, 101):
            marker = f"\n{mass_index}  {S14_INDEX}    {S24_INDEX}   ".encode("ascii")
            start = document.find(marker, cursor)
            if start < 0:
                raise ValueError(f"official fixed slice misses mass index {mass_index}")
            start += 1
            stop = document.find(b"\n", start)
            values = document[start:stop].decode("ascii").split()
            if len(values) != 7:
                raise ValueError(f"unexpected official row: {values}")
            _, i14, i24, mass, s14, s24, delta = values
            if int(i14) != S14_INDEX or int(i24) != S24_INDEX:
                raise ValueError("official slice index mismatch")
            rows.append((float(mass), float(s14), float(s24), float(delta)))
            cursor = stop
        document.close()
    return pd.DataFrame(rows, columns=[
        "delta_m2_41_eV2", "sin2_theta14", "sin2_theta24", "official_delta_chi2"
    ])


def smooth_kernel(kernel: NumiFourChannelEmpiricalKernel, sigma_bins: float) -> NumiFourChannelEmpiricalKernel:
    updates = {}
    for name in PROCESS_FIELDS:
        original = np.asarray(getattr(kernel, name), dtype=float)
        smoothed = gaussian_filter1d(original, sigma=sigma_bins, axis=1, mode="nearest")
        original_sum = original.sum(axis=1)
        smoothed_sum = smoothed.sum(axis=1)
        scale = np.divide(original_sum, smoothed_sum, out=np.zeros_like(original_sum), where=smoothed_sum > 0.0)
        updates[name] = smoothed * scale[:, None]
    return replace(kernel, **updates)


def predict_with_true_bin_integration(
    predictor: NumiEnergyBaselinePredictor,
    parameters: ThreePlusOneParameters,
    quadrature_points: int,
) -> np.ndarray:
    """Average P across every 50 MeV kernel bin without changing its row sum.

    The public q(L|E,nu) row assigned to the bin is kept fixed.  Therefore this
    test isolates centre-sampling of the oscillation phase; it does not claim to
    reconstruct the unknown within-bin event density.
    """
    kernel = predictor.kernel
    distribution = predictor.distribution
    centers = kernel.true_energy_GeV
    half_width = 0.025
    nodes, weights = np.polynomial.legendre.leggauss(quadrature_points)
    energies = centers[:, None] + half_width * nodes[None, :]
    indices = np.searchsorted(distribution.energy_high_GeV, centers, side="left")
    ue4_sq = parameters.sin2_theta14
    umu4_sq = (1.0 - parameters.sin2_theta14) * parameters.sin2_theta24
    amplitudes = {
        "nue_to_nue": -4.0 * ue4_sq * (1.0 - ue4_sq),
        "numu_to_nue": 4.0 * umu4_sq * ue4_sq,
        "nue_to_numu": 4.0 * ue4_sq * umu4_sq,
        "numu_to_numu": -4.0 * umu4_sq * (1.0 - umu4_sq),
        "nuebar_to_nuebar": -4.0 * ue4_sq * (1.0 - ue4_sq),
        "numubar_to_nuebar": 4.0 * umu4_sq * ue4_sq,
        "nuebar_to_numubar": 4.0 * ue4_sq * umu4_sq,
        "numubar_to_numubar": -4.0 * umu4_sq * (1.0 - umu4_sq),
    }
    sources = {
        "nue_to_nue": "nue", "numu_to_nue": "numu",
        "nue_to_numu": "nue", "numu_to_numu": "numu",
        "nuebar_to_nuebar": "nuebar", "numubar_to_nuebar": "numubar",
        "nuebar_to_numubar": "nuebar", "numubar_to_numubar": "numubar",
    }
    survival = {"nue_to_nue", "numu_to_numu", "nuebar_to_nuebar", "numubar_to_numubar"}
    prediction = kernel.fixed_published_background_counts.copy()
    for field in PROCESS_FIELDS:
        process = field.removeprefix("beam_").removesuffix("_cc_response_counts")
        q = distribution.probability_mass_by_flavour[sources[process]][indices]
        phase = (
            1.267 * parameters.delta_m2_41_eV2
            * distribution.baseline_center_km[None, None, :] / energies[:, :, None]
        )
        # Legendre weights integrate on [-1,1]; division by two gives a bin average.
        sin2_l_average = np.sum(q[:, None, :] * np.sin(phase) ** 2, axis=2)
        sin2_el_average = np.sum(sin2_l_average * weights[None, :], axis=1) / 2.0
        probability = amplitudes[process] * sin2_el_average
        if process in survival:
            probability = 1.0 + probability
        prediction += np.asarray(getattr(kernel, field)) @ probability
    return prediction


def covariance_for_variant(reference: np.ndarray, systematic: np.ndarray, prediction: np.ndarray, variant: str) -> np.ndarray:
    ratio = prediction / reference
    if variant == "fixed_total":
        return systematic + np.diag(reference)
    if variant == "dynamic_pearson_only":
        return systematic + np.diag(prediction)
    if variant == "dynamic_systematic_only":
        return systematic * ratio[:, None] * ratio[None, :] + np.diag(reference)
    if variant == "current_dynamic":
        return systematic * ratio[:, None] * ratio[None, :] + np.diag(prediction)
    raise ValueError(variant)


def metrics(name: str, values: np.ndarray, official: np.ndarray) -> dict[str, float | str]:
    second = np.diff(values, n=2)
    span = float(np.quantile(values, 0.95) - np.quantile(values, 0.05))
    centered_values = values - np.mean(values)
    centered_official = official - np.mean(official)
    return {
        "variant": name,
        "max_abs_second_difference": float(np.max(np.abs(second))),
        "rms_second_difference": float(np.sqrt(np.mean(second**2))),
        "normalized_rms_second_difference": float(np.sqrt(np.mean(second**2)) / max(span, 1e-15)),
        "total_variation": float(np.sum(np.abs(np.diff(values)))),
        "correlation_with_official": float(np.corrcoef(values, official)[0, 1]),
        "offset_aligned_rmse_to_official": float(np.sqrt(np.mean((centered_values - centered_official) ** 2))),
    }


def main() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    reference_parameters = ThreePlusOneParameters(**{
        key: float(value) for key, value in config["reference_parameters"].items()
    })
    kernel_directory = ROOT / config["diagnostic_four_channel_events"]["kernel_directory"]
    workflow = build_energy_baseline_numi_workflow(kernel_directory, reference_parameters, EL_INPUT)
    fixed_workflow = build_diagnostic_numi_workflow(
        kernel_directory, reference_parameters, float(config["baseline_km"])
    )
    official = official_slice()
    masses = official["delta_m2_41_eV2"].to_numpy(float)
    s14 = float(official["sin2_theta14"].iloc[0])
    s24 = float(official["sin2_theta24"].iloc[0])
    official_values = official["official_delta_chi2"].to_numpy(float)

    observed = workflow.inputs.observed_counts
    reference = workflow.inputs.published_total_prediction_counts
    systematic = workflow.inputs.systematic_covariance
    q_predictor = workflow.predictor
    fixed_predictor = fixed_workflow.predictor
    predictors = {
        "q_current_kernel": q_predictor,
        "fixed_L_current_kernel": fixed_predictor,
    }
    for sigma in (1.0, 2.0, 4.0):
        predictors[f"q_smoothed_kernel_sigma_{sigma:g}"] = NumiEnergyBaselinePredictor(
            smooth_kernel(q_predictor.kernel, sigma), q_predictor.distribution
        )

    predictions: dict[str, np.ndarray] = {}
    for label, predictor in predictors.items():
        predictions[label] = np.stack([
            predictor.predict_total_counts(ThreePlusOneParameters(float(mass), s14, s24))
            for mass in masses
        ])
    for quadrature_points in (4, 16, 64):
        label = f"q_true_bin_integrated_n{quadrature_points}"
        predictions[label] = np.stack([
            predict_with_true_bin_integration(
                q_predictor, ThreePlusOneParameters(float(mass), s14, s24), quadrature_points
            )
            for mass in masses
        ])
    for sigma in (1.0, 2.0, 4.0):
        label = f"q_integrated_n16_smoothed_sigma_{sigma:g}"
        integrated_predictor = NumiEnergyBaselinePredictor(
            smooth_kernel(q_predictor.kernel, sigma), q_predictor.distribution
        )
        predictions[label] = np.stack([
            predict_with_true_bin_integration(
                integrated_predictor, ThreePlusOneParameters(float(mass), s14, s24), 16
            )
            for mass in masses
        ])

    series: dict[str, np.ndarray] = {"official_fixed_slice": official_values}
    current_prediction = predictions["q_current_kernel"]
    for covariance_variant in (
        "fixed_total", "dynamic_pearson_only", "dynamic_systematic_only", "current_dynamic"
    ):
        series[covariance_variant] = np.asarray([
            quadratic(observed - prediction, covariance_for_variant(reference, systematic, prediction, covariance_variant))
            for prediction in current_prediction
        ])
    series["asimov_data_current_dynamic"] = np.asarray([
        quadratic(reference - prediction, covariance_for_variant(reference, systematic, prediction, "current_dynamic"))
        for prediction in current_prediction
    ])
    series["fixed_L_current_dynamic"] = np.asarray([
        quadratic(observed - prediction, covariance_for_variant(reference, systematic, prediction, "current_dynamic"))
        for prediction in predictions["fixed_L_current_kernel"]
    ])
    for sigma in (1.0, 2.0, 4.0):
        key = f"q_smoothed_kernel_sigma_{sigma:g}"
        series[key] = np.asarray([
            quadratic(observed - prediction, covariance_for_variant(reference, systematic, prediction, "current_dynamic"))
            for prediction in predictions[key]
        ])
    for quadrature_points in (4, 16, 64):
        key = f"q_true_bin_integrated_n{quadrature_points}"
        series[key] = np.asarray([
            quadratic(observed - prediction, covariance_for_variant(reference, systematic, prediction, "current_dynamic"))
            for prediction in predictions[key]
        ])
    for sigma in (1.0, 2.0, 4.0):
        key = f"q_integrated_n16_smoothed_sigma_{sigma:g}"
        series[key] = np.asarray([
            quadratic(observed - prediction, covariance_for_variant(reference, systematic, prediction, "current_dynamic"))
            for prediction in predictions[key]
        ])

    # Under a fixed covariance, separate the quadratic signal displacement from
    # its interference with the actual data residual.  Their sum is exactly the
    # fixed-covariance chi2 change relative to the no-oscillation prediction.
    fixed_covariance = systematic + np.diag(reference)
    inverse_residual = cho_solve(
        cho_factor(fixed_covariance, lower=True, check_finite=False),
        observed - reference, check_finite=False,
    )
    quadratic_terms = []
    cross_terms = []
    factor = cho_factor(fixed_covariance, lower=True, check_finite=False)
    for prediction in current_prediction:
        displacement = prediction - reference
        quadratic_terms.append(float(displacement @ cho_solve(factor, displacement, check_finite=False)))
        cross_terms.append(float(-2.0 * displacement @ inverse_residual))
    series["fixed_covariance_quadratic_term"] = np.asarray(quadratic_terms)
    series["fixed_covariance_data_cross_term"] = np.asarray(cross_terms)

    channel_names = ("nue_cc_fc", "nue_cc_pc", "numu_cc_fc", "numu_cc_pc")
    for channel_index, channel_name in enumerate(channel_names):
        selected = np.arange(26 * channel_index, 26 * (channel_index + 1))
        channel_values = []
        for prediction in current_prediction:
            covariance = covariance_for_variant(reference, systematic, prediction, "current_dynamic")
            channel_values.append(quadratic(
                observed[selected] - prediction[selected], covariance[np.ix_(selected, selected)]
            ))
        series[f"channel_{channel_name}"] = np.asarray(channel_values)

    # Additive constants never affect roughness, so align every curve at its minimum.
    aligned = {name: values - np.min(values) for name, values in series.items()}
    output_table = official.copy()
    for name, values in aligned.items():
        output_table[name] = values

    metric_rows = [metrics(name, values, aligned["official_fixed_slice"]) for name, values in aligned.items()]
    metric_table = pd.DataFrame(metric_rows).sort_values("normalized_rms_second_difference")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    output_table.to_csv(OUTPUT / "fixed_slice_layer_series.csv", index=False, float_format="%.17g")
    metric_table.to_csv(OUTPUT / "roughness_metrics.csv", index=False, float_format="%.17g")

    figure, axes = plt.subplots(3, 2, figsize=(13.5, 12.5), sharex=True)
    panels = (
        (axes[0, 0], ["official_fixed_slice", "current_dynamic", "fixed_total", "dynamic_pearson_only", "dynamic_systematic_only"], "Covariance layers"),
        (axes[0, 1], ["official_fixed_slice", "current_dynamic", "asimov_data_current_dynamic"], "Observed residual layer"),
        (axes[1, 0], ["official_fixed_slice", "current_dynamic", "fixed_L_current_dynamic", "q_smoothed_kernel_sigma_1", "q_smoothed_kernel_sigma_2", "q_smoothed_kernel_sigma_4"], "Effective-kernel energy shape"),
        (axes[1, 1], ["official_fixed_slice", "current_dynamic", "q_true_bin_integrated_n16", "q_true_bin_integrated_n64", "q_integrated_n16_smoothed_sigma_2", "q_integrated_n16_smoothed_sigma_4"], "True-energy integration and residual kernel shape"),
        (axes[2, 0], ["fixed_total", "fixed_covariance_quadratic_term", "fixed_covariance_data_cross_term"], "Fixed-covariance exact decomposition"),
        (axes[2, 1], ["official_fixed_slice", "channel_nue_cc_fc", "channel_nue_cc_pc", "channel_numu_cc_fc", "channel_numu_cc_pc"], "Four local channel contributions"),
    )
    for axis, names, title in panels:
        for name in names:
            style = {"linewidth": 2.4} if name == "official_fixed_slice" else {"linewidth": 1.35}
            axis.plot(masses, aligned[name], label=name.replace("_", " "), **style)
        axis.set_xscale("log")
        axis.set_ylabel(r"slice $\chi^2-\min(\chi^2)$")
        axis.set_title(title)
        axis.grid(alpha=0.2)
        axis.legend(fontsize=7)
    axes[2, 0].set_xlabel(r"$\Delta m^2_{41}\;[\mathrm{eV}^2]$")
    axes[2, 1].set_xlabel(r"$\Delta m^2_{41}\;[\mathrm{eV}^2]$")
    figure.suptitle(
        rf"NuMI fixed slice: $\sin^2\theta_{{14}}={s14:.6g}$, $\sin^2\theta_{{24}}={s24:.6g}$"
    )
    figure.tight_layout()
    figure.savefig(OUTPUT / "layer_audit.png", dpi=190)
    plt.close(figure)

    metadata = {
        "purpose": "diagnostic layer isolation; no production-code change",
        "official_grid": str(GRID),
        "fixed_indices": {"sin2_theta14": S14_INDEX, "sin2_theta24": S24_INDEX},
        "fixed_values": {"sin2_theta14": s14, "sin2_theta24": s24},
        "mass_points": int(len(masses)),
        "kernel_smoothing": "Gaussian along true-energy bins, followed by exact preservation of every process-row sum; diagnostic only",
        "true_energy_bin_integration": "Gauss-Legendre average across each 50 MeV bin while holding its q(L|E,nu) row and event-kernel count fixed; diagnostic only",
        "profile_used": False,
    }
    (OUTPUT / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(metric_table.to_string(index=False), flush=True)
    print(OUTPUT, flush=True)


if __name__ == "__main__":
    main()
