"""Reproduce the public BNB four-channel data/prediction panels from HEPData."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from sterile_fit.experiments.microboone.bnb.binning import BNB_FOUR_CHANNELS
from sterile_fit.fitting import prefit_three_plus_one
from sterile_fit.parameters import ThreePlusOneParameters
from sterile_fit.experiments.microboone.bnb.published_inputs import DEFAULT_SPECTRUM_PATH, load_bnb_four_channel_inputs
from sterile_fit.experiments.microboone.bnb.workflow import build_strict_bnb_workflow
from sterile_fit.spectrum_plotting import SpectrumCurve, SpectrumPanel, render_microboone_spectrum_panels


ROOT = Path(__file__).resolve().parents[4]
PAPER_FIGURE1_DELTA_M2_41_EV2 = 1.2
PAPER_FIGURE1_SIN2_2THETA_MUE = 0.003
PAPER_FIGURE1_SIN2_THETA24_VALUES = (0.018, 0.0045)


def _parameters_from_appearance_and_sin2_theta24(
    delta_m2_41_eV2: float,
    sin2_2theta_mue: float,
    sin2_theta24: float,
) -> ThreePlusOneParameters:
    """Use the conventional small-theta14 branch of the paper parameterization."""
    sin2_2theta14 = sin2_2theta_mue / sin2_theta24
    if not 0.0 <= sin2_2theta14 <= 1.0:
        raise ValueError("sin2(2theta_mue)/sin2(theta24) must lie in [0, 1]")
    sin2_theta14 = (1.0 - np.sqrt(1.0 - sin2_2theta14)) / 2.0
    parameters = ThreePlusOneParameters(
        delta_m2_41_eV2=delta_m2_41_eV2,
        sin2_theta14=float(sin2_theta14),
        sin2_theta24=sin2_theta24,
    )
    if not np.isclose(parameters.sin2_2theta_mue_exact, sin2_2theta_mue, rtol=1e-12, atol=1e-15):
        raise RuntimeError("paper Figure 1 parameter conversion failed its exact-amplitude check")
    return parameters


def _reference_setup(path: Path) -> tuple[ThreePlusOneParameters, float]:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    parameters = ThreePlusOneParameters(
        **{name: float(value) for name, value in document["reference_parameters"].items()}
    )
    return parameters, float(document["baseline_km"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "spectra" / "microboone" / "bnb" / "published_four_channels.png",
        help="PNG destination under the unified outputs directory",
    )
    parser.add_argument(
        "--compare-fit-points",
        action="store_true",
        help="overlay the empirical-anchor closure and this BNB-only fit",
    )
    parser.add_argument(
        "--compare-paper-figure1-points",
        action="store_true",
        help="overlay the two 3+1 parameter points shown in paper Figure 1",
    )
    parser.add_argument(
        "--reference-config",
        type=Path,
        default=ROOT / "configs" / "experiments" / "microboone" / "bnb" / "analysis.yaml",
    )
    parser.add_argument(
        "--kernel",
        type=Path,
        default=ROOT / "data" / "experiments" / "microboone" / "bnb" / "reweighting",
    )
    parser.add_argument(
        "--covariance",
        type=Path,
        default=ROOT / "data" / "experiments" / "microboone" / "bnb" / "derived" / "bnb_four_channel_total_covariance.csv",
    )
    arguments = parser.parse_args()
    inputs = load_bnb_four_channel_inputs()
    comparison_predictions: dict[str, np.ndarray] = {}
    comparison_parameters: dict[str, ThreePlusOneParameters] = {}
    comparison_chi2: dict[str, float] = {}
    workflow = None
    if arguments.compare_fit_points or arguments.compare_paper_figure1_points:
        reference, baseline_km = _reference_setup(arguments.reference_config)
        workflow = build_strict_bnb_workflow(
            arguments.kernel,
            arguments.covariance,
            reference,
            baseline_km,
        )
    if arguments.compare_fit_points:
        assert workflow is not None
        objective = lambda parameters: workflow.likelihood.chi2(workflow.predictor.predict_total_counts(parameters))
        bnb_best_fit = min(
            (prefit_three_plus_one(objective, seed=seed) for seed in (42, 137, 314)),
            key=lambda point: point.chi2,
        )
        comparison_parameters = {
            "bnb_only_reweighted_best_fit": bnb_best_fit.parameters,
        }
        comparison_predictions = {
            "zero_anchor_reconstruction": workflow.predictor.predict_total_counts(reference),
            "bnb_only_reweighted_best_fit": workflow.predictor.predict_total_counts(bnb_best_fit.parameters),
        }
        comparison_chi2 = {
            name: float(objective(parameters)) for name, parameters in comparison_parameters.items()
        }
    if arguments.compare_paper_figure1_points:
        assert workflow is not None
        for sin2_theta24 in PAPER_FIGURE1_SIN2_THETA24_VALUES:
            key = f"paper_figure1_sin2_theta24_{sin2_theta24:g}"
            parameters = _parameters_from_appearance_and_sin2_theta24(
                PAPER_FIGURE1_DELTA_M2_41_EV2,
                PAPER_FIGURE1_SIN2_2THETA_MUE,
                sin2_theta24,
            )
            comparison_parameters[key] = parameters
            comparison_predictions[key] = workflow.predictor.predict_total_counts(parameters)
    # The released table has 25 bins from 0 to 2.5 GeV and one overflow bin.
    edges_GeV = np.append(np.arange(0.0, 2.6, 0.1), 3.0)
    panels: list[SpectrumPanel] = []
    audit_rows: list[dict[str, object]] = []
    for channel in BNB_FOUR_CHANNELS:
        start, stop = channel.first_global_bin, channel.stop_global_bin
        observed = inputs.observed_counts[start:stop]
        total = inputs.published_total_prediction_counts[start:stop]
        background = inputs.published_background_counts[start:stop]
        error_up = inputs.observed_statistical_error_up[start:stop]
        error_down = inputs.observed_statistical_error_down[start:stop]
        prediction_systematic_sigma = np.sqrt(np.diag(inputs.systematic_covariance)[start:stop])
        prediction_lower = np.maximum(total - prediction_systematic_sigma, 0.0)
        prediction_upper = total + prediction_systematic_sigma
        curves: list[SpectrumCurve] = []
        if arguments.compare_fit_points:
            curves.append(SpectrumCurve(
                "Empirical zero-mixing anchor (unsupported as paper null)",
                comparison_predictions["zero_anchor_reconstruction"][start:stop],
                "tab:purple", ":", 2.0,
            ))
            curves.append(SpectrumCurve(
                "BNB-only reweighted best fit",
                comparison_predictions["bnb_only_reweighted_best_fit"][start:stop],
                "tab:green", "-.", 1.5,
            ))
        if arguments.compare_paper_figure1_points:
            curves.append(SpectrumCurve(
                r"Paper Fig. 1 point: $\sin^2\theta_{24}=0.018$",
                comparison_predictions["paper_figure1_sin2_theta24_0.018"][start:stop],
                "#00A6C8", "-", 2.0,
            ))
            curves.append(SpectrumCurve(
                r"Paper Fig. 1 point: $\sin^2\theta_{24}=0.0045$",
                comparison_predictions["paper_figure1_sin2_theta24_0.0045"][start:stop],
                "#C24E00", "-", 2.0,
            ))
        panels.append(SpectrumPanel(
            title=channel.identifier,
            energy_edges_GeV=edges_GeV,
            observed_counts=observed,
            observed_error_down=error_down,
            observed_error_up=error_up,
            background_counts=background,
            signal_plus_background_counts=total,
            comparison_curves=tuple(curves),
            prediction_systematic_sigma=prediction_systematic_sigma,
        ))
        for local_bin in range(26):
            row: dict[str, object] = {
                "channel": channel.identifier,
                "channel_reco_bin": local_bin,
                "reco_energy_low_GeV": edges_GeV[local_bin],
                "reco_energy_high_GeV": edges_GeV[local_bin + 1],
                "is_overflow_bin": local_bin == 25,
                "plotted_data_counts": observed[local_bin],
                "plotted_background_counts": background[local_bin],
                "plotted_signal_plus_background_counts": total[local_bin],
                "prediction_systematic_sigma_counts": prediction_systematic_sigma[local_bin],
                "prediction_systematic_lower_counts": prediction_lower[local_bin],
                "prediction_systematic_upper_counts": prediction_upper[local_bin],
                "derived_signal_counts": total[local_bin] - background[local_bin],
            }
            if arguments.compare_fit_points:
                global_bin = start + local_bin
                row.update({
                    "zero_anchor_reconstruction_counts": comparison_predictions[
                        "zero_anchor_reconstruction"
                    ][global_bin],
                    "bnb_only_reweighted_best_fit_counts": comparison_predictions[
                        "bnb_only_reweighted_best_fit"
                    ][global_bin],
                })
            if arguments.compare_paper_figure1_points:
                global_bin = start + local_bin
                row.update({
                    "paper_figure1_sin2_theta24_0p018_counts": comparison_predictions[
                        "paper_figure1_sin2_theta24_0.018"
                    ][global_bin],
                    "paper_figure1_sin2_theta24_0p0045_counts": comparison_predictions[
                        "paper_figure1_sin2_theta24_0.0045"
                    ][global_bin],
                })
            audit_rows.append(row)
    render_microboone_spectrum_panels(
        panels,
        arguments.output,
        title="MicroBooNE BNB: public four-channel input reproduction\n"
        "(last bin is the released overflow bin)",
    )
    audit_path = arguments.output.with_suffix(".csv")
    pd.DataFrame(audit_rows).to_csv(
        audit_path,
        index=False,
        float_format="%.17g",
    )
    spectrum_source = DEFAULT_SPECTRUM_PATH
    metadata = {
        "plot_semantics": {
            "data": "HEPData Data",
            "background": "HEPData Background, plotted separately",
            "signal_plus_background": "HEPData Signal + Background, used directly; Background is not added again",
            "prediction_uncertainty_band": "plus/minus sqrt of the released systematic covariance diagonal; correlations are used by chi2 but cannot be represented by independent per-bin bars",
            "data_error_bars": "released asymmetric data statistical errors only",
            "derived_signal": "Signal + Background minus Background; audit table only",
        },
        "binning": "25 bins of width 0.1 GeV from 0 to 2.5 GeV, followed by the released overflow bin displayed to 3.0 GeV",
        "spectrum_source": str(spectrum_source),
        "spectrum_source_sha256": sha256(spectrum_source.read_bytes()).hexdigest(),
        "audit_table": str(audit_path),
        "audit_table_sha256": sha256(audit_path.read_bytes()).hexdigest(),
        "fit_point_comparison": None,
        "paper_figure1_comparison": None,
    }
    if arguments.compare_fit_points:
        closure = comparison_predictions["zero_anchor_reconstruction"] - inputs.published_total_prediction_counts
        metadata["fit_point_comparison"] = {
            "zero_anchor_closure_max_absolute_counts": float(np.max(np.abs(closure))),
            "zero_anchor_closure_expected": "must agree algebraically because the HEPData unconstrained total was imposed as the anchor",
            "zero_anchor_scientific_status": "unsupported as the paper null prediction; invalid for a strict paper-null reproduction",
            "bnb_only_fit_parameters_and_chi2": {
                name: {
                    "delta_m2_41_eV2": parameters.delta_m2_41_eV2,
                    "sin2_theta14": parameters.sin2_theta14,
                    "sin2_theta24": parameters.sin2_theta24,
                    "sin2_2theta_mue_exact": parameters.sin2_2theta_mue_exact,
                    "bnb_only_chi2": comparison_chi2[name],
                }
                for name, parameters in comparison_parameters.items()
                if name == "bnb_only_reweighted_best_fit"
            },
            "interpretation": "The anchor closure is exact by construction and is not an independent physics validation. The local best fit minimizes this four-channel BNB-only empirical-reweighting likelihood. No NuMI-derived fit coordinate is plotted.",
        }
    if arguments.compare_paper_figure1_points:
        metadata["paper_figure1_comparison"] = {
            "source_coordinates": {
                "delta_m2_41_eV2": PAPER_FIGURE1_DELTA_M2_41_EV2,
                "sin2_2theta_mue": PAPER_FIGURE1_SIN2_2THETA_MUE,
                "sin2_theta24_values": list(PAPER_FIGURE1_SIN2_THETA24_VALUES),
            },
            "converted_parameters": {
                name: {
                    "delta_m2_41_eV2": parameters.delta_m2_41_eV2,
                    "sin2_theta14": parameters.sin2_theta14,
                    "sin2_theta24": parameters.sin2_theta24,
                    "sin2_2theta_mue_exact": parameters.sin2_2theta_mue_exact,
                }
                for name, parameters in comparison_parameters.items()
                if name.startswith("paper_figure1_")
            },
            "comparison_scope": "Only nue_cc_fc is a direct comparison with the paper's BNB panel in Figure 1; the other BNB channels are local-model diagnostics.",
            "prediction_provenance": "The coordinates are published; the curves are produced by this repository's empirical BNB kernel and are not digitized collaboration curves.",
        }
    arguments.output.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
