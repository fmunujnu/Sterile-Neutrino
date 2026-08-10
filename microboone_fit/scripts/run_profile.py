"""Compute a true 3+1 profile grid with template-validated inputs."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from sterile_fit.fitting import profile_appearance_amplitude_grid, profile_grid
from sterile_fit.parameters import ThreePlusOneParameters
from sterile_fit.workflows import build_strict_bnb_workflow


def _positive_csv(value: str) -> list[float]:
    values = [float(item) for item in value.split(",")]
    if not values or any(item <= 0.0 for item in values):
        raise argparse.ArgumentTypeError("provide one or more positive comma-separated Δm² values")
    return values


def _unit_interval_csv(value: str) -> list[float]:
    values = [float(item) for item in value.split(",")]
    if not values or any(item < 0.0 or item > 1.0 for item in values):
        raise argparse.ArgumentTypeError("provide one or more comma-separated sin²(theta) values in [0, 1]")
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-delta-m2-eV2", type=float, required=True)
    parser.add_argument("--reference-sin2-theta14", type=float, required=True)
    parser.add_argument("--reference-sin2-theta24", type=float, required=True)
    parser.add_argument("--delta-m2-grid-eV2", type=_positive_csv, required=True)
    scan_group = parser.add_mutually_exclusive_group(required=True)
    scan_group.add_argument("--sin2-theta14-grid", type=_unit_interval_csv)
    scan_group.add_argument(
        "--sin2-2theta-mue-grid",
        type=_unit_interval_csv,
        help="exact fixed sin2(2theta_mue) scan; profiles constrained s14 and s24",
    )
    parser.add_argument("--templates", type=Path, required=True)
    parser.add_argument("--total-covariance", type=Path, required=True)
    arguments = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    reference = ThreePlusOneParameters(
        arguments.reference_delta_m2_eV2,
        arguments.reference_sin2_theta14,
        arguments.reference_sin2_theta24,
    )
    workflow = build_strict_bnb_workflow(arguments.templates, arguments.total_covariance, reference)
    objective = lambda parameters: workflow.likelihood.chi2(workflow.predictor.predict_total_counts(parameters))
    if arguments.sin2_2theta_mue_grid is None:
        points = profile_grid(
            objective,
            {
                "delta_m2_41_eV2": arguments.delta_m2_grid_eV2,
                "sin2_theta14": arguments.sin2_theta14_grid,
            },
        )
        scan_coordinates = "delta_m2_41_eV2,sin2_theta14"
        serialized_points = [
            {
                "fixed_parameters": dict(point.fixed_parameters),
                "best_fit": {
                    "delta_m2_41_eV2": point.best_fit.parameters.delta_m2_41_eV2,
                    "sin2_theta14": point.best_fit.parameters.sin2_theta14,
                    "sin2_theta24": point.best_fit.parameters.sin2_theta24,
                    "sin2_2theta_mue_exact": point.best_fit.parameters.sin2_2theta_mue_exact,
                    "chi2": point.best_fit.chi2,
                },
            }
            for point in points
        ]
    else:
        points = profile_appearance_amplitude_grid(
            objective,
            arguments.delta_m2_grid_eV2,
            arguments.sin2_2theta_mue_grid,
        )
        scan_coordinates = "delta_m2_41_eV2,sin2_2theta_mue"
        serialized_points = [
            {
                "fixed_parameters": {
                    "delta_m2_41_eV2": point.delta_m2_41_eV2,
                    "sin2_2theta_mue": point.sin2_2theta_mue,
                },
                "best_fit": {
                    "delta_m2_41_eV2": point.best_fit.parameters.delta_m2_41_eV2,
                    "sin2_theta14": point.best_fit.parameters.sin2_theta14,
                    "sin2_theta24": point.best_fit.parameters.sin2_theta24,
                    "sin2_2theta_mue_exact": point.best_fit.parameters.sin2_2theta_mue_exact,
                    "chi2": point.best_fit.chi2,
                },
            }
            for point in points
        ]
    run_dir = root / "runs" / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ_profile_3plus1")
    run_dir.mkdir(parents=True, exist_ok=False)
    payload = {
        "scope": "MicroBooNE BNB first four channels template-validated 3+1 profile",
        "statistical_treatment": workflow.statistical_treatment,
        "covariance_parameter_dependence": workflow.covariance_parameter_dependence,
        "scan_coordinates": scan_coordinates,
        "profile_points": serialized_points,
    }
    (run_dir / "result.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(run_dir)


if __name__ == "__main__":
    main()
