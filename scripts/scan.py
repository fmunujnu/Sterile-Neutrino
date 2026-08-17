"""Single visible entry point for 3+1 prefit and profile scans."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from sterile_fit.analysis.registry import build_three_plus_one_analysis
from sterile_fit.analysis.selection import load_analysis_selection
from sterile_fit.fitting import (
    prefit_three_plus_one,
    profile_appearance_amplitude_grid,
    profile_electron_disappearance_grid,
    profile_grid,
)
from sterile_fit.parameters import ThreePlusOneParameters
from sterile_fit.statistics import GaussianHypothesis, asymptotic_cls


ROOT = Path(__file__).resolve().parents[1]


def _csv_values(value: str) -> list[float]:
    values = [float(item) for item in value.split(",")]
    if not values:
        raise argparse.ArgumentTypeError("provide comma-separated numerical values")
    return values


def _log_cell_edges(values: np.ndarray) -> np.ndarray:
    """Cell edges whose midpoints are geometric on a logarithmic plot."""
    coordinates = np.asarray(values, dtype=float)
    if coordinates.ndim != 1 or coordinates.size < 2:
        raise ValueError("a plotted logarithmic scan axis needs at least two coordinates")
    if np.any(coordinates <= 0.0) or np.any(np.diff(coordinates) <= 0.0):
        raise ValueError("plotted logarithmic scan coordinates must be positive and increasing")
    edges = np.empty(coordinates.size + 1, dtype=float)
    edges[1:-1] = np.sqrt(coordinates[:-1] * coordinates[1:])
    edges[0] = coordinates[0] ** 2 / edges[1]
    edges[-1] = coordinates[-1] ** 2 / edges[-2]
    return edges


def _chi2_breakdown_columns(analysis, parameters) -> dict[str, float]:
    """Keep each selected experiment contribution next to the combined chi2."""
    return {
        f"chi2__{experiment_id.replace('.', '__')}": value
        for experiment_id, value in analysis.objective.breakdown(parameters).items()
    }


def _hypothesis_pairs(analysis, null_parameters, tested_parameters):
    pairs = []
    for experiment in analysis.experiments:
        null_prediction = experiment.predict_counts(null_parameters)
        tested_prediction = experiment.predict_counts(tested_parameters)
        pairs.append((
            GaussianHypothesis(
                null_prediction,
                experiment.covariance_for_prediction(null_prediction),
            ),
            GaussianHypothesis(
                tested_prediction,
                experiment.covariance_for_prediction(tested_prediction),
            ),
        ))
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("prefit", "appearance-profile", "electron-disappearance-profile", "s14-profile"),
        default="appearance-profile",
    )
    parser.add_argument(
        "--analysis-config",
        type=Path,
        default=ROOT / "configs" / "analyses" / "microboone_bnb.yaml",
        help="explicit list of experiment/beam likelihoods to include",
    )
    parser.add_argument("--kernel", type=Path, help="optional MicroBooNE BNB kernel override")
    parser.add_argument("--covariance", type=Path, help="optional MicroBooNE BNB covariance override")
    parser.add_argument("--delta-m2-grid-eV2", type=_csv_values, help="explicit comma-separated override")
    parser.add_argument("--sin2-2theta-mue-grid", type=_csv_values, help="explicit comma-separated override")
    parser.add_argument("--sin2-2theta-ee-grid", type=_csv_values, help="explicit comma-separated override")
    parser.add_argument("--sin2-theta14-grid", type=_csv_values, default=[0.001, 0.01, 0.05])
    parser.add_argument("--grid-points", type=int, default=61, help="points per logarithmic axis when no explicit grid is given")
    parser.add_argument("--delta-m2-min-eV2", type=float, default=1e-2)
    parser.add_argument("--delta-m2-max-eV2", type=float, default=1e2)
    parser.add_argument("--sin2-2theta-mue-min", type=float, default=1e-5)
    parser.add_argument("--sin2-2theta-mue-max", type=float, default=1.0)
    parser.add_argument("--sin2-2theta-ee-min", type=float, default=1e-5)
    parser.add_argument("--sin2-2theta-ee-max", type=float, default=1.0)
    parser.add_argument("--output-directory", type=Path)
    arguments = parser.parse_args()
    selection = load_analysis_selection(arguments.analysis_config, repository_root=ROOT)
    bnb_overrides = {
        name: value
        for name, value in {"kernel": arguments.kernel, "covariance": arguments.covariance}.items()
        if value is not None
    }
    analysis = build_three_plus_one_analysis(
        selection,
        repository_root=ROOT,
        bnb_overrides=bnb_overrides,
    )
    objective = analysis.objective.chi2
    if arguments.grid_points < 8:
        raise ValueError("--grid-points must be at least 8 for a resolved two-dimensional scan")
    delta_m2_grid = arguments.delta_m2_grid_eV2 or np.geomspace(
        arguments.delta_m2_min_eV2,
        arguments.delta_m2_max_eV2,
        arguments.grid_points,
    ).tolist()
    appearance_grid = arguments.sin2_2theta_mue_grid or np.geomspace(
        arguments.sin2_2theta_mue_min,
        arguments.sin2_2theta_mue_max,
        arguments.grid_points,
    ).tolist()
    electron_disappearance_grid = arguments.sin2_2theta_ee_grid or np.geomspace(
        arguments.sin2_2theta_ee_min,
        arguments.sin2_2theta_ee_max,
        arguments.grid_points,
    ).tolist()
    if any(value <= 0.0 for value in delta_m2_grid):
        raise ValueError("delta-m2 grid values must be positive")
    if any(not 0.0 < value <= 1.0 for value in appearance_grid):
        raise ValueError("sin2(2theta_mue) grid values must lie in (0, 1]")
    if any(not 0.0 < value <= 1.0 for value in electron_disappearance_grid):
        raise ValueError("sin2(2theta_ee) grid values must lie in (0, 1]")
    if any(not 0.0 < value <= 1.0 for value in arguments.sin2_theta14_grid):
        raise ValueError("sin2(theta14) grid values must lie in (0, 1]")
    rows: list[dict[str, object]] = []
    row_parameters: list[ThreePlusOneParameters] = []
    global_best = min(
        (prefit_three_plus_one(objective, seed=seed) for seed in (42, 137, 314)),
        key=lambda point: point.chi2,
    )

    if arguments.mode == "prefit":
        point = global_best
        rows.append({
            "delta_m2_41_eV2": point.parameters.delta_m2_41_eV2,
            "sin2_theta14": point.parameters.sin2_theta14,
            "sin2_theta24": point.parameters.sin2_theta24,
            "sin2_2theta_mue_exact": point.parameters.sin2_2theta_mue_exact,
            "chi2": point.chi2,
            **_chi2_breakdown_columns(analysis, point.parameters),
            "optimizer_message": "best of differential-evolution seeds 42, 137, 314",
        })
        row_parameters.append(point.parameters)
    elif arguments.mode == "appearance-profile":
        points = profile_appearance_amplitude_grid(
            objective,
            delta_m2_grid,
            appearance_grid,
        )
        for point in points:
            if not np.isclose(
                point.best_fit.parameters.sin2_2theta_mue_exact,
                point.sin2_2theta_mue,
                rtol=1e-10,
                atol=1e-12,
            ):
                raise RuntimeError("Fig. 3a profile point violates fixed sin2(2theta_mue)")
            if point.best_fit.chi2 < global_best.chi2:
                global_best = point.best_fit
            rows.append({
                "fixed_delta_m2_41_eV2": point.delta_m2_41_eV2,
                "fixed_sin2_2theta_mue": point.sin2_2theta_mue,
                "profiled_sin2_theta14": point.best_fit.parameters.sin2_theta14,
                "derived_sin2_theta24": point.best_fit.parameters.sin2_theta24,
                "chi2": point.best_fit.chi2,
                **_chi2_breakdown_columns(analysis, point.best_fit.parameters),
                "optimizer_message": point.optimizer_message,
            })
            row_parameters.append(point.best_fit.parameters)
    elif arguments.mode == "electron-disappearance-profile":
        points = profile_electron_disappearance_grid(
            objective,
            delta_m2_grid,
            electron_disappearance_grid,
        )
        for point in points:
            if not np.isclose(
                point.best_fit.parameters.sin2_2theta_ee_exact,
                point.sin2_2theta_ee,
                rtol=1e-10,
                atol=1e-12,
            ):
                raise RuntimeError("Fig. 3b profile point violates fixed sin2(2theta_ee)")
            if point.best_fit.chi2 < global_best.chi2:
                global_best = point.best_fit
            rows.append({
                "fixed_delta_m2_41_eV2": point.delta_m2_41_eV2,
                "fixed_sin2_2theta_ee": point.sin2_2theta_ee,
                "selected_sin2_theta14_branch": point.best_fit.parameters.sin2_theta14,
                "profiled_sin2_theta24": point.best_fit.parameters.sin2_theta24,
                "sin2_2theta_mue_exact": point.best_fit.parameters.sin2_2theta_mue_exact,
                "chi2": point.best_fit.chi2,
                **_chi2_breakdown_columns(analysis, point.best_fit.parameters),
                "optimizer_message": point.optimizer_message,
            })
            row_parameters.append(point.best_fit.parameters)
    else:
        points = profile_grid(
            objective,
            {
                "delta_m2_41_eV2": delta_m2_grid,
                "sin2_theta14": arguments.sin2_theta14_grid,
            },
        )
        for point in points:
            if point.best_fit.chi2 < global_best.chi2:
                global_best = point.best_fit
            rows.append({
                "fixed_delta_m2_41_eV2": point.fixed_parameters["delta_m2_41_eV2"],
                "fixed_sin2_theta14": point.fixed_parameters["sin2_theta14"],
                "profiled_sin2_theta24": point.best_fit.parameters.sin2_theta24,
                "sin2_2theta_mue_exact": point.best_fit.parameters.sin2_2theta_mue_exact,
                "chi2": point.best_fit.chi2,
                **_chi2_breakdown_columns(analysis, point.best_fit.parameters),
                "optimizer_message": point.optimizer_message,
            })
            row_parameters.append(point.best_fit.parameters)

    output_directory = arguments.output_directory or (
        ROOT
        / "outputs"
        / "scans"
        / analysis.analysis_name
        / "three_plus_one"
        / datetime.now(UTC).strftime(f"%Y%m%dT%H%M%SZ_{arguments.mode}")
    )
    output_directory.mkdir(parents=True, exist_ok=False)
    result_table = pd.DataFrame(rows)
    result_table["delta_chi2"] = result_table["chi2"] - global_best.chi2
    if (result_table["delta_chi2"] < -1e-9).any():
        raise RuntimeError("negative delta_chi2 indicates an inconsistent global minimum")
    result_table["delta_chi2"] = result_table["delta_chi2"].clip(lower=0.0)
    null_parameters = ThreePlusOneParameters(1.0, 0.0, 0.0)
    chi2_3nu = analysis.objective.chi2(null_parameters)
    if arguments.mode != "prefit":
        cls_rows = []
        for tested_parameters, chi2_4nu in zip(row_parameters, result_table["chi2"], strict=True):
            comparison = asymptotic_cls(
                float(chi2_4nu) - chi2_3nu,
                _hypothesis_pairs(analysis, null_parameters, tested_parameters),
            )
            cls_rows.append(comparison)
        result_table["chi2_3nu"] = chi2_3nu
        result_table["test_statistic_chi2_4nu_minus_chi2_3nu"] = [
            item.test_statistic for item in cls_rows
        ]
        result_table["p_value_4nu_asymptotic"] = [item.p_value_4nu for item in cls_rows]
        result_table["p_value_3nu_asymptotic"] = [item.p_value_3nu for item in cls_rows]
        result_table["cls_asymptotic"] = [item.cls for item in cls_rows]
    result_table.to_csv(output_directory / "result.csv", index=False, float_format="%.17g")
    if arguments.mode != "prefit":
        x_name = {
            "appearance-profile": "fixed_sin2_2theta_mue",
            "electron-disappearance-profile": "fixed_sin2_2theta_ee",
            "s14-profile": "fixed_sin2_theta14",
        }[arguments.mode]
        pivot = result_table.pivot(index="fixed_delta_m2_41_eV2", columns=x_name, values="delta_chi2")
        x_values = pivot.columns.to_numpy(dtype=float)
        y_values = pivot.index.to_numpy(dtype=float)
        x_grid, y_grid = np.meshgrid(x_values, y_values)
        x_edges = _log_cell_edges(x_values)
        y_edges = _log_cell_edges(y_values)
        figure, axis = plt.subplots(figsize=(7.5, 5.8))
        cls_surface = result_table.pivot(
            index="fixed_delta_m2_41_eV2", columns=x_name, values="cls_asymptotic"
        ).to_numpy(dtype=float)
        colour = axis.pcolormesh(
            x_edges,
            y_edges,
            cls_surface,
            shading="flat",
            cmap="viridis_r",
            vmin=0.0,
            vmax=1.0,
        )
        has_cls_95 = float(cls_surface.min()) <= 0.05 <= float(cls_surface.max())
        if has_cls_95 and len(x_values) >= 8 and len(y_values) >= 8:
            axis.contour(
                x_grid,
                y_grid,
                cls_surface,
                levels=[0.05],
                colors="tab:red",
                linewidths=2.0,
            )
            axis.legend(
                handles=[Line2D([0], [0], color="tab:red", linewidth=2.0, label=r"95% $CL_s$ (analytic Gaussian approximation)")],
                loc="best",
                fontsize=8,
            )
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlabel({
            "appearance-profile": r"$\sin^2(2\theta_{\mu e})$",
            "electron-disappearance-profile": r"$\sin^2(2\theta_{ee})$",
            "s14-profile": r"$\sin^2\theta_{14}$",
        }[arguments.mode])
        axis.set_ylabel(r"$\Delta m^2_{41}\;[\mathrm{eV}^2]$")
        analysis_label = (
            "MicroBooNE BNB + NuMI"
            if analysis.analysis_name == "microboone_bnb_numi_joint_diagnostic"
            else analysis.analysis_name.replace("_", " ")
        )
        axis.set_title(rf"$3+1$ {analysis_label}: profiled analytic $CL_s$")
        figure.colorbar(colour, ax=axis, label=r"$CL_s$")
        figure.tight_layout()
        figure.savefig(output_directory / "profile.png", dpi=180, bbox_inches="tight")
        plt.close(figure)
    metadata = {
        "model": "3+1",
        "oscillation_approximation": "short-baseline vacuum limit: m1^2=m2^2=m3^2; only delta_m2_41 retained",
        "mode": arguments.mode,
        "analysis_selection": str(arguments.analysis_config),
        "selected_experiments": [
            {
                "experiment_id": experiment.experiment_id,
                "status": experiment.status,
                "correlation_group": experiment.correlation_group,
                "configuration": str(experiment.configuration),
                **experiment.metadata,
            }
            for experiment in analysis.experiments
        ],
        "reference_anchor_status": "empirical HEPData-unconstrained anchor; unsupported as the paper 3nu/null prediction and invalid for strict paper-null reproduction",
        "best_fit_found": {
            "delta_m2_41_eV2": global_best.parameters.delta_m2_41_eV2,
            "sin2_theta14": global_best.parameters.sin2_theta14,
            "sin2_theta24": global_best.parameters.sin2_theta24,
            "sin2_2theta_mue_exact": global_best.parameters.sin2_2theta_mue_exact,
            "chi2": global_best.chi2,
            "chi2_by_experiment": analysis.objective.breakdown(global_best.parameters),
        },
        "optimizer": {
            "algorithm": "SciPy differential_evolution with polishing; full volume plus explicit s24=0 and s14=0 boundary profiles",
            "prefit_seeds": [42, 137, 314],
            "appearance_profile_method": "deterministic 33-point basin search in sin2_theta14 plus bounded polishing of every sampled local basin",
            "delta_m2_41_eV2_bounds": [0.01, 100.0],
            "sin2_theta14_bounds": [0.0, 1.0],
            "sin2_theta24_bounds": [0.0, 1.0],
            "minimum_claim": "lowest point found among the multiseed prefit and all evaluated profile points; not a proof of the mathematical global minimum",
        },
        "scan_axes": {
            "delta_m2_41_eV2": delta_m2_grid,
            "sin2_2theta_mue": appearance_grid if arguments.mode == "appearance-profile" else None,
            "sin2_2theta_ee": electron_disappearance_grid if arguments.mode == "electron-disappearance-profile" else None,
            "sin2_theta14": arguments.sin2_theta14_grid if arguments.mode == "s14-profile" else None,
        },
        "paper_coordinate_definitions": {
            "sin2_2theta_mue": "4*abs(Ue4)^2*abs(Umu4)^2 = 4*s14*(1-s14)*s24",
            "sin2_2theta_ee": "4*abs(Ue4)^2*(1-abs(Ue4)^2) = 4*s14*(1-s14)",
            "s14": "sin^2(theta14)",
            "s24": "sin^2(theta24)",
            "fig3a_profile": "fix delta_m2_41 and exact sin2_2theta_mue; profile the complete allowed s14 curve with derived s24",
            "fig3b_profile": "fix delta_m2_41 and exact sin2_2theta_ee; profile s24 on both physical s14 branches",
        },
        "grid_note": "the default is a resolved logarithmic grid; increase --grid-points for convergence studies",
        "plot_colour_note": "the heatmap and colour bar show analytic CLs from 0 to 1; the red contour is CLs=0.05",
        "statistical_inference": {
            "test_statistic": "chi2_4nu - chi2_3nu",
            "decision": "exclude where CLs = p_4nu / p_3nu <= 0.05",
            "calibration": "analytic moment-matched Gaussian approximation; no Toy MC",
            "tail": "right-tailed under both fixed hypotheses",
            "profile_treatment": "the observed-data profiled 4nu prediction is held fixed while calibrating each point",
            "paper_difference": "the paper obtains both p-values from pseudo-experiments and repeats its full statistical procedure; this interim contour is not toy-calibrated",
            "chi2_3nu": chi2_3nu,
        },
        "contour_note": "the heatmap and red exclusion contour both use analytic CLs; chi2 values remain internal diagnostics and the test-statistic input",
    }
    (output_directory / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(output_directory)


if __name__ == "__main__":
    main()
