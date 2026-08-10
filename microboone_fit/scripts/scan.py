"""Single visible entry point for 3+1 prefit and profile scans."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from sterile_fit.fitting import prefit_three_plus_one, profile_appearance_amplitude_grid, profile_grid
from sterile_fit.parameters import ThreePlusOneParameters
from sterile_fit.workflows import build_strict_bnb_workflow


ROOT = Path(__file__).resolve().parents[1]


def _csv_values(value: str) -> list[float]:
    values = [float(item) for item in value.split(",")]
    if not values:
        raise argparse.ArgumentTypeError("provide comma-separated numerical values")
    return values


def _reference_setup(path: Path) -> tuple[ThreePlusOneParameters, float]:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    values = document["reference_parameters"]
    parameters = ThreePlusOneParameters(**{name: float(value) for name, value in values.items()})
    baseline_km = float(document["baseline_km"])
    if not np.isfinite(baseline_km) or baseline_km <= 0.0:
        raise ValueError("baseline_km must be finite and positive")
    return parameters, baseline_km


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("prefit", "appearance-profile", "s14-profile"), default="appearance-profile")
    parser.add_argument("--reference-config", type=Path, default=ROOT / "configs" / "bnb_3nu_anchor.yaml")
    parser.add_argument("--kernel", type=Path, default=ROOT / "data" / "anchor" / "bnb_reference_reweight_kernel")
    parser.add_argument(
        "--covariance",
        type=Path,
        default=ROOT / "data" / "derived" / "bnb_four_channel_total_covariance.csv",
    )
    parser.add_argument("--delta-m2-grid-eV2", type=_csv_values, help="explicit comma-separated override")
    parser.add_argument("--sin2-2theta-mue-grid", type=_csv_values, help="explicit comma-separated override")
    parser.add_argument("--sin2-theta14-grid", type=_csv_values, default=[0.001, 0.01, 0.05])
    parser.add_argument("--grid-points", type=int, default=61, help="points per logarithmic axis when no explicit grid is given")
    parser.add_argument("--delta-m2-min-eV2", type=float, default=1e-2)
    parser.add_argument("--delta-m2-max-eV2", type=float, default=1e2)
    parser.add_argument("--sin2-2theta-mue-min", type=float, default=1e-5)
    parser.add_argument("--sin2-2theta-mue-max", type=float, default=1.0)
    parser.add_argument("--output-directory", type=Path)
    arguments = parser.parse_args()
    reference, baseline_km = _reference_setup(arguments.reference_config)
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
    if any(value <= 0.0 for value in delta_m2_grid):
        raise ValueError("delta-m2 grid values must be positive")
    if any(not 0.0 < value <= 1.0 for value in appearance_grid):
        raise ValueError("sin2(2theta_mue) grid values must lie in (0, 1]")
    if any(not 0.0 < value <= 1.0 for value in arguments.sin2_theta14_grid):
        raise ValueError("sin2(theta14) grid values must lie in (0, 1]")
    workflow = build_strict_bnb_workflow(arguments.kernel, arguments.covariance, reference, baseline_km)
    objective = lambda parameters: workflow.likelihood.chi2(workflow.predictor.predict_total_counts(parameters))
    rows: list[dict[str, object]] = []
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
            "optimizer_message": "best of differential-evolution seeds 42, 137, 314",
        })
    elif arguments.mode == "appearance-profile":
        points = profile_appearance_amplitude_grid(
            objective,
            delta_m2_grid,
            appearance_grid,
        )
        for point in points:
            if point.best_fit.chi2 < global_best.chi2:
                global_best = point.best_fit
            rows.append({
                "fixed_delta_m2_41_eV2": point.delta_m2_41_eV2,
                "fixed_sin2_2theta_mue": point.sin2_2theta_mue,
                "profiled_sin2_theta14": point.best_fit.parameters.sin2_theta14,
                "derived_sin2_theta24": point.best_fit.parameters.sin2_theta24,
                "chi2": point.best_fit.chi2,
                "optimizer_message": point.optimizer_message,
            })
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
                "optimizer_message": point.optimizer_message,
            })

    output_directory = arguments.output_directory or (
        ROOT / "runs" / datetime.now(UTC).strftime(f"%Y%m%dT%H%M%SZ_{arguments.mode}")
    )
    output_directory.mkdir(parents=True, exist_ok=False)
    result_table = pd.DataFrame(rows)
    result_table["delta_chi2"] = result_table["chi2"] - global_best.chi2
    if (result_table["delta_chi2"] < -1e-9).any():
        raise RuntimeError("negative delta_chi2 indicates an inconsistent global minimum")
    result_table["delta_chi2"] = result_table["delta_chi2"].clip(lower=0.0)
    result_table.to_csv(output_directory / "result.csv", index=False, float_format="%.17g")
    if arguments.mode != "prefit":
        x_name = "fixed_sin2_2theta_mue" if arguments.mode == "appearance-profile" else "fixed_sin2_theta14"
        pivot = result_table.pivot(index="fixed_delta_m2_41_eV2", columns=x_name, values="delta_chi2")
        x_values = pivot.columns.to_numpy(dtype=float)
        y_values = pivot.index.to_numpy(dtype=float)
        x_grid, y_grid = np.meshgrid(x_values, y_values)
        x_edges = _log_cell_edges(x_values)
        y_edges = _log_cell_edges(y_values)
        figure, axis = plt.subplots(figsize=(7.5, 5.8))
        delta_surface = pivot.to_numpy(dtype=float)
        # The physics of interest is the low-delta-chi2 contour region.  Very
        # strongly excluded points can reach thousands and would otherwise
        # compress every relevant colour into a nearly uniform dark patch.
        # This cap changes only the raster colour scale, never CSV values or
        # contour construction.
        colour_limit = min(25.0, max(9.21, float(delta_surface.max())))
        colour = axis.pcolormesh(
            x_edges,
            y_edges,
            delta_surface,
            shading="flat",
            cmap="viridis",
            vmin=0.0,
            vmax=colour_limit,
        )
        available_levels = [level for level in (2.30, 4.61, 5.99, 9.21) if level <= float(delta_surface.max())]
        # Contour interpolation across the tiny default smoke-test grid is
        # visually misleading.  Only draw contours after the caller provides
        # enough scan coordinates to support a genuine resolved surface.
        if available_levels and len(x_values) >= 8 and len(y_values) >= 8:
            contours = axis.contour(x_grid, y_grid, pivot.to_numpy(dtype=float), levels=available_levels, colors="white")
            axis.clabel(contours, inline=True, fontsize=8, fmt="%.2f")
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlabel("sin2(2theta_mue)" if arguments.mode == "appearance-profile" else "sin2(theta14)")
        axis.set_ylabel("delta_m2_41 [eV2]")
        axis.set_title(f"3+1 BNB {arguments.mode}: profiled delta chi2")
        figure.colorbar(colour, ax=axis, label="delta chi2")
        figure.tight_layout()
        figure.savefig(output_directory / "profile.png", dpi=180, bbox_inches="tight")
        plt.close(figure)
    metadata = {
        "model": "3+1",
        "oscillation_approximation": "short-baseline vacuum limit: m1^2=m2^2=m3^2; only delta_m2_41 retained",
        "mode": arguments.mode,
        "kernel": str(arguments.kernel),
        "covariance": str(arguments.covariance),
        "statistical_treatment": workflow.statistical_treatment,
        "covariance_parameter_dependence": workflow.covariance_parameter_dependence,
        "reference_parameters": {
            "delta_m2_41_eV2": reference.delta_m2_41_eV2,
            "sin2_theta14": reference.sin2_theta14,
            "sin2_theta24": reference.sin2_theta24,
        },
        "baseline_km": baseline_km,
        "best_fit_found": {
            "delta_m2_41_eV2": global_best.parameters.delta_m2_41_eV2,
            "sin2_theta14": global_best.parameters.sin2_theta14,
            "sin2_theta24": global_best.parameters.sin2_theta24,
            "sin2_2theta_mue_exact": global_best.parameters.sin2_2theta_mue_exact,
            "chi2": global_best.chi2,
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
            "sin2_theta14": arguments.sin2_theta14_grid if arguments.mode == "s14-profile" else None,
        },
        "grid_note": "the default is a resolved logarithmic grid; increase --grid-points for convergence studies",
        "plot_colour_note": "heatmap colour maximum is capped at delta_chi2=25 (never below 9.21) to resolve the exclusion region; CSV values and contours use unclipped delta_chi2",
        "contour_note": "delta_chi2 is relative to the lowest fit found in this run; any Wilks contour labels are diagnostic, not the paper CLs construction",
    }
    (output_directory / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(output_directory)


if __name__ == "__main__":
    main()
