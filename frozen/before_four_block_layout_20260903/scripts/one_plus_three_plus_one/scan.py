"""Parallel 1+3+1 mass-pair profile scan using the existing likelihood stack."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sterile_fit.analysis.selection import load_analysis_selection
from sterile_fit.covariance import solve_quadratic_form
from sterile_fit.one_plus_three_plus_one import (
    OnePlusThreePlusOneParameters,
    build_one_plus_three_plus_one_analysis,
    profile_at_fixed_mass_pair,
)
from sterile_fit.statistics import (
    GaussianHypothesis,
    asymptotic_cls,
    prepare_fixed_hypothesis_chi2,
    toy_cls,
)


ROOT = Path(__file__).resolve().parents[2]


def _csv_values(value: str) -> list[float]:
    values = [float(item) for item in value.split(",")]
    if len(values) < 2 or any(not np.isfinite(item) or item <= 0.0 for item in values):
        raise argparse.ArgumentTypeError(
            "provide at least two positive finite comma-separated coordinates"
        )
    if any(right <= left for left, right in zip(values[:-1], values[1:], strict=True)):
        raise argparse.ArgumentTypeError("scan coordinates must be strictly increasing")
    return values


def _log_cell_edges(values: np.ndarray) -> np.ndarray:
    coordinates = np.asarray(values, dtype=float)
    edges = np.empty(coordinates.size + 1, dtype=float)
    edges[1:-1] = np.sqrt(coordinates[:-1] * coordinates[1:])
    edges[0] = coordinates[0] ** 2 / edges[1]
    edges[-1] = coordinates[-1] ** 2 / edges[-2]
    return edges


def _format_duration(seconds: float) -> str:
    value = max(0.0, float(seconds))
    if value < 60.0:
        return f"{value:.1f}s"
    minutes, remaining = divmod(value, 60.0)
    if minutes < 60.0:
        return f"{int(minutes)}m {remaining:04.1f}s"
    hours, minutes = divmod(int(minutes), 60)
    return f"{hours}h {minutes:02d}m"


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
    return tuple(pairs)


def _objective_for_toy(analysis, toy_dataset):
    if len(toy_dataset) != len(analysis.experiments):
        raise ValueError("toy dataset does not match the selected analysis")

    def objective(parameters: OnePlusThreePlusOneParameters) -> float:
        total = 0.0
        for experiment, observation in zip(
            analysis.experiments, toy_dataset, strict=True
        ):
            prediction = experiment.predict_counts(parameters)
            covariance = experiment.covariance_for_prediction(prediction)
            total += solve_quadratic_form(observation - prediction, covariance)
        return float(total)

    return objective


def _same_hypotheses(pairs) -> bool:
    return all(
        np.allclose(null.mean, tested.mean, rtol=1e-12, atol=1e-12)
        and np.allclose(null.covariance, tested.covariance, rtol=1e-12, atol=1e-12)
        for null, tested in pairs
    )


def _plot_result(table: pd.DataFrame, output: Path) -> None:
    q41 = np.sort(table["delta_m2_41_absolute_eV2"].unique())
    q51 = np.sort(table["delta_m2_51_eV2"].unique())
    surface = (
        table.pivot(
            index="delta_m2_51_eV2",
            columns="delta_m2_41_absolute_eV2",
            values="cls",
        )
        .reindex(index=q51, columns=q41)
        .to_numpy(dtype=float)
    )
    figure, axis = plt.subplots(figsize=(7.2, 5.8), constrained_layout=True)
    image = axis.pcolormesh(
        _log_cell_edges(q41),
        _log_cell_edges(q51),
        surface,
        shading="flat",
        cmap="viridis_r",
        vmin=0.0,
        vmax=1.0,
    )
    if np.nanmin(surface) <= 0.05 <= np.nanmax(surface):
        axis.contour(q41, q51, surface, levels=[0.05], colors=["red"], linewidths=2.0)
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel(r"$|\Delta m^2_{41}|\;[\mathrm{eV}^2]$")
    axis.set_ylabel(r"$\Delta m^2_{51}\;[\mathrm{eV}^2]$")
    axis.set_title("1+3+1 mixing-profile diagnostic")
    colorbar = figure.colorbar(image, ax=axis)
    colorbar.set_label(r"$CL_s$")
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    started = perf_counter()
    parser = argparse.ArgumentParser(
        description=(
            "Profile the five e/mu mixing and CP coordinates at each 1+3+1 mass pair. "
            "The detector inputs and covariance are selected through the unchanged 3+1 registry."
        )
    )
    parser.add_argument(
        "--analysis-config",
        type=Path,
        default=ROOT / "configs" / "analyses" / "microboone_bnb.yaml",
    )
    parser.add_argument(
        "--delta-m2-41-absolute-grid-eV2",
        type=_csv_values,
        default=np.geomspace(1e-2, 1e2, 7).tolist(),
    )
    parser.add_argument(
        "--delta-m2-51-grid-eV2",
        type=_csv_values,
        default=np.geomspace(1e-2, 1e2, 7).tolist(),
    )
    parser.add_argument("--cls-calibration", choices=("analytic", "toy"), default="analytic")
    parser.add_argument("--number-of-toys", type=int, default=100)
    parser.add_argument("--toy-workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--profile-maxiter", type=int, default=400)
    parser.add_argument("--profile-popsize", type=int, default=15)
    parser.add_argument("--profile-tolerance", type=float, default=1e-8)
    parser.add_argument("--no-polish", action="store_true")
    parser.add_argument("--kernel", type=Path, help="optional BNB kernel override")
    parser.add_argument("--covariance", type=Path, help="optional BNB covariance override")
    parser.add_argument("--output-directory", type=Path)
    arguments = parser.parse_args()

    if arguments.number_of_toys < 2:
        parser.error("--number-of-toys must be at least 2")
    if arguments.toy_workers < 1:
        parser.error("--toy-workers must be at least 1")

    selection = load_analysis_selection(
        arguments.analysis_config.resolve(), repository_root=ROOT
    )
    overrides = {
        name: path.resolve()
        for name, path in {
            "kernel": arguments.kernel,
            "covariance": arguments.covariance,
        }.items()
        if path is not None
    }
    analysis = build_one_plus_three_plus_one_analysis(
        selection,
        repository_root=ROOT,
        bnb_overrides=overrides or None,
    )
    output_directory = arguments.output_directory
    if output_directory is None:
        run_group = "run2_toy_mc" if arguments.cls_calibration == "toy" else "run1_non_toy"
        output_directory = (
            ROOT
            / "outputs"
            / run_group
            / "one_plus_three_plus_one"
            / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        )
    output_directory = output_directory.resolve()
    output_directory.mkdir(parents=True, exist_ok=True)

    null_parameters = OnePlusThreePlusOneParameters.three_neutrino_null()
    null_chi2 = analysis.objective.chi2(null_parameters)
    q41_values = np.asarray(arguments.delta_m2_41_absolute_grid_eV2, dtype=float)
    q51_values = np.asarray(arguments.delta_m2_51_grid_eV2, dtype=float)
    total_points = q41_values.size * q51_values.size
    rows: list[dict[str, object]] = []

    print(
        f"Starting {arguments.cls_calibration} 1+3+1 scan: {total_points} mass pairs; "
        "all four heavy-state moduli and phi_mue are profiled."
    )
    for point_index, (q41, q51) in enumerate(
        ((left, right) for right in q51_values for left in q41_values), start=1
    ):
        point_started = perf_counter()
        profile = profile_at_fixed_mass_pair(
            analysis.objective.chi2,
            delta_m2_41_absolute_eV2=float(q41),
            delta_m2_51_eV2=float(q51),
            seed=arguments.seed + point_index,
            maxiter=arguments.profile_maxiter,
            popsize=arguments.profile_popsize,
            tolerance=arguments.profile_tolerance,
            polish=not arguments.no_polish,
        )
        tested = profile.best_fit.parameters
        observed_test_statistic = profile.best_fit.chi2 - null_chi2
        pairs = _hypothesis_pairs(analysis, null_parameters, tested)

        if _same_hypotheses(pairs):
            cls_value = p_4nu = p_3nu = 1.0
            calibration_note = "tested spectrum equals the nested 3nu boundary"
            toy_count_4nu = toy_count_3nu = -1
        elif arguments.cls_calibration == "analytic":
            comparison = asymptotic_cls(observed_test_statistic, pairs)
            cls_value = comparison.cls
            p_4nu = comparison.p_value_4nu
            p_3nu = comparison.p_value_3nu
            calibration_note = "Gaussian moment approximation; no pseudo-experiments"
            toy_count_4nu = toy_count_3nu = -1
        else:
            null_hypotheses = tuple(pair[0] for pair in pairs)
            tested_hypotheses = tuple(pair[1] for pair in pairs)
            null_toy_chi2 = prepare_fixed_hypothesis_chi2(null_hypotheses)

            def profiled_test_statistic(toy_dataset):
                toy_objective = _objective_for_toy(analysis, toy_dataset)
                toy_profile = profile_at_fixed_mass_pair(
                    toy_objective,
                    delta_m2_41_absolute_eV2=float(q41),
                    delta_m2_51_eV2=float(q51),
                    seed=arguments.seed + point_index,
                    maxiter=arguments.profile_maxiter,
                    popsize=arguments.profile_popsize,
                    tolerance=arguments.profile_tolerance,
                    polish=not arguments.no_polish,
                )
                return toy_profile.best_fit.chi2 - null_toy_chi2(toy_dataset)

            comparison = toy_cls(
                observed_test_statistic,
                null_hypotheses,
                tested_hypotheses,
                profiled_test_statistic,
                number_of_toys=arguments.number_of_toys,
                seed=arguments.seed + 1_000_000 + point_index,
                workers=arguments.toy_workers,
            )
            cls_value = comparison.cls
            p_4nu = comparison.p_value_4nu
            p_3nu = comparison.p_value_3nu
            toy_count_4nu = comparison.right_tail_count_under_4nu
            toy_count_3nu = comparison.right_tail_count_under_3nu
            calibration_note = "empirical Toy MC with the same five-coordinate profile in every toy"

        row: dict[str, object] = {
            "delta_m2_41_absolute_eV2": q41,
            "delta_m2_41_signed_eV2": -q41,
            "delta_m2_51_eV2": q51,
            "delta_m2_54_eV2": q41 + q51,
            "best_abs_Ue4_squared": tested.abs_Ue4_squared,
            "best_abs_Umu4_squared": tested.abs_Umu4_squared,
            "best_abs_Ue5_squared": tested.abs_Ue5_squared,
            "best_abs_Umu5_squared": tested.abs_Umu5_squared,
            "best_cp_phase_mue_rad": tested.cp_phase_mue_rad,
            "chi2_profiled_1p3p1": profile.best_fit.chi2,
            "chi2_3nu": null_chi2,
            "test_statistic_chi2_1p3p1_minus_chi2_3nu": observed_test_statistic,
            "p_value_1p3p1": p_4nu,
            "p_value_3nu": p_3nu,
            "cls": cls_value,
            "excluded_at_95_percent_cls": bool(cls_value <= 0.05),
            "right_tail_count_under_1p3p1": toy_count_4nu,
            "right_tail_count_under_3nu": toy_count_3nu,
            "optimizer_message": profile.optimizer_message,
            "calibration_note": calibration_note,
        }
        row.update({
            f"chi2__{name.replace('.', '__')}": value
            for name, value in analysis.objective.breakdown(tested).items()
        })
        rows.append(row)

        elapsed = perf_counter() - started
        mean_time = elapsed / point_index
        remaining = mean_time * (total_points - point_index)
        print(
            f"[{point_index}/{total_points}] |dm41|={q41:.4g}, dm51={q51:.4g}, "
            f"CLs={cls_value:.4g}; point {_format_duration(perf_counter() - point_started)}, "
            f"ETA {_format_duration(remaining)}",
            flush=True,
        )

    table = pd.DataFrame(rows)
    result_path = output_directory / "one_plus_three_plus_one_mass_pair_profile.csv"
    table.to_csv(result_path, index=False, float_format="%.17g")
    plot_path = output_directory / "one_plus_three_plus_one_mass_pair_cls.png"
    _plot_result(table, plot_path)
    metadata = {
        "analysis_name": analysis.analysis_name,
        "analysis_config": str(arguments.analysis_config.resolve()),
        "selected_experiments": [item.experiment_id for item in analysis.experiments],
        "cls_calibration": arguments.cls_calibration,
        "number_of_toys_per_hypothesis": (
            arguments.number_of_toys if arguments.cls_calibration == "toy" else 0
        ),
        "profiled_coordinates": [
            "abs_Ue4_squared",
            "abs_Umu4_squared",
            "abs_Ue5_squared",
            "abs_Umu5_squared",
            "cp_phase_mue_rad",
        ],
        "fixed_zero_coordinates": ["theta34", "theta35"],
        "omitted_redundant_coordinate": "theta45",
        "warning": (
            "This mass-only plane profiles to the nested zero-mixing boundary and is a preference/diagnostic "
            "map, not by itself a full 1+3+1 exclusion presentation. Scientifically meaningful exclusion "
            "slices must keep specified mixing coordinates nonzero."
        ),
        "detector_and_covariance_logic": "unchanged existing 3+1 registry inputs",
        "runtime_seconds": perf_counter() - started,
        "outputs": {"table": str(result_path), "plot": str(plot_path)},
    }
    (output_directory / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Finished in {_format_duration(perf_counter() - started)}: {output_directory}")


if __name__ == "__main__":
    main()
