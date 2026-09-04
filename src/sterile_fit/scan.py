"""scan.py: regrouped existing implementations; see docs/ARCHITECTURE.md."""
from __future__ import annotations

from sterile_fit.output import write_csv, write_json, plot_three_plus_one_scan, result_directory
import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
import json
from pathlib import Path
from time import perf_counter
import numpy as np
import pandas as pd
from sterile_fit.adapter import _hypothesis_pairs, extended_hypothesis_pairs, build_three_plus_one_analysis
from sterile_fit.adapter import load_analysis_selection
from sterile_fit.core.profile_three_plus_one import profile_appearance_amplitude_grid, profile_electron_disappearance_grid, profile_grid, profile_s14_s24_at_fixed_sin2_2theta_ee, profile_s14_s24_at_fixed_sin2_2theta_mue, profile_three_plus_one
from sterile_fit.core.three_plus_one import ThreePlusOneParameters
from sterile_fit.core.calibration import GaussianHypothesis, asymptotic_cls, prepare_fixed_test_statistic, toy_cls
from sterile_fit.core.likelihood import solve_quadratic_form
from sterile_fit.core.one_plus_three_plus_one import OnePlusThreePlusOneParameters
from sterile_fit.adapter import build_one_plus_three_plus_one_analysis
from sterile_fit.core.profile_one_plus_three_plus_one import profile_at_fixed_mass_pair
from sterile_fit.output import plot_one_plus_three_plus_one_scan as extended_plot_result
# Profile-only engines. Global prefit and its delta-chi2 diagnostics are archived.


from sterile_fit.paths import REPOSITORY_ROOT as ROOT
_PROCESS_ANALYSIS = None
_PROCESS_MODE = None


def _initialise_scan_process(analysis_config: str, bnb_overrides: dict[str, str], mode: str) -> None:
    """Construct one read-only analysis instance inside each worker process."""
    global _PROCESS_ANALYSIS, _PROCESS_MODE
    selection = load_analysis_selection(Path(analysis_config), repository_root=ROOT)
    _PROCESS_ANALYSIS = build_three_plus_one_analysis(
        selection,
        repository_root=ROOT,
        bnb_overrides={key: Path(value) for key, value in bnb_overrides.items()},
    )
    _PROCESS_MODE = mode


def _evaluate_toy_point_in_process(payload):
    """Evaluate one seeded scan point using process-local immutable inputs."""
    if _PROCESS_ANALYSIS is None or _PROCESS_MODE is None:
        raise RuntimeError("scan worker process was not initialised")
    point_index, tested_parameters, observed_test_statistic, toy_count, seed, batch_size = payload
    null_parameters = ThreePlusOneParameters(1.0, 0.0, 0.0)
    pairs = _hypothesis_pairs(_PROCESS_ANALYSIS, null_parameters, tested_parameters)
    null_hypotheses = tuple(pair[0] for pair in pairs)
    tested_hypotheses = tuple(pair[1] for pair in pairs)
    fixed_test_statistic = prepare_fixed_test_statistic(null_hypotheses, tested_hypotheses)

    comparison = toy_cls(
        float(observed_test_statistic),
        null_hypotheses,
        tested_hypotheses,
        fixed_test_statistic,
        number_of_toys=toy_count,
        seed=seed,
        workers=1,
        batch_size=batch_size,
    )
    return int(point_index), comparison


def _format_duration(seconds: float) -> str:
    """Format a non-negative runtime without hiding sub-minute progress."""
    value = max(0.0, float(seconds))
    if value < 60.0:
        return f"{value:.1f}s"
    minutes, remaining_seconds = divmod(value, 60.0)
    if minutes < 60.0:
        return f"{int(minutes)}m {remaining_seconds:04.1f}s"
    hours, remaining_minutes = divmod(int(minutes), 60)
    return f"{hours}h {remaining_minutes:02d}m"


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


def _stable_point_seed(base_seed: int, point_index: int) -> int:
    """Derive a platform-stable independent random stream per scan point."""
    sequence = np.random.SeedSequence([base_seed, point_index])
    return int(sequence.generate_state(1, dtype=np.uint64)[0])


def _adaptive_toy_candidate_mask(
    result_table: pd.DataFrame,
    *,
    x_name: str,
    lower_analytic_cls: float,
    upper_analytic_cls: float,
    neighbour_padding: int,
    threshold: float = 0.05,
) -> np.ndarray:
    """Select a wide analytic contour band plus neighbouring x-grid cells."""
    if not 0.0 <= lower_analytic_cls < threshold < upper_analytic_cls <= 1.0:
        raise ValueError("adaptive analytic CLs bounds must bracket 0.05 inside [0, 1]")
    if neighbour_padding < 0:
        raise ValueError("adaptive neighbour padding must be non-negative")
    required = {"fixed_delta_m2_41_eV2", x_name, "cls_asymptotic"}
    missing = required.difference(result_table.columns)
    if missing:
        raise ValueError(f"adaptive candidate table is missing columns: {sorted(missing)}")
    selected = np.zeros(len(result_table), dtype=bool)
    for _, group in result_table.groupby("fixed_delta_m2_41_eV2", sort=False):
        ordered = group.sort_values(x_name)
        indices = ordered.index.to_numpy(dtype=int)
        values = ordered["cls_asymptotic"].to_numpy(dtype=float)
        local = (values >= lower_analytic_cls) & (values <= upper_analytic_cls)
        crossing = np.flatnonzero(
            (values[:-1] - threshold) * (values[1:] - threshold) <= 0.0
        )
        for left in crossing:
            local[left : left + 2] = True
        expanded = local.copy()
        for position in np.flatnonzero(local):
            start = max(0, position - neighbour_padding)
            stop = min(local.size, position + neighbour_padding + 1)
            expanded[start:stop] = True
        selected[indices] = expanded
    return selected


def scan_three_plus_one() -> None:
    run_started_at = perf_counter()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("appearance-profile", "electron-disappearance-profile", "s14-profile"),
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
    parser.add_argument(
        "--cls-calibration",
        choices=("analytic", "toy", "adaptive-toy"),
        default="analytic",
        help="adaptive-toy runs analytic CLs globally and fixed-hypothesis toys only near its 0.05 contour",
    )
    parser.add_argument(
        "--number-of-toys",
        type=int,
        default=100,
        help="pseudo-experiments generated under each of 3nu and tested 4nu at every scan point",
    )
    parser.add_argument("--toy-seed", type=int, default=20250821)
    parser.add_argument(
        "--toy-workers",
        type=int,
        default=1,
        help="parallel fixed-hypothesis Toy evaluations in shared-memory threads; 1 is the deterministic conservative default",
    )
    parser.add_argument(
        "--scan-workers",
        type=int,
        default=1,
        help="parallel independent scan points; use this level for multi-core adaptive scans",
    )
    parser.add_argument(
        "--scan-parallel-backend",
        choices=("threads", "processes"),
        default="threads",
        help="processes bypass Python's interpreter lock for expensive pointwise profiles",
    )
    parser.add_argument(
        "--toy-batch-size",
        type=int,
        default=256,
        help="maximum pseudo-experiments held in memory while retaining the same seeded random stream",
    )
    parser.add_argument(
        "--store-toy-distributions",
        action="store_true",
        help="write every generated test statistic as visible per-point CSV files",
    )
    parser.add_argument("--adaptive-analytic-cls-min", type=float, default=0.01)
    parser.add_argument("--adaptive-analytic-cls-max", type=float, default=0.1)
    parser.add_argument("--adaptive-neighbour-padding", type=int, default=1)
    parser.add_argument(
        "--adaptive-toy-point-limit",
        type=int,
        help="diagnostic cap on selected Toy points; omitted means evaluate the complete selected band",
    )
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
    toy_enabled = arguments.cls_calibration in {"toy", "adaptive-toy"}
    if toy_enabled and arguments.number_of_toys < 2:
        raise ValueError("--number-of-toys must be at least 2 per hypothesis")
    if toy_enabled and arguments.toy_workers < 1:
        raise ValueError("--toy-workers must be at least 1")
    if toy_enabled and arguments.scan_workers < 1:
        raise ValueError("--scan-workers must be at least 1")
    if toy_enabled and arguments.scan_workers > 1 and arguments.toy_workers > 1:
        raise ValueError(
            "nested scan-point and within-point parallelism is disabled; "
            "use --scan-workers N with --toy-workers 1"
        )
    if toy_enabled and arguments.toy_batch_size < 1:
        raise ValueError("--toy-batch-size must be at least 1")
    if arguments.store_toy_distributions and not toy_enabled:
        raise ValueError("--store-toy-distributions requires a Toy CLs calibration")
    if arguments.adaptive_toy_point_limit is not None and arguments.adaptive_toy_point_limit < 0:
        raise ValueError("--adaptive-toy-point-limit must be non-negative")
    if (
        arguments.adaptive_toy_point_limit is not None
        and arguments.cls_calibration != "adaptive-toy"
    ):
        raise ValueError("--adaptive-toy-point-limit requires --cls-calibration adaptive-toy")
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

    if arguments.mode == "appearance-profile":
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

    output_directory = arguments.output_directory or result_directory(
        analysis.analysis_name, "three_plus_one", f"scan_{arguments.mode}_{arguments.cls_calibration}")
    output_directory.mkdir(parents=True, exist_ok=False)
    result_table = pd.DataFrame(rows)
    null_parameters = ThreePlusOneParameters(1.0, 0.0, 0.0)
    chi2_3nu = analysis.objective.chi2(null_parameters)
    result_table["chi2_3nu"] = chi2_3nu
    result_table["test_statistic_chi2_4nu_minus_chi2_3nu"] = (
        result_table["chi2"] - chi2_3nu
    )
    if arguments.cls_calibration in {"analytic", "adaptive-toy"}:
        cls_rows = []
        for tested_parameters, observed_test_statistic in zip(
            row_parameters,
            result_table["test_statistic_chi2_4nu_minus_chi2_3nu"],
            strict=True,
        ):
            cls_rows.append(asymptotic_cls(
                float(observed_test_statistic),
                _hypothesis_pairs(analysis, null_parameters, tested_parameters),
            ))
        result_table["p_value_4nu_asymptotic"] = [item.p_value_4nu for item in cls_rows]
        result_table["p_value_3nu_asymptotic"] = [item.p_value_3nu for item in cls_rows]
        result_table["cls_asymptotic"] = [item.cls for item in cls_rows]
    if arguments.cls_calibration == "analytic":
        cls_column = "cls_asymptotic"
    else:
        if arguments.cls_calibration == "toy":
            candidate_mask = np.ones(len(result_table), dtype=bool)
        else:
            adaptive_x_name = {
                "appearance-profile": "fixed_sin2_2theta_mue",
                "electron-disappearance-profile": "fixed_sin2_2theta_ee",
                "s14-profile": "fixed_sin2_theta14",
            }[arguments.mode]
            candidate_mask = _adaptive_toy_candidate_mask(
                result_table,
                x_name=adaptive_x_name,
                lower_analytic_cls=arguments.adaptive_analytic_cls_min,
                upper_analytic_cls=arguments.adaptive_analytic_cls_max,
                neighbour_padding=arguments.adaptive_neighbour_padding,
            )
            result_table["adaptive_toy_candidate"] = candidate_mask
        candidate_indices = np.flatnonzero(candidate_mask)
        full_candidate_count = int(candidate_indices.size)
        if arguments.cls_calibration == "adaptive-toy" and arguments.adaptive_toy_point_limit is not None:
            limit = min(arguments.adaptive_toy_point_limit, full_candidate_count)
            if limit == 0:
                candidate_indices = candidate_indices[:0]
            elif limit < full_candidate_count:
                sample_positions = np.linspace(
                    0, full_candidate_count - 1, limit, dtype=int
                )
                candidate_indices = candidate_indices[sample_positions]
        if arguments.cls_calibration == "adaptive-toy":
            print(
                f"Adaptive Toy selection: {full_candidate_count}/{len(result_table)} "
                f"candidate points; evaluating {candidate_indices.size}"
            )
        evaluated_mask = np.zeros(len(result_table), dtype=bool)
        evaluated_mask[candidate_indices] = True
        if arguments.cls_calibration == "adaptive-toy":
            result_table["adaptive_toy_evaluated"] = evaluated_mask
        toy_summary_rows: dict[int, dict[str, float | int]] = {}
        raw_distribution_directory = output_directory / "toy_distributions"
        if arguments.store_toy_distributions and candidate_indices.size:
            raw_distribution_directory.mkdir(parents=True, exist_ok=False)

        def evaluate_toy_point(point_index: int):
            tested_parameters = row_parameters[int(point_index)]
            observed_test_statistic = result_table.loc[
                point_index, "test_statistic_chi2_4nu_minus_chi2_3nu"
            ]
            pairs = _hypothesis_pairs(analysis, null_parameters, tested_parameters)
            null_hypotheses = tuple(pair[0] for pair in pairs)
            tested_hypotheses = tuple(pair[1] for pair in pairs)
            fixed_test_statistic = prepare_fixed_test_statistic(null_hypotheses, tested_hypotheses)

            return int(point_index), toy_cls(
                float(observed_test_statistic),
                null_hypotheses,
                tested_hypotheses,
                fixed_test_statistic,
                number_of_toys=arguments.number_of_toys,
                seed=_stable_point_seed(arguments.toy_seed, point_index),
                workers=arguments.toy_workers,
                batch_size=arguments.toy_batch_size,
            )

        toy_stage_started_at = perf_counter()
        pre_toy_elapsed = toy_stage_started_at - run_started_at
        total_selected_points = int(candidate_indices.size)

        def record_toy_result(completed_count: int, point_index: int, comparison) -> None:
            if arguments.store_toy_distributions:
                write_csv(pd.DataFrame({
                    "test_statistic_under_3nu": comparison.test_statistics_under_3nu,
                    "test_statistic_under_4nu": comparison.test_statistics_under_4nu,
                }), 
                    raw_distribution_directory / f"point_{point_index:06d}.csv",
                    index=False,
                    float_format="%.17g",
                )
            toy_elapsed = perf_counter() - toy_stage_started_at
            estimated_toy_total = (
                toy_elapsed * total_selected_points / completed_count
            )
            estimated_overall_total = pre_toy_elapsed + estimated_toy_total
            overall_elapsed = pre_toy_elapsed + toy_elapsed
            remaining = max(0.0, estimated_toy_total - toy_elapsed)
            print(
                f"Toy CLs progress {completed_count}/{total_selected_points} "
                f"({100.0 * completed_count / total_selected_points:.1f}%; "
                f"global index {point_index}; CLs={comparison.cls:.6g}); "
                f"elapsed={_format_duration(overall_elapsed)}; "
                f"estimated total={_format_duration(estimated_overall_total)}; "
                f"remaining={_format_duration(remaining)}",
                flush=True,
            )
            summary = {
                "toy_count_per_hypothesis": comparison.number_of_toys_per_hypothesis,
                "right_tail_count_under_4nu": comparison.right_tail_count_under_4nu,
                "right_tail_count_under_3nu": comparison.right_tail_count_under_3nu,
                "p_value_4nu_toy": comparison.p_value_4nu,
                "p_value_3nu_toy": comparison.p_value_3nu,
                "cls_toy": comparison.cls,
                "p_value_4nu_toy_standard_error": comparison.p_value_4nu_standard_error,
                "p_value_3nu_toy_standard_error": comparison.p_value_3nu_standard_error,
                "cls_toy_standard_error_delta_method": comparison.cls_standard_error_delta_method,
            }
            for hypothesis_name, values in (
                ("3nu", comparison.test_statistics_under_3nu),
                ("4nu", comparison.test_statistics_under_4nu),
            ):
                summary[f"test_statistic_mean_under_{hypothesis_name}"] = float(
                    np.mean(values)
                )
                summary[f"test_statistic_std_under_{hypothesis_name}"] = float(
                    np.std(values, ddof=1)
                )
                for quantile in (0.05, 0.5, 0.95):
                    suffix = str(quantile).replace(".", "p")
                    summary[
                        f"test_statistic_quantile_{suffix}_under_{hypothesis_name}"
                    ] = float(np.quantile(values, quantile))
            toy_summary_rows[int(point_index)] = summary

        if arguments.scan_workers == 1:
            for completed_count, point_index in enumerate(candidate_indices, start=1):
                evaluated_index, comparison = evaluate_toy_point(int(point_index))
                record_toy_result(completed_count, evaluated_index, comparison)
        elif arguments.scan_parallel_backend == "threads":
            print(
                f"Starting {total_selected_points} Toy-calibrated scan points with "
                f"{arguments.scan_workers} point workers; pre-Toy stage took "
                f"{_format_duration(pre_toy_elapsed)}",
                flush=True,
            )
            with ThreadPoolExecutor(max_workers=arguments.scan_workers) as executor:
                futures = [
                    executor.submit(evaluate_toy_point, int(point_index))
                    for point_index in candidate_indices
                ]
                for completed_count, future in enumerate(
                    as_completed(futures), start=1
                ):
                    evaluated_index, comparison = future.result()
                    record_toy_result(completed_count, evaluated_index, comparison)
        else:
            print(
                f"Starting {total_selected_points} Toy-calibrated scan points with "
                f"{arguments.scan_workers} worker processes; pre-Toy stage took "
                f"{_format_duration(pre_toy_elapsed)}",
                flush=True,
            )
            process_overrides = {
                key: str(value) for key, value in bnb_overrides.items()
            }
            payloads = [
                (
                    int(point_index),
                    row_parameters[int(point_index)],
                    float(result_table.loc[
                        point_index, "test_statistic_chi2_4nu_minus_chi2_3nu"
                    ]),
                    arguments.number_of_toys,
                    _stable_point_seed(arguments.toy_seed, int(point_index)),
                    arguments.toy_batch_size,
                )
                for point_index in candidate_indices
            ]
            with ProcessPoolExecutor(
                max_workers=arguments.scan_workers,
                initializer=_initialise_scan_process,
                initargs=(str(arguments.analysis_config), process_overrides, arguments.mode),
            ) as executor:
                futures = [executor.submit(_evaluate_toy_point_in_process, item) for item in payloads]
                for completed_count, future in enumerate(as_completed(futures), start=1):
                    evaluated_index, comparison = future.result()
                    record_toy_result(completed_count, evaluated_index, comparison)
        if toy_summary_rows:
            toy_summary_table = pd.DataFrame.from_dict(toy_summary_rows, orient="index")
            for column in toy_summary_table:
                result_table[column] = np.nan
                result_table.loc[toy_summary_table.index, column] = toy_summary_table[
                    column
                ].to_numpy()
        else:
            result_table["cls_toy"] = np.nan
        if arguments.cls_calibration == "toy":
            if result_table["cls_toy"].isna().any():
                raise RuntimeError("complete Toy calibration left unevaluated scan points")
            cls_column = "cls_toy"
        else:
            result_table["cls_adaptive_hybrid"] = result_table["cls_asymptotic"]
            result_table.loc[evaluated_mask, "cls_adaptive_hybrid"] = result_table.loc[
                evaluated_mask, "cls_toy"
            ]
            cls_column = "cls_adaptive_hybrid"
    write_csv(result_table, output_directory / "result.csv", index=False, float_format="%.17g")
    plot_three_plus_one_scan(result_table, output_directory, arguments, analysis, cls_column)
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
        "optimizer": {
            "algorithm": "unchanged coordinate-specific profile; no standalone global prefit",
            "appearance_profile_method": "deterministic 33-point basin search in sin2_theta14 plus bounded polishing of every sampled local basin",
            "delta_m2_41_eV2_bounds": [0.01, 100.0],
            "sin2_theta14_bounds": [0.0, 1.0],
            "sin2_theta24_bounds": [0.0, 1.0],
            "minimum_claim": "pointwise constrained profile only; no global-minimum claim",
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
        "plot_colour_note": f"the heatmap and colour bar show {arguments.cls_calibration} CLs from 0 to 1; the red contour is CLs=0.05",
        "statistical_inference": {
            "test_statistic": "chi2_4nu - chi2_3nu",
            "decision": "exclude where CLs = p_4nu / p_3nu <= 0.05",
            "calibration": (
                "empirical Toy MC under both hypotheses; no assumed test-statistic distribution"
                if arguments.cls_calibration == "toy"
                else (
                    "global analytic CLs with fixed-hypothesis Toy MC only in the declared adaptive contour band"
                    if arguments.cls_calibration == "adaptive-toy"
                    else "analytic moment-matched Gaussian approximation; no Toy MC"
                )
            ),
            "tail": "right-tailed under both fixed hypotheses",
            "profile_treatment": (
                "profile the observed data once at each scan point, then hold the resulting 3nu and 4nu predictions and covariances fixed for every pseudo-experiment"
                if toy_enabled
                else "the observed-data profiled 4nu prediction is held fixed while calibrating each point"
            ),
            "toy_generator_nuisance_treatment": (
                "plug-in at the observed-data profiled 4nu point"
                if toy_enabled
                else None
            ),
            "toy_fluctuation_model": (
                "multivariate Gaussian using the complete prediction-dependent covariance; no clipping and no extra Poisson draw"
                if toy_enabled
                else None
            ),
            "toy_count_per_hypothesis_per_point": (
                arguments.number_of_toys if toy_enabled else None
            ),
            "toy_seed": arguments.toy_seed if toy_enabled else None,
            "toy_workers": arguments.toy_workers if toy_enabled else None,
            "scan_workers": arguments.scan_workers if toy_enabled else None,
            "scan_parallel_backend": arguments.scan_parallel_backend if toy_enabled else None,
            "toy_batch_size": arguments.toy_batch_size if toy_enabled else None,
            "finite_toy_p_value_correction": (
                "(right_tail_count + 1) / (number_of_toys + 1)"
                if toy_enabled
                else None
            ),
            "adaptive_selection": (
                {
                    "analytic_cls_band": [
                        arguments.adaptive_analytic_cls_min,
                        arguments.adaptive_analytic_cls_max,
                    ],
                    "neighbour_padding_per_mass_row": arguments.adaptive_neighbour_padding,
                    "candidate_points": int(result_table["adaptive_toy_candidate"].sum()),
                    "evaluated_toy_points": int(result_table["adaptive_toy_evaluated"].sum()),
                    "diagnostic_point_limit": arguments.adaptive_toy_point_limit,
                    "hybrid_surface": "Toy CLs at evaluated points; analytic CLs elsewhere",
                }
                if arguments.cls_calibration == "adaptive-toy"
                else None
            ),
            "paper_difference": (
                "paper does not publish exact toy count, seed, nuisance-generation or optimizer settings; these choices are explicit here"
                if toy_enabled
                else "the paper obtains both p-values from pseudo-experiments and repeats its full statistical procedure; this interim contour is not toy-calibrated"
            ),
            "chi2_3nu": chi2_3nu,
        },
        "contour_note": f"the heatmap and red exclusion contour both use {arguments.cls_calibration} CLs; chi2 values remain internal diagnostics and the test-statistic input",
    }
    write_json(output_directory / "metadata.json", metadata)
    print(output_directory)


# Parallel 1+3+1 mass-pair profile scan using the existing likelihood stack.


from sterile_fit.paths import REPOSITORY_ROOT as EXTENDED_ROOT


def extended_csv_values(value: str) -> list[float]:
    values = [float(item) for item in value.split(",")]
    if len(values) < 2 or any(not np.isfinite(item) or item <= 0.0 for item in values):
        raise argparse.ArgumentTypeError(
            "provide at least two positive finite comma-separated coordinates"
        )
    if any(right <= left for left, right in zip(values[:-1], values[1:], strict=True)):
        raise argparse.ArgumentTypeError("scan coordinates must be strictly increasing")
    return values


def extended_log_cell_edges(values: np.ndarray) -> np.ndarray:
    coordinates = np.asarray(values, dtype=float)
    edges = np.empty(coordinates.size + 1, dtype=float)
    edges[1:-1] = np.sqrt(coordinates[:-1] * coordinates[1:])
    edges[0] = coordinates[0] ** 2 / edges[1]
    edges[-1] = coordinates[-1] ** 2 / edges[-2]
    return edges


def extended_format_duration(seconds: float) -> str:
    value = max(0.0, float(seconds))
    if value < 60.0:
        return f"{value:.1f}s"
    minutes, remaining = divmod(value, 60.0)
    if minutes < 60.0:
        return f"{int(minutes)}m {remaining:04.1f}s"
    hours, minutes = divmod(int(minutes), 60)
    return f"{hours}h {minutes:02d}m"


def extended_same_hypotheses(pairs) -> bool:
    return all(
        np.allclose(null.mean, tested.mean, rtol=1e-12, atol=1e-12)
        and np.allclose(null.covariance, tested.covariance, rtol=1e-12, atol=1e-12)
        for null, tested in pairs
    )


def scan_one_plus_three_plus_one() -> None:
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
        default=EXTENDED_ROOT / "configs" / "analyses" / "microboone_bnb.yaml",
    )
    parser.add_argument(
        "--delta-m2-41-absolute-grid-eV2",
        type=extended_csv_values,
        default=np.geomspace(1e-2, 1e2, 7).tolist(),
    )
    parser.add_argument(
        "--delta-m2-51-grid-eV2",
        type=extended_csv_values,
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
        arguments.analysis_config.resolve(), repository_root=EXTENDED_ROOT
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
        repository_root=EXTENDED_ROOT,
        bnb_overrides=overrides or None,
    )
    output_directory = arguments.output_directory
    if output_directory is None:
        output_directory = result_directory(selection.analysis_name, "one_plus_three_plus_one",
                                            f"scan_mass_pair_{arguments.cls_calibration}")
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
        pairs = extended_hypothesis_pairs(analysis, null_parameters, tested)

        if extended_same_hypotheses(pairs):
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
            fixed_test_statistic = prepare_fixed_test_statistic(null_hypotheses, tested_hypotheses)

            comparison = toy_cls(
                observed_test_statistic,
                null_hypotheses,
                tested_hypotheses,
                fixed_test_statistic,
                number_of_toys=arguments.number_of_toys,
                seed=arguments.seed + 1_000_000 + point_index,
                workers=arguments.toy_workers,
            )
            cls_value = comparison.cls
            p_4nu = comparison.p_value_4nu
            p_3nu = comparison.p_value_3nu
            toy_count_4nu = comparison.right_tail_count_under_4nu
            toy_count_3nu = comparison.right_tail_count_under_3nu
            calibration_note = "observed-data profile followed by fixed-hypothesis Toy MC; no Toy refits"

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
            f"CLs={cls_value:.4g}; point {extended_format_duration(perf_counter() - point_started)}, "
            f"ETA {extended_format_duration(remaining)}",
            flush=True,
        )

    table = pd.DataFrame(rows)
    result_path = output_directory / "one_plus_three_plus_one_mass_pair_profile.csv"
    write_csv(table, result_path, index=False, float_format="%.17g")
    plot_path = output_directory / "one_plus_three_plus_one_mass_pair_cls.png"
    extended_plot_result(table, plot_path)
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
    write_json(output_directory / "metadata.json", metadata)
    print(f"Finished in {extended_format_duration(perf_counter() - started)}: {output_directory}")



