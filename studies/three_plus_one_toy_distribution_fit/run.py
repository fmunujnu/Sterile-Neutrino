"""Fit candidate analytic families to canonical profiled 3+1 Toy statistics."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear, minimize
from scipy.stats import ks_1samp, ks_2samp, ncx2, pearsonr, spearmanr


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sterile_fit.adapter import _hypothesis_pairs, _objective_for_toy
from sterile_fit.scan import _stable_point_seed
from sterile_fit.adapter import build_three_plus_one_analysis
from sterile_fit.adapter import load_analysis_selection
from sterile_fit.core.profile_three_plus_one import profile_s14_s24_at_fixed_sin2_2theta_mue
from sterile_fit.core.three_plus_one import ThreePlusOneParameters
from sterile_fit.core.calibration import prepare_fixed_hypothesis_chi2
from sterile_fit.core.calibration import _draw_gaussian_toys
from scipy.linalg import cholesky

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from sterile_fit.output import result_directory


_WORKER_ANALYSIS = None


@dataclass(frozen=True, slots=True)
class DifferenceParameters:
    positive_df: float
    positive_noncentrality: float
    negative_df: float
    negative_noncentrality: float


def _build_analysis(configuration: Path):
    selection = load_analysis_selection(configuration, repository_root=ROOT)
    return build_three_plus_one_analysis(selection, repository_root=ROOT)


def _initialise_worker(configuration: str) -> None:
    global _WORKER_ANALYSIS
    _WORKER_ANALYSIS = _build_analysis(Path(configuration))


def _profile_appearance(analysis, toy_dataset, tested_parameters):
    objective = _objective_for_toy(analysis, toy_dataset)
    return profile_s14_s24_at_fixed_sin2_2theta_mue(
        objective,
        delta_m2_41_eV2=tested_parameters.delta_m2_41_eV2,
        sin2_2theta_mue=tested_parameters.sin2_2theta_mue_exact,
    ).best_fit


def _sample_one_point(payload):
    if _WORKER_ANALYSIS is None:
        raise RuntimeError("worker analysis was not initialized")
    point_index, parameter_values, toys, base_seed, generator_name = payload
    tested = ThreePlusOneParameters(**parameter_values)
    null = ThreePlusOneParameters(1.0, 0.0, 0.0)
    pairs = _hypothesis_pairs(_WORKER_ANALYSIS, null, tested)
    null_hypotheses = tuple(pair[0] for pair in pairs)
    tested_hypotheses = tuple(pair[1] for pair in pairs)
    fixed_null_chi2 = prepare_fixed_hypothesis_chi2(null_hypotheses)

    point_seed = _stable_point_seed(base_seed, point_index)
    hypothesis_seeds = np.random.SeedSequence(point_seed).spawn(2)
    hypothesis_index = 0 if generator_name == "3nu" else 1
    generator_hypotheses = (
        null_hypotheses if generator_name == "3nu" else tested_hypotheses
    )
    component_generators = tuple(
        np.random.default_rng(seed)
        for seed in hypothesis_seeds[hypothesis_index].spawn(len(generator_hypotheses))
    )
    lower_factors = tuple(
        cholesky(item.covariance, lower=True, check_finite=False)
        for item in generator_hypotheses
    )
    draws = _draw_gaussian_toys(
        generator_hypotheses, lower_factors, toys, component_generators
    )

    chi2_4nu = np.empty(toys, dtype=float)
    chi2_3nu = np.empty(toys, dtype=float)
    sampling_started = perf_counter()
    for toy_index in range(toys):
        dataset = tuple(component[toy_index] for component in draws)
        chi2_4nu[toy_index] = _profile_appearance(
            _WORKER_ANALYSIS, dataset, tested
        ).chi2
        chi2_3nu[toy_index] = fixed_null_chi2(dataset)
        if toys >= 1000 and (toy_index + 1) % 500 == 0:
            elapsed = perf_counter() - sampling_started
            print(f"point={point_index} H={generator_name}: {toy_index+1}/{toys}, elapsed={elapsed:.0f}s, remaining~{elapsed*(toys/(toy_index+1)-1):.0f}s", flush=True)
    return {
        "point_index": point_index,
        "generator": generator_name,
        "chi2_4nu": chi2_4nu,
        "chi2_3nu": chi2_3nu,
        "test_statistic": chi2_4nu - chi2_3nu,
    }


def _fit_noncentral_chi_square(values: np.ndarray) -> tuple[float, float]:
    sample = np.asarray(values, dtype=float)
    mean = float(np.mean(sample))
    variance = float(np.var(sample, ddof=0))
    initial_noncentrality = max(1e-6, variance / 2.0 - mean)
    initial_df = max(1e-3, mean - initial_noncentrality)

    def objective(log_parameters: np.ndarray) -> float:
        df, noncentrality = np.exp(log_parameters)
        log_pdf = ncx2.logpdf(sample, df, noncentrality)
        if not np.all(np.isfinite(log_pdf)):
            return 1e100
        return float(-np.sum(log_pdf))

    upper = max(1e4, 10.0 * float(np.max(sample)))
    starts = (
        (initial_df, initial_noncentrality),
        (max(mean, 1e-3), 1e-6),
        (1e-3, max(mean, 1e-6)),
        (max(mean / 2.0, 1e-3), max(mean / 2.0, 1e-6)),
    )
    candidates = []
    for start in starts:
        result = minimize(
            objective,
            np.log(start),
            method="L-BFGS-B",
            bounds=((np.log(1e-3), np.log(upper)), (np.log(1e-8), np.log(upper))),
        )
        if result.success and np.isfinite(result.fun):
            candidates.append(result)
    if not candidates:
        raise RuntimeError("all noncentral chi-square fit starts failed")
    best = min(candidates, key=lambda item: item.fun)
    return tuple(float(value) for value in np.exp(best.x))


def _difference_characteristic(
    frequencies: np.ndarray, parameters: DifferenceParameters
) -> np.ndarray:
    t = np.asarray(frequencies, dtype=float)

    def log_ncx2_cf(df: float, noncentrality: float, argument: np.ndarray):
        denominator = 1.0 - 2.0j * argument
        return (
            -0.5 * df * np.log(denominator)
            + 1.0j * noncentrality * argument / denominator
        )

    return np.exp(
        log_ncx2_cf(parameters.positive_df, parameters.positive_noncentrality, t)
        + log_ncx2_cf(parameters.negative_df, parameters.negative_noncentrality, -t)
    )


def _empirical_cumulants(values: np.ndarray) -> np.ndarray:
    centered = values - np.mean(values)
    return np.asarray((
        float(np.mean(values)),
        float(np.mean(centered**2)),
        float(np.mean(centered**3)),
        float(np.mean(centered**4) - 3.0 * np.mean(centered**2) ** 2),
    ))


def _difference_cumulant_matrix() -> np.ndarray:
    # Columns are k_plus, lambda_plus, k_minus, lambda_minus.
    return np.asarray([
        [1.0, 1.0, -1.0, -1.0],
        [2.0, 4.0, 2.0, 4.0],
        [8.0, 24.0, -8.0, -24.0],
        [48.0, 192.0, 48.0, 192.0],
    ])


def _unconstrained_difference_moments(values: np.ndarray) -> np.ndarray:
    return np.linalg.solve(_difference_cumulant_matrix(), _empirical_cumulants(values))


def _fit_difference_of_noncentral_chi_squares(
    values: np.ndarray,
    *,
    component_initial: DifferenceParameters | None = None,
) -> DifferenceParameters:
    sample = np.asarray(values, dtype=float)
    cumulants = _empirical_cumulants(sample)
    standard_deviation = max(float(np.std(sample, ddof=0)), 1e-8)
    scales = np.maximum(
        np.abs(cumulants),
        np.asarray([
            standard_deviation,
            standard_deviation**2,
            standard_deviation**3,
            standard_deviation**4,
        ]),
    )
    matrix = _difference_cumulant_matrix()
    result = lsq_linear(
        matrix / scales[:, None],
        cumulants / scales,
        bounds=(1e-8, np.inf),
        lsmr_tol="auto",
        max_iter=10_000,
    )
    if not result.success:
        raise RuntimeError(f"constrained four-cumulant fit failed: {result.message}")
    return DifferenceParameters(*map(float, result.x))


def _simulate_difference(
    parameters: DifferenceParameters, size: int, generator: np.random.Generator
) -> np.ndarray:
    positive = generator.noncentral_chisquare(
        parameters.positive_df, parameters.positive_noncentrality, size=size
    )
    negative = generator.noncentral_chisquare(
        parameters.negative_df, parameters.negative_noncentrality, size=size
    )
    return positive - negative


def _fit_and_score_group(frame: pd.DataFrame, fit_seed: int):
    chi4 = frame["chi2_4nu"].to_numpy(dtype=float)
    chi3 = frame["chi2_3nu"].to_numpy(dtype=float)
    statistic = frame["test_statistic"].to_numpy(dtype=float)
    chi4_df, chi4_nc = _fit_noncentral_chi_square(chi4)
    chi3_df, chi3_nc = _fit_noncentral_chi_square(chi3)
    component_parameters = DifferenceParameters(chi4_df, chi4_nc, chi3_df, chi3_nc)
    free_parameters = _fit_difference_of_noncentral_chi_squares(
        statistic, component_initial=component_parameters
    )
    chi4_ks = ks_1samp(chi4, lambda x: ncx2.cdf(x, chi4_df, chi4_nc))
    random_generator = np.random.default_rng(fit_seed)
    free_reference = _simulate_difference(free_parameters, 200_000, random_generator)
    component_reference = _simulate_difference(
        component_parameters, 200_000, random_generator
    )
    free_ks = ks_2samp(statistic, free_reference, method="asymp")
    component_ks = ks_2samp(statistic, component_reference, method="asymp")
    pearson = pearsonr(chi4, chi3)
    spearman = spearmanr(chi4, chi3)
    chi4_mean = float(np.mean(chi4))
    chi4_variance = float(np.var(chi4, ddof=0))
    unconstrained_difference = _unconstrained_difference_moments(statistic)
    predicted_cumulants = _difference_cumulant_matrix() @ np.asarray(
        list(asdict(free_parameters).values())
    )
    empirical_cumulants = _empirical_cumulants(statistic)
    cumulant_scales = np.maximum(
        np.abs(empirical_cumulants),
        np.asarray([
            max(float(np.std(statistic)), 1e-8),
            max(float(np.std(statistic)), 1e-8) ** 2,
            max(float(np.std(statistic)), 1e-8) ** 3,
            max(float(np.std(statistic)), 1e-8) ** 4,
        ]),
    )
    return {
        "sample_size": len(frame),
        "chi2_4nu_df": chi4_df,
        "chi2_4nu_noncentrality": chi4_nc,
        "chi2_4nu_ks_statistic": float(chi4_ks.statistic),
        "chi2_4nu_ks_p_value_naive": float(chi4_ks.pvalue),
        "chi2_4nu_variance_over_mean": chi4_variance / chi4_mean,
        "chi2_4nu_ncx2_moment_feasible": bool(
            2.0 * chi4_mean <= chi4_variance <= 4.0 * chi4_mean
        ),
        **{f"T_free_{key}": value for key, value in asdict(free_parameters).items()},
        "T_free_ks_statistic": float(free_ks.statistic),
        "T_free_ks_p_value_naive": float(free_ks.pvalue),
        "T_unconstrained_four_cumulant_feasible": bool(
            np.all(unconstrained_difference >= 0.0)
        ),
        "T_constrained_four_cumulant_relative_residual": float(
            np.linalg.norm((predicted_cumulants - empirical_cumulants) / cumulant_scales)
        ),
        **{f"T_component_{key}": value for key, value in asdict(component_parameters).items()},
        "T_component_ks_statistic": float(component_ks.statistic),
        "T_component_ks_p_value_naive": float(component_ks.pvalue),
        "chi2_4nu_chi2_3nu_pearson_r": float(pearson.statistic),
        "chi2_4nu_chi2_3nu_pearson_p": float(pearson.pvalue),
        "chi2_4nu_chi2_3nu_spearman_r": float(spearman.statistic),
        "chi2_4nu_chi2_3nu_spearman_p": float(spearman.pvalue),
    }, free_parameters


def _make_plots(samples, summaries, free_parameters_by_group, output_directory):
    for point_index in sorted(samples["point_index"].unique()):
        figure, axes = plt.subplots(2, 2, figsize=(11, 7.5))
        for column, generator_name in enumerate(("3nu", "4nu")):
            group = samples[
                (samples["point_index"] == point_index)
                & (samples["generator"] == generator_name)
            ]
            summary = summaries[
                (summaries["point_index"] == point_index)
                & (summaries["generator"] == generator_name)
            ].iloc[0]
            chi4 = group["chi2_4nu"].to_numpy(dtype=float)
            statistic = group["test_statistic"].to_numpy(dtype=float)
            axes[0, column].hist(chi4, bins="fd", density=True, alpha=0.45, color="tab:blue")
            grid = np.linspace(max(0.0, chi4.min()), chi4.max(), 500)
            axes[0, column].plot(
                grid,
                ncx2.pdf(grid, summary.chi2_4nu_df, summary.chi2_4nu_noncentrality),
                color="black",
                linewidth=1.8,
            )
            axes[0, column].set_title(
                rf"$\chi^2_{{4\nu}}$ under {generator_name}; "
                rf"KS $p={summary.chi2_4nu_ks_p_value_naive:.3g}$"
            )
            axes[0, column].set_xlabel(r"profiled $\chi^2_{4\nu}$")
            axes[0, column].set_ylabel("density")

            parameters = free_parameters_by_group[(point_index, generator_name)]
            reference = _simulate_difference(
                parameters, 200_000, np.random.default_rng(70_000 + 10 * point_index + column)
            )
            axes[1, column].hist(statistic, bins="fd", density=True, alpha=0.45, color="tab:orange")
            axes[1, column].hist(
                reference,
                bins=120,
                density=True,
                histtype="step",
                color="black",
                linewidth=1.5,
                range=(float(statistic.min()), float(statistic.max())),
            )
            axes[1, column].set_title(
                rf"$T$ under {generator_name}; KS $p={summary.T_free_ks_p_value_naive:.3g}$"
            )
            axes[1, column].set_xlabel(r"$T=\chi^2_{4\nu}-\chi^2_{3\nu}$")
            axes[1, column].set_ylabel("density")
        point = samples[samples["point_index"] == point_index].iloc[0]
        figure.suptitle(
            rf"Point {point_index}: $\Delta m^2_{{41}}={point.fixed_delta_m2_41_eV2:.4g}$ eV$^2$, "
            rf"$\sin^2(2\theta_{{\mu e}})={point.fixed_sin2_2theta_mue:.4g}$"
        )
        figure.tight_layout()
        figure.savefig(output_directory / f"point_{point_index:02d}_fits.png", dpi=180)
        plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--analysis-config",
        type=Path,
        default=ROOT / "configs" / "analyses" / "microboone_bnb_numi.yaml",
    )
    parser.add_argument("--points", type=int, default=4)
    parser.add_argument("--toys", type=int, default=200)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--point-seed", type=int, default=20260825)
    parser.add_argument("--toy-seed", type=int, default=20250821)
    parser.add_argument("--samples-only", action="store_true", help="Save raw profiled Toys; skip unrelated distribution-family fitting")
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=result_directory("studies", "three_plus_one_toy_distribution_fit", "results"),
    )
    arguments = parser.parse_args()
    if arguments.points < 3:
        raise ValueError("at least three random points are needed for a cross-point check")
    if arguments.toys < 20:
        raise ValueError("at least 20 toys per hypothesis are needed for distribution fitting")
    if arguments.workers < 1:
        raise ValueError("workers must be positive")
    output_directory = arguments.output_directory.resolve()
    output_directory.mkdir(parents=True, exist_ok=True)

    analysis = _build_analysis(arguments.analysis_config.resolve())
    coordinate_generator = np.random.default_rng(arguments.point_seed)
    delta_m2_values = 10.0 ** coordinate_generator.uniform(-2.0, 2.0, arguments.points)
    appearance_values = 10.0 ** coordinate_generator.uniform(-4.0, 0.0, arguments.points)
    points = []
    for point_index, (delta_m2, appearance) in enumerate(
        zip(delta_m2_values, appearance_values, strict=True)
    ):
        profile = profile_s14_s24_at_fixed_sin2_2theta_mue(
            analysis.objective.chi2,
            delta_m2_41_eV2=float(delta_m2),
            sin2_2theta_mue=float(appearance),
        )
        parameters = profile.best_fit.parameters
        points.append((point_index, parameters, profile.best_fit.chi2))

    payloads = [
        (
            point_index,
            {
                "delta_m2_41_eV2": parameters.delta_m2_41_eV2,
                "sin2_theta14": parameters.sin2_theta14,
                "sin2_theta24": parameters.sin2_theta24,
            },
            arguments.toys,
            arguments.toy_seed,
            generator_name,
        )
        for point_index, parameters, _ in points
        for generator_name in ("3nu", "4nu")
    ]
    started = perf_counter()
    results = []
    with ProcessPoolExecutor(
        max_workers=min(arguments.workers, len(payloads)),
        initializer=_initialise_worker,
        initargs=(str(arguments.analysis_config.resolve()),),
    ) as executor:
        futures = [executor.submit(_sample_one_point, payload) for payload in payloads]
        for completed, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            results.append(result)
            # Save each completed hypothesis immediately, before later groups finish.
            checkpoint = output_directory / "checkpoints"
            checkpoint.mkdir(exist_ok=True)
            pd.DataFrame({key: result[key] for key in ("chi2_4nu", "chi2_3nu", "test_statistic")}).to_csv(
                checkpoint / f"point_{result['point_index']:02d}_{result['generator']}.csv",
                index=False, float_format="%.17g")
            elapsed = perf_counter() - started
            estimated_total = elapsed * len(futures) / completed
            print(
                f"completed {completed}/{len(futures)} point-hypotheses; "
                f"elapsed={elapsed:.1f}s; estimated_remaining={max(0.0, estimated_total-elapsed):.1f}s",
                flush=True,
            )

    parameter_lookup = {point_index: (parameters, chi2) for point_index, parameters, chi2 in points}
    sample_frames = []
    for result in results:
        parameters, observed_chi2 = parameter_lookup[result["point_index"]]
        sample_frames.append(pd.DataFrame({
            "point_index": result["point_index"],
            "generator": result["generator"],
            "toy_index": np.arange(arguments.toys),
            "fixed_delta_m2_41_eV2": parameters.delta_m2_41_eV2,
            "fixed_sin2_2theta_mue": parameters.sin2_2theta_mue_exact,
            "profiled_observed_sin2_theta14": parameters.sin2_theta14,
            "derived_observed_sin2_theta24": parameters.sin2_theta24,
            "observed_profiled_chi2_4nu": observed_chi2,
            "chi2_4nu": result["chi2_4nu"],
            "chi2_3nu": result["chi2_3nu"],
            "test_statistic": result["test_statistic"],
        }))
    samples = pd.concat(sample_frames, ignore_index=True).sort_values(
        ["point_index", "generator", "toy_index"]
    )
    samples.to_csv(output_directory / "toy_samples.csv", index=False, float_format="%.17g")

    if arguments.samples_only:
        from sterile_fit.output import write_json
        write_json(output_directory / "metadata.json", {
            "analysis_configuration": str(arguments.analysis_config.resolve()),
            "analysis_name": analysis.analysis_name, "scan_mode": "appearance-profile",
            "point_selection": "deterministic log-uniform random coordinates",
            "point_seed": arguments.point_seed, "toy_seed": arguments.toy_seed,
            "points": arguments.points, "toys_per_hypothesis_per_point": arguments.toys,
            "workers": arguments.workers, "samples_only": True,
            "runtime_seconds": perf_counter()-started,
            "toy_profile": "canonical fixed-mass and fixed-exact-appearance profile for every Toy",
        })
        print(output_directory, flush=True)
        return

    summary_rows = []
    free_parameters_by_group = {}
    for (point_index, generator_name), group in samples.groupby(
        ["point_index", "generator"], sort=True
    ):
        fitted, free_parameters = _fit_and_score_group(
            group, 50_000 + 10 * int(point_index) + (generator_name == "4nu")
        )
        free_parameters_by_group[(int(point_index), str(generator_name))] = free_parameters
        first = group.iloc[0]
        summary_rows.append({
            "point_index": point_index,
            "generator": generator_name,
            "fixed_delta_m2_41_eV2": first.fixed_delta_m2_41_eV2,
            "fixed_sin2_2theta_mue": first.fixed_sin2_2theta_mue,
            **fitted,
        })
    summaries = pd.DataFrame(summary_rows)

    shared_rows = []
    for generator_name, generator_group in samples.groupby("generator", sort=True):
        shared_fit, shared_difference = _fit_and_score_group(
            generator_group, 80_000 + (generator_name == "4nu")
        )
        shared_rows.append({
            "scope": "all_points_pooled",
            "generator": generator_name,
            "held_out_point_index": "",
            **shared_fit,
        })
        for held_out_point in sorted(samples["point_index"].unique()):
            training = generator_group[generator_group["point_index"] != held_out_point]
            held_out = generator_group[generator_group["point_index"] == held_out_point]
            training_fit, training_difference = _fit_and_score_group(
                training, 90_000 + int(held_out_point)
            )
            chi4_ks = ks_1samp(
                held_out["chi2_4nu"].to_numpy(),
                lambda x: ncx2.cdf(
                    x,
                    training_fit["chi2_4nu_df"],
                    training_fit["chi2_4nu_noncentrality"],
                ),
            )
            reference = _simulate_difference(
                training_difference,
                200_000,
                np.random.default_rng(100_000 + int(held_out_point)),
            )
            statistic_ks = ks_2samp(
                held_out["test_statistic"].to_numpy(), reference, method="asymp"
            )
            shared_rows.append({
                "scope": "leave_one_point_out",
                "generator": generator_name,
                "held_out_point_index": int(held_out_point),
                "training_chi2_4nu_df": training_fit["chi2_4nu_df"],
                "training_chi2_4nu_noncentrality": training_fit["chi2_4nu_noncentrality"],
                "held_out_chi2_4nu_ks_statistic": float(chi4_ks.statistic),
                "held_out_chi2_4nu_ks_p_value": float(chi4_ks.pvalue),
                **{f"training_T_{key}": value for key, value in asdict(training_difference).items()},
                "held_out_T_ks_statistic": float(statistic_ks.statistic),
                "held_out_T_ks_p_value": float(statistic_ks.pvalue),
            })
    summaries.to_csv(output_directory / "pointwise_fit_summary.csv", index=False, float_format="%.17g")
    pd.DataFrame(shared_rows).to_csv(
        output_directory / "shared_parameter_checks.csv", index=False, float_format="%.17g"
    )
    _make_plots(samples, summaries, free_parameters_by_group, output_directory)

    metadata = {
        "analysis_configuration": str(arguments.analysis_config.resolve()),
        "analysis_name": analysis.analysis_name,
        "scientific_status": [experiment.status for experiment in analysis.experiments],
        "scan_mode": "appearance-profile",
        "point_selection": "deterministic log-uniform random coordinates",
        "point_seed": arguments.point_seed,
        "toy_seed": arguments.toy_seed,
        "points": arguments.points,
        "toys_per_hypothesis_per_point": arguments.toys,
        "workers": min(arguments.workers, len(payloads)),
        "toy_generator": "canonical full prediction-dependent Gaussian covariance; no clipping and no extra Poisson draw",
        "toy_profile": "canonical fixed-delta_m2_41 and fixed exact sin2(2theta_mue) profile repeated in every toy",
        "chi2_4nu_candidate": "noncentral chi-square with loc=0 and scale=1; df and noncentrality fitted",
        "T_candidate": "difference of two independent noncentral chi-square variables; four positive parameters fitted by nonnegative weighted four-cumulant matching",
        "goodness_of_fit_warning": "reported KS p-values are naive because the same finite sample is used for fitting; use them as diagnostics, not calibrated acceptance probabilities",
        "runtime_seconds": perf_counter() - started,
    }
    (output_directory / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(output_directory)


if __name__ == "__main__":
    main()
