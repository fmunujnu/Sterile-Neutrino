from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import sys
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import cholesky
from scipy.stats import beta

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from studies.microboone_profile_toy_block_shift.run import _profile_observed  # noqa: E402
from studies.three_plus_one_toy_distribution_fit.run import _build_analysis  # noqa: E402
from sterile_fit.experiments.microboone.adapter import _hypothesis_pairs  # noqa: E402
from sterile_fit.core.calibration import _draw_gaussian_toys, prepare_fixed_hypothesis_chi2  # noqa: E402
from sterile_fit.core.three_plus_one import ThreePlusOneParameters  # noqa: E402
from sterile_fit.scan import _profile_toy_at_scan_point  # noqa: E402

SPECS = {
    "fig3a": {
        "path": ROOT / "outputs/microboone_bnb_numi_joint/three_plus_one/fullgrid_fixed_toy5000_20260904/scan_fig3a_toy/result.csv",
        "mode": "appearance-profile", "x": "fixed_sin2_2theta_mue",
        "xlabel": r"$\sin^2(2\theta_{\mu e})$",
    },
    "fig3b": {
        "path": ROOT / "outputs/microboone_bnb_numi_joint/three_plus_one/fullgrid_fixed_toy5000_20260904/scan_electron-disfig3a_toy/result.csv",
        "mode": "electron-disappearance-profile", "x": "fixed_sin2_2theta_ee",
        "xlabel": r"$\sin^2(2\theta_{ee})$",
    },
}

_ANALYSIS = None
_NULL = None


def initialize_worker() -> None:
    global _ANALYSIS, _NULL
    _ANALYSIS = _build_analysis(ROOT / "configs/analyses/microboone_bnb_numi.yaml")
    _NULL = ThreePlusOneParameters(1.0, 0.0, 0.0)


def evaluate(payload: tuple[str, str, float, float, int, int]) -> tuple[dict, list[dict]]:
    figure, mode, mass, amplitude, toys, seed = payload
    fitted = _profile_observed(_ANALYSIS, mode, mass, amplitude)
    observed_t = fitted.chi2 - _ANALYSIS.objective.chi2(_NULL)
    pairs = _hypothesis_pairs(_ANALYSIS, _NULL, fitted.parameters)
    hypotheses = [tuple(pair[index] for pair in pairs) for index in (0, 1)]
    null_chi2 = prepare_fixed_hypothesis_chi2(hypotheses[0])
    hypothesis_seeds = np.random.SeedSequence(seed).spawn(2)
    values = []
    toy_rows = []
    for hypothesis_index, hypothesis_name in enumerate(("3nu", "4nu")):
        generators = tuple(
            np.random.default_rng(item)
            for item in hypothesis_seeds[hypothesis_index].spawn(len(hypotheses[hypothesis_index]))
        )
        factors = tuple(
            cholesky(item.covariance, lower=True, check_finite=False)
            for item in hypotheses[hypothesis_index]
        )
        draws = _draw_gaussian_toys(hypotheses[hypothesis_index], factors, toys, generators)
        statistics = np.empty(toys, dtype=float)
        for toy_index in range(toys):
            dataset = tuple(draw[toy_index] for draw in draws)
            chi3 = null_chi2(dataset)
            toy_fit = _profile_toy_at_scan_point(
                _ANALYSIS, dataset, mode=mode, tested_parameters=fitted.parameters
            )
            statistics[toy_index] = toy_fit.chi2 - chi3
            toy_rows.append({"figure": figure, "mass": mass, "amplitude": amplitude,
                             "generating_hypothesis": hypothesis_name,
                             "toy_index": toy_index, "test_statistic": statistics[toy_index],
                             "observed_test_statistic": observed_t,
                             "is_right_tail": int(statistics[toy_index] >= observed_t)})
        values.append(statistics)
    c3 = int(np.count_nonzero(values[0] >= observed_t))
    c4 = int(np.count_nonzero(values[1] >= observed_t))
    p3, p4 = (c3 + 1) / (toys + 1), (c4 + 1) / (toys + 1)
    summary = {"figure": figure, "mass": mass, "amplitude": amplitude,
            "toys_per_hypothesis": toys, "tail_3nu": c3, "tail_4nu": c4,
            "p3_corrected": p3, "p4_corrected": p4,
            "cls_corrected": min(1.0, p4 / p3), "q": p4 - .05 * p3,
            "observed_test_statistic": observed_t}
    return summary, toy_rows


def binomial_interval(successes: np.ndarray, trials: int, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    successes = np.asarray(successes, dtype=int)
    lower = np.where(successes == 0, 0.0, beta.ppf(alpha / 2, successes, trials - successes + 1))
    upper = np.where(successes == trials, 1.0, beta.ppf(1 - alpha / 2, successes + 1, trials - successes))
    return lower, upper


def crossing(group: pd.DataFrame, column: str, target: float = .05) -> float:
    group = group.sort_values("amplitude")
    x = group.amplitude.to_numpy(float)
    y = group[column].to_numpy(float) - target
    hits = np.flatnonzero(y[:-1] * y[1:] <= 0)
    if not len(hits):
        return np.nan
    i = hits[np.argmin(abs(y[hits]) + abs(y[hits + 1]))]
    fraction = -y[i] / (y[i + 1] - y[i]) if y[i + 1] != y[i] else .5
    return float(10 ** (np.log10(x[i]) + fraction * (np.log10(x[i + 1]) - np.log10(x[i]))))


def render(results: pd.DataFrame, output: Path, toys: int, confidence: float) -> None:
    alpha = 1 - confidence
    p3_lo, p3_hi = binomial_interval(results.tail_3nu.to_numpy(), toys, alpha)
    p4_lo, p4_hi = binomial_interval(results.tail_4nu.to_numpy(), toys, alpha)
    results = results.copy()
    results["cls_lower"] = p4_lo / np.maximum(p3_hi, np.finfo(float).tiny)
    results["cls_upper"] = p4_hi / np.maximum(p3_lo, np.finfo(float).tiny)
    results["cls_upper"] = np.minimum(results.cls_upper, 1.0)
    results.to_csv(output / "profiled_toy_points_with_intervals.csv", index=False)

    contour_rows = []
    rng = np.random.default_rng(20260906)
    bootstrap_draws = 5000
    for (figure, mass), group in results.groupby(["figure", "mass"]):
        group = group.sort_values("amplitude")
        amplitudes = group.amplitude.to_numpy(float)
        rate3 = (group.tail_3nu.to_numpy(int) + .5) / (toys + 1)
        rate4 = (group.tail_4nu.to_numpy(int) + .5) / (toys + 1)
        count3 = rng.binomial(toys, rate3, size=(bootstrap_draws, len(group)))
        count4 = rng.binomial(toys, rate4, size=(bootstrap_draws, len(group)))
        cls_draws = np.minimum(1.0, (count4 + 1) / np.maximum(count3 + 1, 1))
        roots = []
        central = crossing(group, "cls_corrected")
        central_log = np.log10(central) if np.isfinite(central) else float(np.mean(np.log10(amplitudes)))
        for values in cls_draws:
            q = values - .05
            hits = np.flatnonzero(q[:-1] * q[1:] <= 0)
            candidates = []
            for index in hits:
                denominator = q[index + 1] - q[index]
                fraction = -q[index] / denominator if denominator else .5
                candidates.append(
                    np.log10(amplitudes[index])
                    + fraction * (np.log10(amplitudes[index + 1]) - np.log10(amplitudes[index]))
                )
            if candidates:
                roots.append(min(candidates, key=lambda value: abs(value - central_log)))
        valid_fraction = len(roots) / bootstrap_draws
        if len(roots) >= 30:
            quantiles = 10 ** np.quantile(roots, [(1-confidence)/2, .5, (1+confidence)/2])
            lower_curve, predicted_curve, upper_curve = (
                float(quantiles[0]), float(quantiles[1]), float(quantiles[2])
            )
        else:
            lower_curve = predicted_curve = upper_curve = np.nan
        contour_rows.append({"figure": figure, "mass": mass,
                             "lower_amplitude": lower_curve,
                             "predicted_amplitude": predicted_curve,
                             "upper_amplitude": upper_curve,
                             "central_amplitude": central,
                             "bootstrap_valid_crossing_fraction": valid_fraction})
    contours = pd.DataFrame(contour_rows)
    contours.to_csv(output / "pointwise_confidence_contours.csv", index=False)

    for figure, spec in SPECS.items():
        if figure not in set(results.figure):
            continue
        selected = contours[contours.figure == figure].sort_values("mass").dropna()
        fig, ax = plt.subplots(figsize=(7.3, 5.5), constrained_layout=True)
        ax.plot(selected.lower_amplitude, selected.mass, color="tab:blue", lw=1.8,
                label=f"{confidence:.0%} pointwise lower/upper contours")
        ax.plot(selected.upper_amplitude, selected.mass, color="tab:blue", lw=1.8)
        ax.plot(selected.predicted_amplitude, selected.mass, color="tab:red", lw=2.0,
                label=r"Interpolated median $CL_s=0.05$ contour")
        ax.set(xscale="log", yscale="log", xlabel=spec["xlabel"],
               ylabel=r"$\Delta m^2_{41}\;[\mathrm{eV}^2]$")
        ax.set_title(f"MicroBooNE {figure}: {toys} individually-profiled Toys")
        ax.legend(); ax.grid(alpha=.15)
        fig.savefig(output / f"{figure}_profiled_toy_{confidence:.0%}_band.png", dpi=180)
        fig.savefig(output / f"{figure}_profiled_toy_{confidence:.0%}_three_lines.png", dpi=180)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--toys", type=int, default=200)
    parser.add_argument("--confidence", type=float, default=.95)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260906)
    parser.add_argument("--cls-min", type=float, default=.01)
    parser.add_argument("--cls-max", type=float, default=.07)
    parser.add_argument("--mass-stride", type=int, default=2)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    args.output_directory.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output_directory / "profiled_toy_points_checkpoint.csv"
    toy_values_path = args.output_directory / "profiled_toy_values.csv"
    completed = pd.read_csv(checkpoint) if checkpoint.exists() else pd.DataFrame()
    completed_keys = set(zip(completed.figure, completed.mass, completed.amplitude)) if len(completed) else set()

    payloads = []
    for figure_index, (figure, spec) in enumerate(SPECS.items()):
        frame = pd.read_csv(spec["path"])
        selected = frame[frame.cls_toy.between(args.cls_min, args.cls_max)].sort_values(
            ["fixed_delta_m2_41_eV2", spec["x"]]
        )
        retained_masses = np.sort(selected.fixed_delta_m2_41_eV2.unique())[::args.mass_stride]
        selected = selected[selected.fixed_delta_m2_41_eV2.isin(retained_masses)]
        for point_index, row in selected.reset_index(drop=True).iterrows():
            key = (figure, float(row.fixed_delta_m2_41_eV2), float(row[spec["x"]]))
            if key in completed_keys:
                continue
            payloads.append((figure, spec["mode"], key[1], key[2], args.toys,
                             args.seed + figure_index * 1_000_000 + point_index))

    rows = completed.to_dict("records") if len(completed) else []
    total = len(rows) + len(payloads)
    started = perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers, initializer=initialize_worker) as executor:
        futures = [executor.submit(evaluate, payload) for payload in payloads]
        for future in as_completed(futures):
            summary, toy_rows = future.result()
            rows.append(summary)
            pd.DataFrame(toy_rows).to_csv(
                toy_values_path, mode="a", header=not toy_values_path.exists(), index=False
            )
            pd.DataFrame(rows).sort_values(["figure", "mass", "amplitude"]).to_csv(checkpoint, index=False)
            done = len(rows)
            elapsed = perf_counter() - started
            new_done = done - len(completed)
            eta = (len(payloads) - new_done) * elapsed / max(new_done, 1)
            print(f"points={done}/{total} elapsed={elapsed:.1f}s ETA={eta:.1f}s", flush=True)

    results = pd.DataFrame(rows).sort_values(["figure", "mass", "amplitude"])
    render(results, args.output_directory, args.toys, args.confidence)
    print(args.output_directory)


if __name__ == "__main__":
    main()
