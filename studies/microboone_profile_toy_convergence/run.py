from __future__ import annotations

import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"

import argparse
from pathlib import Path
import sys
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import cholesky
from scipy.stats import beta

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from studies.three_plus_one_toy_distribution_fit.run import _build_analysis
from sterile_fit.experiments.microboone.adapter import _hypothesis_pairs
from sterile_fit.core.calibration import _draw_gaussian_toys, prepare_fixed_hypothesis_chi2
from sterile_fit.core.profile_three_plus_one import (
    profile_s14_s24_at_fixed_sin2_2theta_ee,
    profile_s14_s24_at_fixed_sin2_2theta_mue,
)
from sterile_fit.core.three_plus_one import ThreePlusOneParameters
from sterile_fit.output import write_csv, write_json
from sterile_fit.scan import _profile_toy_at_scan_point, _stable_point_seed


SOURCES = {
    "fig3a": ROOT / "outputs/microboone_bnb_numi_joint/three_plus_one/fullgrid_fixed_toy5000_20260904/scan_fig3a_toy/result.csv",
    "fig3b": ROOT / "outputs/microboone_bnb_numi_joint/three_plus_one/fullgrid_fixed_toy5000_20260904/scan_electron-disfig3a_toy/result.csv",
}
CHECKPOINTS = (50, 100, 200, 500, 1000, 2000, 5000)


def _select_three(frame: pd.DataFrame, mode: str) -> list[dict]:
    x_column = "fixed_sin2_2theta_mue" if mode == "appearance-profile" else "fixed_sin2_2theta_ee"
    candidates = []
    for mass, group in frame.groupby("fixed_delta_m2_41_eV2"):
        if group.cls_toy.min() <= 0.05 <= group.cls_toy.max():
            row = group.loc[(group.cls_toy - 0.05).abs().idxmin()]
            candidates.append({"mass": float(mass), "amplitude": float(row[x_column]),
                               "source_fixed_toy_cls": float(row.cls_toy)})
    indices = np.linspace(0, len(candidates)-1, 3, dtype=int)
    return [dict(candidates[i], mode=mode) for i in indices]


def _profile_observed(analysis, point):
    if point["mode"] == "appearance-profile":
        return profile_s14_s24_at_fixed_sin2_2theta_mue(
            analysis.objective.chi2, delta_m2_41_eV2=point["mass"],
            sin2_2theta_mue=point["amplitude"],
        ).best_fit
    return profile_s14_s24_at_fixed_sin2_2theta_ee(
        analysis.objective.chi2, delta_m2_41_eV2=point["mass"],
        sin2_2theta_ee=point["amplitude"],
    ).best_fit


def _tail_interval(count: int, total: int):
    estimate = (count + 1) / (total + 1)
    low = 0.0 if count == 0 else float(beta.ppf(0.025, count, total-count+1))
    high = 1.0 if count == total else float(beta.ppf(0.975, count+1, total-count))
    return estimate, low, high


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--toys", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.toys != CHECKPOINTS[-1]:
        parser.error(f"this convergence design requires exactly {CHECKPOINTS[-1]} Toys per hypothesis")
    args.output_directory.mkdir(parents=True, exist_ok=args.resume)
    analysis = _build_analysis(ROOT / "configs/analyses/microboone_bnb_numi.yaml")
    points = _select_three(pd.read_csv(SOURCES["fig3a"]), "appearance-profile")
    points += _select_three(pd.read_csv(SOURCES["fig3b"]), "electron-disappearance-profile")
    for i, point in enumerate(points):
        point["point"] = i
    selected_path = args.output_directory / "selected_points.csv"
    if args.resume:
        previous_points = pd.read_csv(selected_path)
        if not np.allclose(previous_points[["mass", "amplitude"]],
                           pd.DataFrame(points)[["mass", "amplitude"]]):
            raise RuntimeError("resume point definitions do not match")
    else:
        write_csv(pd.DataFrame(points), selected_path, index=False)

    null = ThreePlusOneParameters(1.0, 0.0, 0.0)
    raw_path = args.output_directory / "toy_statistics.csv"
    convergence_path = args.output_directory / "convergence.csv"
    raw_rows = (pd.read_csv(raw_path).to_dict("records")
                if args.resume and raw_path.exists() else [])
    convergence_rows = (pd.read_csv(convergence_path).to_dict("records")
                        if args.resume and convergence_path.exists() else [])
    completed_points = {
        int(point) for point, count in
        (pd.DataFrame(raw_rows).groupby("point").size().items() if raw_rows else [])
        if count == 2 * args.toys
    }
    pending_points = [point for point in points if point["point"] not in completed_points]
    started = perf_counter()
    total_profiles = 2 * len(pending_points) * args.toys
    completed = 0
    for point in pending_points:
        fitted = _profile_observed(analysis, point)
        tested = fitted.parameters
        observed_t = fitted.chi2 - analysis.objective.chi2(null)
        pairs = _hypothesis_pairs(analysis, null, tested)
        hypotheses = [tuple(pair[index] for pair in pairs) for index in (0, 1)]
        null_evaluator = prepare_fixed_hypothesis_chi2(hypotheses[0])
        point_seed = _stable_point_seed(args.seed, point["point"])
        generator_seeds = np.random.SeedSequence(point_seed).spawn(2)
        tails = {}
        for hypothesis_index, generator_name in enumerate(("3nu", "4nu")):
            generators = tuple(
                np.random.default_rng(seed)
                for seed in generator_seeds[hypothesis_index].spawn(len(hypotheses[hypothesis_index]))
            )
            factors = tuple(cholesky(item.covariance, lower=True, check_finite=False)
                            for item in hypotheses[hypothesis_index])
            draws = _draw_gaussian_toys(
                hypotheses[hypothesis_index], factors, args.toys, generators
            )
            indicators = np.empty(args.toys, dtype=bool)
            statistics = np.empty(args.toys, dtype=float)
            for toy_index in range(args.toys):
                dataset = tuple(draw[toy_index] for draw in draws)
                chi3 = null_evaluator(dataset)
                toy_fit = _profile_toy_at_scan_point(
                    analysis, dataset, mode=point["mode"], tested_parameters=tested
                )
                statistic = toy_fit.chi2 - chi3
                statistics[toy_index] = statistic
                indicators[toy_index] = statistic >= observed_t
                completed += 1
                if completed % 250 == 0:
                    elapsed = perf_counter()-started
                    eta = elapsed * (total_profiles-completed) / completed
                    print(f"profiles={completed}/{total_profiles} elapsed={elapsed:.1f}s ETA={eta:.1f}s", flush=True)
            tails[generator_name] = indicators
            raw_rows.extend({"point": point["point"], "mode": point["mode"],
                             "generator": generator_name, "toy_index": i,
                             "test_statistic": value, "is_right_tail": bool(indicators[i])}
                            for i, value in enumerate(statistics))
        for sample_size in CHECKPOINTS:
            c3 = int(tails["3nu"][:sample_size].sum())
            c4 = int(tails["4nu"][:sample_size].sum())
            p3, p3low, p3high = _tail_interval(c3, sample_size)
            p4, p4low, p4high = _tail_interval(c4, sample_size)
            convergence_rows.append({**point, "toys_per_hypothesis": sample_size,
                "observed_T": observed_t, "tail_3nu": c3, "tail_4nu": c4,
                "p3": p3, "p4": p4, "cls": min(1.0, p4/p3),
                "cls_interval_low": p4low/p3high if p3high else np.nan,
                "cls_interval_high": min(1.0, p4high/p3low) if p3low else 1.0})
        write_csv(pd.DataFrame(raw_rows), raw_path, index=False)
        write_csv(pd.DataFrame(convergence_rows), convergence_path, index=False)

    convergence = pd.DataFrame(convergence_rows)
    final = convergence[convergence.toys_per_hypothesis == CHECKPOINTS[-1]][["point", "cls"]].rename(columns={"cls": "final_cls"})
    convergence = convergence.merge(final, on="point")
    convergence["absolute_change_from_5000"] = abs(convergence.cls-convergence.final_cls)
    write_csv(convergence, args.output_directory / "convergence.csv", index=False)
    write_json(args.output_directory / "metadata.json", {
        "status": "complete", "toys_per_hypothesis": args.toys,
        "total_profiled_toys": total_profiles, "seed": args.seed,
        "nested_prefixes": list(CHECKPOINTS), "elapsed_seconds": perf_counter()-started,
        "method": "profile allowed mixing parameter(s) independently inside every Toy at each fixed plotted coordinate",
        "sources": {k: str(v) for k, v in SOURCES.items()},
    })

    figure, axes = plt.subplots(1, 2, figsize=(13, 5.2), constrained_layout=True)
    for axis, (mode, title) in zip(axes, (("appearance-profile", "Fig.3a appearance"),
                                         ("electron-disappearance-profile", "Fig.3b electron disappearance"))):
        subset = convergence[convergence["mode"] == mode]
        for point, group in subset.groupby("point"):
            group = group.sort_values("toys_per_hypothesis")
            label = rf"$\Delta m^2={group.mass.iloc[0]:.3g}$ eV$^2$"
            axis.plot(group.toys_per_hypothesis, group.cls, "o-", label=label)
            axis.fill_between(group.toys_per_hypothesis, group.cls_interval_low,
                              group.cls_interval_high, alpha=0.12)
        axis.axhline(0.05, color="black", linestyle="--", linewidth=1)
        axis.set_xscale("log")
        axis.set_xlabel("Toy count per generating hypothesis")
        axis.set_ylabel(r"Profiled-Toy $CL_s$")
        axis.set_title(title)
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    figure.savefig(args.output_directory / "cls_convergence.png", dpi=180, bbox_inches="tight")
    plt.close(figure)
    print(args.output_directory, flush=True)


if __name__ == "__main__":
    main()
