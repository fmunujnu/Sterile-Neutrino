from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from studies.microboone_profile_toy_convergence.run import _profile_observed
from studies.three_plus_one_toy_distribution_fit.run import _build_analysis
from sterile_fit.core.calibration import prepare_fixed_test_statistic, toy_cls
from sterile_fit.core.three_plus_one import ThreePlusOneParameters
from sterile_fit.experiments.microboone.adapter import _hypothesis_pairs
from sterile_fit.output import write_csv
from sterile_fit.scan import _stable_point_seed


SOURCE = (
    ROOT
    / "outputs/studies/microboone_profile_toy_convergence"
    / "near_contour_5000_20260905"
)
OUTPUT = SOURCE / "fixed_vs_profiled_boundary_points"


def main() -> None:
    points = pd.read_csv(SOURCE / "selected_points.csv")
    raw = pd.read_csv(SOURCE / "toy_statistics.csv")
    analysis = _build_analysis(ROOT / "configs/analyses/microboone_bnb_numi.yaml")
    null = ThreePlusOneParameters(1.0, 0.0, 0.0)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    completed = raw.groupby(["point", "generator"]).size().unstack(fill_value=0)
    completed_points = completed.index[(completed.get("3nu", 0) >= 5000) & (completed.get("4nu", 0) >= 5000)]
    points = points.loc[points["point"].isin(completed_points)].copy()
    figure, axes = plt.subplots(len(points), 2, figsize=(12, 3.35 * len(points)), constrained_layout=True)
    fixed_frames = []
    summaries = []
    for row_index, (_, point_row) in enumerate(points.iterrows()):
        point = point_row.to_dict()
        point_index = int(point["point"])
        profiled = raw.loc[raw["point"] == point_index].copy()
        number_of_toys = int(profiled.groupby("generator").size().min())
        fitted = _profile_observed(analysis, point)
        tested = fitted.parameters
        observed_t = float(fitted.chi2 - analysis.objective.chi2(null))
        pairs = _hypothesis_pairs(analysis, null, tested)
        null_hypotheses = tuple(pair[0] for pair in pairs)
        tested_hypotheses = tuple(pair[1] for pair in pairs)
        statistic = prepare_fixed_test_statistic(null_hypotheses, tested_hypotheses)
        comparison = toy_cls(
            observed_t,
            null_hypotheses,
            tested_hypotheses,
            statistic,
            number_of_toys=number_of_toys,
            seed=_stable_point_seed(20260905, point_index),
            batch_size=4096,
        )
        profile_summary = pd.read_csv(SOURCE / "convergence.csv")
        profile_cls = float(profile_summary.loc[
            (profile_summary["point"] == point_index)
            & (profile_summary["toys_per_hypothesis"] == number_of_toys),
            "cls",
        ].iloc[0])
        fixed = pd.DataFrame({
            "point": point_index,
            "generator": np.repeat(["3nu", "4nu"], number_of_toys),
            "toy_index": np.tile(np.arange(number_of_toys), 2),
            "test_statistic": np.concatenate([
                comparison.test_statistics_under_3nu,
                comparison.test_statistics_under_4nu,
            ]),
        })
        fixed_frames.append(fixed)
        summaries.append({
            "point": point_index,
            "mode": point["mode"],
            "mass": point["mass"],
            "amplitude": point["amplitude"],
            "observed_T": observed_t,
            "fixed_cls_same_toys": comparison.cls,
            "profile_each_toy_cls": profile_cls,
            "cls_difference_profile_minus_fixed": profile_cls - comparison.cls,
        })
        for column_index, generator in enumerate(("3nu", "4nu")):
            axis = axes[row_index, column_index]
            values_fixed = fixed.loc[fixed.generator == generator, "test_statistic"].to_numpy()
            values_profiled = profiled.loc[profiled.generator == generator, "test_statistic"].to_numpy()
            combined = np.concatenate([values_fixed, values_profiled])
            lo, hi = np.quantile(combined, [0.002, 0.998])
            bins = np.linspace(lo, hi, 60)
            axis.hist(values_fixed, bins=bins, density=True, histtype="step", linewidth=1.8,
                      color="tab:red", label=rf"Fixed, $CL_s={comparison.cls:.4f}$")
            axis.hist(values_profiled, bins=bins, density=True, histtype="step", linewidth=1.8,
                      color="tab:blue", label=rf"Profile each Toy, $CL_s={profile_cls:.4f}$")
            axis.axvline(observed_t, color="black", linestyle="--", linewidth=1.2,
                         label=rf"Observed $T={observed_t:.3f}$")
            coordinate = "mue" if point["mode"] == "appearance-profile" else "ee"
            axis.set_title(
                rf"Point {point_index}, {generator} generator: $\Delta m^2={point['mass']:.3g}$, "
                rf"$\sin^2(2\theta_{{{coordinate}}})={point['amplitude']:.3g}$"
            )
            axis.set_xlabel(r"$T=\chi^2_{4\nu}-\chi^2_{3\nu}$")
            axis.set_ylabel("Probability density")
            axis.grid(alpha=0.2)
            axis.legend(fontsize=7)
    fixed_all = pd.concat(fixed_frames, ignore_index=True)
    write_csv(fixed_all, OUTPUT / "fixed_toy_statistics.csv", index=False)
    write_csv(pd.DataFrame(summaries), OUTPUT / "boundary_point_comparison.csv", index=False)
    figure.savefig(OUTPUT / "fixed_vs_profiled_boundary_t_distributions.png", dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    main()
