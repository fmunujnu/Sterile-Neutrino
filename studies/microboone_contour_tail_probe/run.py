from __future__ import annotations

import argparse
from pathlib import Path
import sys
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from studies.microboone_profile_toy_block_shift.run import (  # noqa: E402
    SPECS,
    _baseline,
    _evaluate_point,
)
from studies.three_plus_one_toy_distribution_fit.run import _build_analysis  # noqa: E402
from sterile_fit.core.three_plus_one import ThreePlusOneParameters  # noqa: E402


def evenly_spaced_rows(frame: pd.DataFrame, count: int) -> pd.DataFrame:
    indices = np.unique(np.rint(np.linspace(0, len(frame) - 1, count)).astype(int))
    return frame.iloc[indices].reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--points-per-figure", type=int, default=24)
    parser.add_argument("--toys", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260906)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    args.output_directory.mkdir(parents=True, exist_ok=False)

    analysis = _build_analysis(ROOT / "configs/analyses/microboone_bnb_numi.yaml")
    null = ThreePlusOneParameters(1.0, 0.0, 0.0)
    rows: list[dict[str, float | int | str]] = []
    total = len(SPECS) * args.points_per_figure
    started = perf_counter()

    for figure_index, (figure_name, spec) in enumerate(SPECS.items()):
        scan = pd.read_csv(spec["path"])
        contour = evenly_spaced_rows(_baseline(scan, spec["x"]), args.points_per_figure)
        for point_index, point in contour.iterrows():
            mass = float(point.mass)
            amplitude = float(point.baseline_amplitude)
            seed = args.seed + figure_index * 100000 + point_index
            c3, c4, p3, p4, q, cls = _evaluate_point(
                analysis, null, spec["mode"], mass, amplitude, args.toys, seed
            )
            rows.append({
                "figure": figure_name,
                "point_index": point_index,
                "delta_m2_41_eV2": mass,
                "contour_amplitude": amplitude,
                "toys_per_hypothesis": args.toys,
                "tail_3nu": c3,
                "tail_4nu": c4,
                "p3_corrected": p3,
                "p4_corrected": p4,
                "cls_corrected": cls,
                "q": q,
            })
            done = len(rows)
            elapsed = perf_counter() - started
            print(f"points={done}/{total} elapsed={elapsed:.1f}s ETA={(total-done)*elapsed/done:.1f}s", flush=True)

    result = pd.DataFrame(rows)
    result.to_csv(args.output_directory / "contour_tail_probe.csv", index=False)

    figure, axes = plt.subplots(2, 1, figsize=(7.4, 7.2), sharex=True, constrained_layout=True)
    colors = {"fig3a": "tab:blue", "fig3b": "tab:orange"}
    for figure_name, group in result.groupby("figure"):
        ordered = group.sort_values("delta_m2_41_eV2")
        axes[0].plot(ordered.delta_m2_41_eV2, ordered.tail_3nu, "o-", ms=3,
                     color=colors[figure_name], label=figure_name)
        axes[1].plot(ordered.delta_m2_41_eV2, ordered.tail_4nu, "o-", ms=3,
                     color=colors[figure_name], label=figure_name)
    axes[0].set_ylabel(r"$3\nu$ tail count")
    axes[1].set_ylabel(r"$4\nu$ tail count")
    axes[1].set_xlabel(r"$\Delta m^2_{41}\;[\mathrm{eV}^2]$")
    for axis in axes:
        axis.set_xscale("log")
        axis.set_ylim(bottom=-0.5)
        axis.legend()
        axis.grid(alpha=.2)
    figure.suptitle(f"Tail counts along existing contours ({args.toys} Toys/hypothesis)")
    figure.savefig(args.output_directory / "tail_counts_comparison.png", dpi=180)
    plt.close(figure)

    summary = result.groupby("figure").agg(
        points=("point_index", "size"),
        zero_tail_3nu=("tail_3nu", lambda x: int((x == 0).sum())),
        zero_tail_4nu=("tail_4nu", lambda x: int((x == 0).sum())),
        median_tail_3nu=("tail_3nu", "median"),
        median_tail_4nu=("tail_4nu", "median"),
        mean_tail_3nu=("tail_3nu", "mean"),
        mean_tail_4nu=("tail_4nu", "mean"),
    ).reset_index()
    summary.to_csv(args.output_directory / "summary.csv", index=False)
    print(args.output_directory)


if __name__ == "__main__":
    main()
