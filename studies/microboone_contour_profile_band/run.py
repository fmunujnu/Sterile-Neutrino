from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "outputs/studies/microboone_profile_toy_block_shift/contiguous_blocks_toy50_20260906/sample_points.csv"
BASELINES = {
    "fig3a": (ROOT / "outputs/microboone_bnb_numi_joint/three_plus_one/fullgrid_fixed_toy5000_20260904/scan_fig3a_toy/result.csv",
              "fixed_sin2_2theta_mue", r"$\sin^2(2\theta_{\mu e})$"),
    "fig3b": (ROOT / "outputs/microboone_bnb_numi_joint/three_plus_one/fullgrid_fixed_toy5000_20260904/scan_electron-disfig3a_toy/result.csv",
              "fixed_sin2_2theta_ee", r"$\sin^2(2\theta_{ee})$"),
}


def reference_contour(frame: pd.DataFrame, xcol: str) -> pd.DataFrame:
    rows = []
    for mass, group in frame.groupby("fixed_delta_m2_41_eV2"):
        group = group.sort_values(xcol)
        x = group[xcol].to_numpy(float)
        q = group.cls_toy.to_numpy(float) - .05
        hits = np.flatnonzero(q[:-1] * q[1:] <= 0)
        if not len(hits):
            continue
        i = hits[np.argmin(abs(q[hits]) + abs(q[hits + 1]))]
        f = -q[i] / (q[i + 1] - q[i]) if q[i + 1] != q[i] else .5
        root = 10 ** (np.log10(x[i]) + f * (np.log10(x[i + 1]) - np.log10(x[i])))
        rows.append((mass, root))
    return pd.DataFrame(rows, columns=["mass", "amplitude"])


def roots_for_mass(points: pd.DataFrame, toys: int, draws: int, rng: np.random.Generator) -> np.ndarray:
    points = points.sort_values("amplitude")
    logx = np.log10(points.amplitude.to_numpy(float))
    source_n = points.toys_per_hypothesis.to_numpy(int)
    # Jeffreys-smoothed rates prevent zero-count points from being treated as known zero probability.
    rate3 = (points.tail_3nu.to_numpy(int) + .5) / (source_n + 1)
    rate4 = (points.tail_4nu.to_numpy(int) + .5) / (source_n + 1)
    k3 = rng.binomial(toys, rate3, size=(draws, len(points)))
    k4 = rng.binomial(toys, rate4, size=(draws, len(points)))
    q = (k4 + 1) / (toys + 1) - .05 * (k3 + 1) / (toys + 1)
    roots = np.full(draws, np.nan)
    center = float(np.log10(points.baseline_amplitude.median()))
    for row in range(draws):
        crossings = np.flatnonzero(q[row, :-1] * q[row, 1:] <= 0)
        if not len(crossings):
            continue
        candidates = []
        for i in crossings:
            denominator = q[row, i + 1] - q[row, i]
            f = -q[row, i] / denominator if denominator else .5
            candidates.append(logx[i] + f * (logx[i + 1] - logx[i]))
        roots[row] = min(candidates, key=lambda value: abs(value - center))
    return roots


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap-draws", type=int, default=3000)
    parser.add_argument("--toy-counts", default="10,20,30,50")
    parser.add_argument("--seed", type=int, default=20260906)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    args.output_directory.mkdir(parents=True, exist_ok=False)
    toy_counts = [int(value) for value in args.toy_counts.split(",")]
    samples = pd.read_csv(SOURCE)
    rng = np.random.default_rng(args.seed)
    all_bands = []

    for figure_name, (reference_path, xcol, xlabel) in BASELINES.items():
        reference_frame = pd.read_csv(reference_path)
        reference = reference_contour(reference_frame, xcol)
        subset = samples[samples.figure == figure_name].copy()
        reference_masses = np.sort(reference_frame.fixed_delta_m2_41_eV2.unique())
        reference_cls = []
        for row in subset.itertuples():
            nearest_mass = reference_masses[np.argmin(abs(np.log(reference_masses / row.mass)))]
            line = reference_frame[reference_frame.fixed_delta_m2_41_eV2 == nearest_mass].sort_values(xcol)
            reference_cls.append(float(np.interp(
                np.log10(row.amplitude), np.log10(line[xcol]), line.cls_toy
            )))
        subset["reference_cls"] = reference_cls
        subset = subset[(subset.reference_cls >= .005) & (subset.reference_cls <= .1)]
        for toys in toy_counts:
            for mass, points in subset.groupby("mass"):
                if len(points) < 2:
                    continue
                roots = roots_for_mass(points, toys, args.bootstrap_draws, rng)
                finite = roots[np.isfinite(roots)]
                valid_fraction = len(finite) / len(roots)
                if len(finite) < 30:
                    lower = median = upper = np.nan
                else:
                    lower, median, upper = 10 ** np.quantile(finite, [.025, .5, .975])
                all_bands.append({"figure": figure_name, "toys_per_hypothesis": toys,
                                  "mass": mass, "lower_amplitude": lower,
                                  "median_amplitude": median, "upper_amplitude": upper,
                                  "valid_crossing_fraction": valid_fraction})
        bands = pd.DataFrame(all_bands)
        final = bands[(bands.figure == figure_name) & (bands.toys_per_hypothesis == max(toy_counts))].sort_values("mass")
        fig, ax = plt.subplots(figsize=(7.3, 5.5), constrained_layout=True)
        ax.plot(reference.amplitude, reference.mass, color="black", lw=1.7,
                label="5000-Toy reference contour")
        valid = final.dropna()
        ax.plot(valid.lower_amplitude, valid.mass, color="tab:blue", lw=1.8,
                label=f"Pointwise 95% lower/upper limits ({max(toy_counts)} profiled Toys)")
        ax.plot(valid.upper_amplitude, valid.mass, color="tab:blue", lw=1.8)
        ax.fill_betweenx(valid.mass, valid.lower_amplitude, valid.upper_amplitude,
                         color="tab:blue", alpha=.18)
        ax.set(xscale="log", yscale="log", xlabel=xlabel,
               ylabel=r"$\Delta m^2_{41}\;[\mathrm{eV}^2]$")
        ax.set_title(f"MicroBooNE {figure_name}: low-Toy profiled pointwise band")
        ax.legend(fontsize=8); ax.grid(alpha=.15)
        fig.savefig(args.output_directory / f"{figure_name}_profiled_toy_band.png", dpi=180)
        plt.close(fig)

    bands = pd.DataFrame(all_bands)
    bands.to_csv(args.output_directory / "pointwise_bands.csv", index=False)
    widths = bands.dropna().copy()
    widths["log10_full_width"] = np.log10(widths.upper_amplitude / widths.lower_amplitude)
    summary = widths.groupby(["figure", "toys_per_hypothesis"]).agg(
        resolved_mass_points=("mass", "size"),
        median_log10_full_width=("log10_full_width", "median"),
        mean_valid_crossing_fraction=("valid_crossing_fraction", "mean"),
    ).reset_index()
    summary["median_amplitude_factor_width"] = 10 ** summary.median_log10_full_width
    summary.to_csv(args.output_directory / "width_vs_toys.csv", index=False)

    fig, ax = plt.subplots(figsize=(6.8, 4.8), constrained_layout=True)
    for name, group in summary.groupby("figure"):
        ax.plot(group.toys_per_hypothesis, group.median_amplitude_factor_width,
                "o-", label=name)
    ax.set(xlabel="Profiled Toys per generating hypothesis",
           ylabel="Median upper/lower amplitude factor",
           title="Low-Toy pointwise-band width")
    ax.set_xscale("log"); ax.legend(); ax.grid(alpha=.2)
    fig.savefig(args.output_directory / "band_width_vs_toys.png", dpi=180)
    plt.close(fig)
    print(summary.to_string(index=False))
    print(args.output_directory)


if __name__ == "__main__":
    main()
