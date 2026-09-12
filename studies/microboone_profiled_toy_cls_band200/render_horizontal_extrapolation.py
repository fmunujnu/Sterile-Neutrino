from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd


X_LIMITS = {"fig3a": (1e-4, 1.0), "fig3b": (1e-2, 1.0)}
X_LABELS = {
    "fig3a": r"$\sin^2(2\theta_{\mu e})$",
    "fig3b": r"$\sin^2(2\theta_{ee})$",
}


def fitted_root(log_amplitude: np.ndarray, cls: np.ndarray) -> tuple[float, float]:
    design = np.c_[np.ones(len(log_amplitude)), log_amplitude]
    intercept, slope = np.linalg.lstsq(design, cls, rcond=None)[0]
    if not np.isfinite(slope) or abs(slope) < 1e-12:
        return np.nan, float(slope)
    return float((.05 - intercept) / slope), float(slope)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--points", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--toys", type=int, default=200)
    parser.add_argument("--confidence", type=float, default=.95)
    parser.add_argument("--bootstrap-draws", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260906)
    args = parser.parse_args()
    args.output_directory.mkdir(parents=True, exist_ok=True)
    points = pd.read_csv(args.points)
    rng = np.random.default_rng(args.seed)
    rows = []

    for (figure_name, mass), group in points.groupby(["figure", "mass"]):
        group = group.sort_values("amplitude")
        logx = np.log10(group.amplitude.to_numpy(float))
        central_root, central_slope = fitted_root(logx, group.cls_corrected.to_numpy(float))
        rate3 = (group.tail_3nu.to_numpy(int) + .5) / (args.toys + 1)
        rate4 = (group.tail_4nu.to_numpy(int) + .5) / (args.toys + 1)
        count3 = rng.binomial(args.toys, rate3, size=(args.bootstrap_draws, len(group)))
        count4 = rng.binomial(args.toys, rate4, size=(args.bootstrap_draws, len(group)))
        cls = np.minimum(1.0, (count4 + 1) / np.maximum(count3 + 1, 1))
        roots = np.array([fitted_root(logx, values)[0] for values in cls])
        roots = roots[np.isfinite(roots)]
        lower_log, median_log, upper_log = np.quantile(
            roots, [(1-args.confidence)/2, .5, (1+args.confidence)/2]
        )
        scan_low, scan_high = X_LIMITS[figure_name]
        sample_low, sample_high = group.amplitude.min(), group.amplitude.max()
        rows.append({
            "figure": figure_name, "mass": mass,
            "lower_amplitude_unclipped": 10**lower_log,
            "predicted_amplitude_unclipped": 10**median_log,
            "upper_amplitude_unclipped": 10**upper_log,
            "lower_amplitude": np.clip(10**lower_log, scan_low, scan_high),
            "predicted_amplitude": np.clip(10**median_log, scan_low, scan_high),
            "upper_amplitude": np.clip(10**upper_log, scan_low, scan_high),
            "central_fit_amplitude": np.clip(10**central_root, scan_low, scan_high),
            "central_fit_slope_dcls_dlog10amplitude": central_slope,
            "sample_amplitude_min": sample_low,
            "sample_amplitude_max": sample_high,
            "predicted_is_extrapolated": bool(10**median_log < sample_low or 10**median_log > sample_high),
        })

    contours = pd.DataFrame(rows).sort_values(["figure", "mass"])
    contours.to_csv(args.output_directory / "horizontal_fit_extrapolated_contours.csv", index=False)

    for figure_name, group in points.groupby("figure"):
        line = contours[contours.figure == figure_name].sort_values("mass")
        fig, ax = plt.subplots(figsize=(7.4, 5.7), constrained_layout=True)
        heat = ax.scatter(
            group.amplitude, group.mass, c=group.cls_corrected,
            cmap="turbo", norm=Normalize(.01, .07), marker="s", s=58,
            edgecolors="none", label="200-Toy profiled points",
        )
        ax.plot(line.lower_amplitude, line.mass, color="white", lw=3.0)
        ax.plot(line.upper_amplitude, line.mass, color="white", lw=3.0)
        ax.plot(line.predicted_amplitude, line.mass, color="white", lw=3.3)
        ax.plot(line.lower_amplitude, line.mass, color="tab:blue", lw=1.7,
                label=f"{args.confidence:.0%} lower/upper horizontal-fit limits")
        ax.plot(line.upper_amplitude, line.mass, color="tab:blue", lw=1.7)
        ax.plot(line.predicted_amplitude, line.mass, color="tab:red", lw=2.0,
                label=r"Median predicted $CL_s=0.05$ contour")
        ax.set(xscale="log", yscale="log", xlim=X_LIMITS[figure_name],
               xlabel=X_LABELS[figure_name],
               ylabel=r"$\Delta m^2_{41}\;[\mathrm{eV}^2]$")
        ax.set_title(f"MicroBooNE {figure_name}: horizontal $CL_s$ fits with extrapolation")
        colorbar = fig.colorbar(heat, ax=ax)
        colorbar.set_label(r"$CL_s$ (display clipped to 0.01--0.07)")
        ax.legend(fontsize=8); ax.grid(alpha=.12)
        fig.savefig(args.output_directory / f"{figure_name}_horizontal_extrapolated_heatmap.png", dpi=180)
        plt.close(fig)

    summary = contours.groupby("figure").agg(
        mass_slices=("mass", "size"),
        extrapolated_slices=("predicted_is_extrapolated", "sum"),
        positive_slope_slices=("central_fit_slope_dcls_dlog10amplitude", lambda values: int((values > 0).sum())),
    )
    print(summary.to_string())


if __name__ == "__main__":
    main()
