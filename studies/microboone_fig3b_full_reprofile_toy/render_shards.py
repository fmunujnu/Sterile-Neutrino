"""Merge completed server shards and render the Fig.3b profiled-Toy surface."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


def log_edges(values: np.ndarray) -> np.ndarray:
    edges = np.empty(values.size + 1, dtype=float)
    edges[1:-1] = np.sqrt(values[:-1] * values[1:])
    edges[0] = values[0] ** 2 / edges[1]
    edges[-1] = values[-1] ** 2 / edges[-2]
    return edges


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-directory", type=Path, required=True)
    args = parser.parse_args()
    paths = sorted(args.batch_directory.glob("shards/worker_*/point_calibration.csv"))
    if not paths:
        raise FileNotFoundError("no point-calibration shards found")
    result = pd.concat((pd.read_csv(path) for path in paths), ignore_index=True)
    result = result.sort_values("tested_point_index").reset_index(drop=True)
    indices = result.tested_point_index.to_numpy(int)
    if len(result) != 4514 or not np.array_equal(indices, np.arange(4514)):
        raise RuntimeError("expected exactly the ordered 4514-point Fig.3b grid")
    if not np.all(result.toys_per_hypothesis.to_numpy(int) == 100):
        raise RuntimeError("this completed batch is not uniformly 100 Toys/hypothesis")

    masses = np.sort(result.mass.unique())
    amplitudes = np.sort(result.amplitude.unique())
    surface = result.pivot(index="mass", columns="amplitude", values="cls_corrected")
    surface = surface.reindex(index=masses, columns=amplitudes).to_numpy(float)
    if surface.shape != (74, 61) or not np.all(np.isfinite(surface)):
        raise RuntimeError("the merged CLs surface is incomplete")
    result.to_csv(args.batch_directory / "result.csv", index=False)

    def decorate(axis) -> None:
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlim(amplitudes.min(), amplitudes.max())
        axis.set_ylim(masses.min(), masses.max())
        axis.set_xlabel(r"$\sin^2(2\theta_{ee})$")
        axis.set_ylabel(r"$\Delta m^2_{41}\;[\mathrm{eV}^2]$")
        axis.set_title(r"MicroBooNE BNB+NuMI: per-Toy-profiled $CL_s$")

    figure, axis = plt.subplots(figsize=(7.5, 5.8), constrained_layout=True)
    image = axis.pcolormesh(
        log_edges(amplitudes), log_edges(masses), surface,
        shading="flat", cmap="viridis_r", vmin=0.0, vmax=1.0,
    )
    axis.contour(amplitudes, masses, surface, levels=[0.05], colors="tab:red", linewidths=2)
    decorate(axis)
    figure.colorbar(image, ax=axis, label=r"$CL_s$")
    axis.legend(handles=[Line2D([0], [0], color="tab:red", lw=2,
                                label=r"95% $CL_s$ exclusion boundary")])
    figure.savefig(args.batch_directory / "fig3b_cls_heatmap.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7.5, 5.8), constrained_layout=True)
    axis.contour(amplitudes, masses, surface, levels=[0.05], colors="tab:red", linewidths=2)
    decorate(axis)
    axis.legend(handles=[Line2D([0], [0], color="tab:red", lw=2,
                                label=r"95% $CL_s$ exclusion boundary")])
    figure.savefig(args.batch_directory / "fig3b_cls_contour_only.png", dpi=180)
    plt.close(figure)

    near = result[result.cls_corrected.between(0.03, 0.07)].copy()
    p3 = near.p3_corrected.to_numpy(float)
    p4 = near.p4_corrected.to_numpy(float)
    relative_error = np.sqrt(
        (1.0 - p4) / (100.0 * p4) + (1.0 - p3) / (100.0 * p3)
    )
    diagnostic = {
        "grid_points": int(len(result)),
        "mass_points": int(len(masses)),
        "amplitude_points": int(len(amplitudes)),
        "toys_per_generating_hypothesis": 100,
        "points_with_0p03_le_cls_le_0p07": int(len(near)),
        "near_boundary_tail_4nu_median": float(near.tail_4nu.median()) if len(near) else None,
        "near_boundary_tail_3nu_median": float(near.tail_3nu.median()) if len(near) else None,
        "near_boundary_cls_relative_standard_error_median": (
            float(np.median(relative_error)) if len(near) else None
        ),
        "interpretation": "binomial counting uncertainty only; excludes interpolation and model-systematic uncertainty",
    }
    (args.batch_directory / "render_diagnostics.json").write_text(
        json.dumps(diagnostic, indent=2), encoding="utf-8"
    )
    print(json.dumps(diagnostic, indent=2))


if __name__ == "__main__":
    main()
