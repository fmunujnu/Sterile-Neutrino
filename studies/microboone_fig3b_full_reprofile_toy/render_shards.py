"""Merge completed server shards and render Fig.3a/Fig.3b Toy surfaces."""
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
    if not np.array_equal(indices, np.arange(len(result))):
        raise RuntimeError("combined point indices are incomplete or reordered")
    toy_counts = result.toys_per_hypothesis.unique()
    if len(toy_counts) != 1:
        raise RuntimeError("Toy count differs across completed grid points")
    toys = int(toy_counts[0])
    result.to_csv(args.batch_directory / "result.csv", index=False)

    expected_shapes = {"fig3a": (61, 61), "fig3b": (74, 61)}
    for figure_name, selected in result.groupby("figure", sort=False):
        masses = np.sort(selected.mass.unique())
        amplitudes = np.sort(selected.amplitude.unique())
        surface = selected.pivot(index="mass", columns="amplitude", values="cls_corrected")
        surface = surface.reindex(index=masses, columns=amplitudes).to_numpy(float)
        if surface.shape != expected_shapes[figure_name] or not np.all(np.isfinite(surface)):
            raise RuntimeError(f"the merged {figure_name} CLs surface is incomplete")

        def decorate(axis) -> None:
            axis.set_xscale("log"); axis.set_yscale("log")
            axis.set_xlim(amplitudes.min(), amplitudes.max())
            axis.set_ylim(masses.min(), masses.max())
            axis.set_xlabel(
                r"$\sin^2(2\theta_{\mu e})$" if figure_name == "fig3a"
                else r"$\sin^2(2\theta_{ee})$"
            )
            axis.set_ylabel(r"$\Delta m^2_{41}\;[\mathrm{eV}^2]$")
            axis.set_title(rf"MicroBooNE BNB+NuMI {figure_name}: per-Toy-profiled $CL_s$")

        for heatmap in (True, False):
            figure, axis = plt.subplots(figsize=(7.5, 5.8), constrained_layout=True)
            if heatmap:
                image = axis.pcolormesh(
                    log_edges(amplitudes), log_edges(masses), surface,
                    shading="flat", cmap="viridis_r", vmin=0.0, vmax=1.0,
                )
                figure.colorbar(image, ax=axis, label=r"$CL_s$")
            axis.contour(amplitudes, masses, surface, levels=[0.05], colors="tab:red", linewidths=2)
            decorate(axis)
            axis.legend(handles=[Line2D([0], [0], color="tab:red", lw=2,
                                        label=r"95% $CL_s$ exclusion boundary")])
            suffix = "heatmap" if heatmap else "contour_only"
            figure.savefig(args.batch_directory / f"{figure_name}_cls_{suffix}.png", dpi=180)
            plt.close(figure)

    near = result[result.cls_corrected.between(0.03, 0.07)].copy()
    p3 = near.p3_corrected.to_numpy(float)
    p4 = near.p4_corrected.to_numpy(float)
    relative_error = np.sqrt(
        (1.0 - p4) / (toys * p4) + (1.0 - p3) / (toys * p3)
    )
    diagnostic = {
        "grid_points": int(len(result)),
        "figures": sorted(result.figure.unique().tolist()),
        "toys_per_generating_hypothesis": toys,
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
