"""Redraw an existing CLs scan without rerunning profile or Toy calculations."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scan-directory", type=Path, required=True)
    parser.add_argument("--x-column", required=True)
    parser.add_argument("--x-label", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--cls-column", default="cls_toy")
    parser.add_argument("--line-only", action="store_true")
    args = parser.parse_args()

    table = pd.read_csv(args.scan_directory / "result.csv")
    pivot = table.pivot(
        index="fixed_delta_m2_41_eV2", columns=args.x_column, values=args.cls_column
    )
    x = pivot.columns.to_numpy(float)
    y = pivot.index.to_numpy(float)
    cls = pivot.to_numpy(float)
    output = args.scan_directory / "redraws"
    output.mkdir(exist_ok=True)

    # Deliberately place many colour intervals around the decision threshold.
    boundaries = np.array([
        0.0, 0.005, 0.015, 0.025, 0.035, 0.042, 0.046, 0.048,
        0.050, 0.052, 0.055, 0.060, 0.070, 0.090, 0.13, 0.20,
        0.40, 0.70, 1.0,
    ])
    cmap = plt.get_cmap("viridis_r", len(boundaries) - 1)
    norm = BoundaryNorm(boundaries, cmap.N, clip=True)

    for line_only in ((True,) if args.line_only else (False, True)):
        figure, axis = plt.subplots(figsize=(8.0, 6.2))
        if not line_only:
            image = axis.pcolormesh(x, y, cls, shading="auto", cmap=cmap, norm=norm)
            figure.colorbar(
                image, ax=axis, label=r"$CL_s$",
                ticks=[0, .015, .035, .046, .05, .055, .07, .13, .2, .4, .7, 1],
            )
        axis.contour(x, y, cls, levels=[0.05], colors="tab:red", linewidths=2.2)
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlim(x.min(), x.max())
        axis.set_ylim(y.min(), y.max())
        axis.set_xlabel(args.x_label)
        axis.set_ylabel(r"$\Delta m^2_{41}\;[\mathrm{eV}^2]$")
        axis.set_title(args.name + (": contour only" if line_only else r": $CL_s=0.05$ focused colour scale"))
        axis.grid(False)
        axis.legend(handles=[Line2D([0], [0], color="tab:red", linewidth=2.2,
                                    label=r"95% $CL_s$")])
        figure.tight_layout()
        suffix = "line_only" if line_only else "threshold_focused_heatmap"
        figure.savefig(output / f"{suffix}.png", dpi=200, bbox_inches="tight")
        plt.close(figure)


if __name__ == "__main__":
    main()
