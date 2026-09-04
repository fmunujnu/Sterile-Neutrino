"""Redraw completed Fig.3a/Fig.3b quadratic CLs contours without heatmaps."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


def load_surface(directory: Path, x_column: str):
    table = pd.read_csv(directory / "result.csv")
    required = {"fixed_delta_m2_41_eV2", x_column, "cls_quadratic"}
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"{directory}: missing columns {sorted(missing)}")
    pivot = table.pivot(
        index="fixed_delta_m2_41_eV2", columns=x_column, values="cls_quadratic"
    )
    return (
        pivot.columns.to_numpy(dtype=float),
        pivot.index.to_numpy(dtype=float),
        pivot.to_numpy(dtype=float),
    )


def draw(axis, surface, x_label: str, title: str) -> None:
    x, y, cls = surface
    if not float(np.min(cls)) <= 0.05 <= float(np.max(cls)):
        raise ValueError(f"{title}: CLs=0.05 is outside the stored surface")
    axis.contour(x, y, cls, levels=[0.05], colors="tab:red", linewidths=2.2)
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlim(x.min(), x.max())
    axis.set_ylim(y.min(), y.max())
    axis.set_xlabel(x_label)
    axis.set_ylabel(r"$\Delta m^2_{41}\;[\mathrm{eV}^2]$")
    axis.set_title(title)
    axis.grid(False)
    axis.legend(
        handles=[Line2D([0], [0], color="tab:red", linewidth=2.2, label=r"95% $CL_s$")] ,
        loc="best",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fig3a", required=True, type=Path)
    parser.add_argument("--fig3b", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    arguments.output.mkdir(parents=True, exist_ok=True)
    surfaces = (
        ("fig3a", load_surface(arguments.fig3a, "fixed_sin2_2theta_mue"),
         r"$\sin^2(2\theta_{\mu e})$", r"Fig. 3a: quadratic-form 95% $CL_s$"),
        ("fig3b", load_surface(arguments.fig3b, "fixed_sin2_2theta_ee"),
         r"$\sin^2(2\theta_{ee})$", r"Fig. 3b: quadratic-form 95% $CL_s$"),
    )
    for name, surface, x_label, title in surfaces:
        figure, axis = plt.subplots(figsize=(7.5, 5.8))
        draw(axis, surface, x_label, title)
        figure.tight_layout()
        figure.savefig(arguments.output / f"{name}_line_only.png", dpi=180, bbox_inches="tight")
        plt.close(figure)
    figure, axes = plt.subplots(1, 2, figsize=(14.5, 5.8))
    for axis, (_, surface, x_label, title) in zip(axes, surfaces, strict=True):
        draw(axis, surface, x_label, title)
    figure.tight_layout()
    figure.savefig(arguments.output / "fig3a_fig3b_line_only.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
