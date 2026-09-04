"""Collect completed scan plots and redraw their CLs=0.05 contours without heatmaps."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

SCANS = (
    (
        "run1",
        "fig3a_analytic",
        "process24_full_fig3a_analytic",
        "fixed_sin2_2theta_mue",
        "cls_asymptotic",
        r"$\sin^2(2\theta_{\mu e})$",
        r"Fig. 3a: analytic $CL_s$",
    ),
    (
        "run2",
        "fig3a_adaptive_toy",
        "process24_full_fig3a_adaptive_toy_001_03_100",
        "fixed_sin2_2theta_mue",
        "cls_adaptive_hybrid",
        r"$\sin^2(2\theta_{\mu e})$",
        r"Fig. 3a: adaptive profiled Toy-MC $CL_s$",
    ),
    (
        "run1",
        "fig3b_analytic",
        "process24_full_fig3b_analytic",
        "fixed_sin2_2theta_ee",
        "cls_asymptotic",
        r"$\sin^2(2\theta_{ee})$",
        r"Fig. 3b: analytic $CL_s$",
    ),
    (
        "run2",
        "fig3b_adaptive_toy",
        "process24_full_fig3b_adaptive_toy_001_03_100",
        "fixed_sin2_2theta_ee",
        "cls_adaptive_hybrid",
        r"$\sin^2(2\theta_{ee})$",
        r"Fig. 3b: adaptive profiled Toy-MC $CL_s$",
    ),
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run1-root",
        type=Path,
        default=ROOT / "outputs" / "run1_non_toy",
    )
    parser.add_argument(
        "--run2-root",
        type=Path,
        default=ROOT / "outputs" / "run2_toy_mc",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=ROOT / "outputs" / "run2_toy_mc" / "contour_comparison",
    )
    arguments = parser.parse_args()
    arguments.output_directory.mkdir(parents=True, exist_ok=True)

    combined_surfaces = {}

    source_roots = {"run1": arguments.run1_root, "run2": arguments.run2_root}
    for run_name, short_name, directory_name, x_column, cls_column, x_label, title in SCANS:
        source = source_roots[run_name] / directory_name
        table = pd.read_csv(source / "result.csv")
        required = {"fixed_delta_m2_41_eV2", x_column, cls_column}
        missing = required.difference(table.columns)
        if missing:
            raise ValueError(f"{source}: missing columns {sorted(missing)}")

        pivot = table.pivot(
            index="fixed_delta_m2_41_eV2",
            columns=x_column,
            values=cls_column,
        )
        x_values = pivot.columns.to_numpy(dtype=float)
        y_values = pivot.index.to_numpy(dtype=float)
        cls_values = pivot.to_numpy(dtype=float)
        if not cls_values.min() <= 0.05 <= cls_values.max():
            raise ValueError(f"{source}: CLs=0.05 is outside the calculated surface")
        combined_surfaces[short_name] = (x_values, y_values, cls_values)

        figure, axis = plt.subplots(figsize=(7.5, 5.8))
        axis.contour(
            x_values,
            y_values,
            cls_values,
            levels=[0.05],
            colors="tab:red",
            linewidths=2.2,
        )
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlim(x_values.min(), x_values.max())
        axis.set_ylim(y_values.min(), y_values.max())
        axis.set_xlabel(x_label)
        axis.set_ylabel(r"$\Delta m^2_{41}\;[\mathrm{eV}^2]$")
        axis.set_title(title)
        axis.grid(False)
        axis.legend(
            handles=[Line2D([0], [0], color="tab:red", linewidth=2.2, label=r"95% $CL_s$")] ,
            loc="best",
        )
        figure.tight_layout()
        figure.savefig(
            arguments.output_directory / f"line_only_{short_name}.png",
            dpi=180,
            bbox_inches="tight",
        )
        plt.close(figure)

        shutil.copy2(
            source / "profile.png",
            arguments.output_directory / f"original_{short_name}.png",
        )

    figure, axis = plt.subplots(figsize=(8.2, 6.2))
    combined_styles = (
        ("fig3a_analytic", "tab:red", "solid"),
        ("fig3a_adaptive_toy", "tab:green", "solid"),
        ("fig3b_analytic", "tab:red", "dashed"),
        ("fig3b_adaptive_toy", "tab:green", "dashed"),
    )
    for short_name, colour, line_style in combined_styles:
        x_values, y_values, cls_values = combined_surfaces[short_name]
        contours = axis.contour(
            x_values,
            y_values,
            cls_values,
            levels=[0.05],
            colors=colour,
            linewidths=2.0,
            linestyles=line_style,
        )
        # Matplotlib may create several disconnected contour paths; apply the
        # requested line style to every component explicitly.
        contours.set_linestyle(line_style)

    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlim(1e-4, 1.0)
    axis.set_ylim(1e-2, 1e2)
    axis.set_xlabel(
        r"$\sin^2(2\theta_{\mu e})$ (Fig. 3a) or $\sin^2(2\theta_{ee})$ (Fig. 3b)"
    )
    axis.set_ylabel(r"$\Delta m^2_{41}\;[\mathrm{eV}^2]$")
    axis.set_title(r"MicroBooNE 95% $CL_s$ exclusion contours")
    axis.grid(False)
    axis.legend(
        handles=[
            Line2D([0], [0], color="tab:red", linestyle="solid", linewidth=2.0,
                   label=r"Fig. 3a analytic (no Toy MC)"),
            Line2D([0], [0], color="tab:green", linestyle="solid", linewidth=2.0,
                   label=r"Fig. 3a adaptive Toy MC"),
            Line2D([0], [0], color="tab:red", linestyle="dashed", linewidth=2.0,
                   label=r"Fig. 3b analytic (no Toy MC)"),
            Line2D([0], [0], color="tab:green", linestyle="dashed", linewidth=2.0,
                   label=r"Fig. 3b adaptive Toy MC"),
        ],
        loc="best",
        fontsize=8,
    )
    figure.tight_layout()
    figure.savefig(
        arguments.output_directory / "combined_four_contours_single_axes.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)

    paired_plots = (
        (
            "fig3a",
            "fig3a_analytic",
            "fig3a_adaptive_toy",
            r"$\sin^2(2\theta_{\mu e})$",
            r"Fig. 3a: 95% $CL_s$ exclusion contours",
        ),
        (
            "fig3b",
            "fig3b_analytic",
            "fig3b_adaptive_toy",
            r"$\sin^2(2\theta_{ee})$",
            r"Fig. 3b: 95% $CL_s$ exclusion contours",
        ),
    )
    for plot_name, analytic_name, toy_name, x_label, title in paired_plots:
        figure, axis = plt.subplots(figsize=(7.5, 5.8))
        for surface_name, colour in (
            (analytic_name, "tab:red"),
            (toy_name, "tab:green"),
        ):
            x_values, y_values, cls_values = combined_surfaces[surface_name]
            axis.contour(
                x_values,
                y_values,
                cls_values,
                levels=[0.05],
                colors=colour,
                linewidths=2.1,
            )
        analytic_x, analytic_y, _ = combined_surfaces[analytic_name]
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlim(analytic_x.min(), analytic_x.max())
        axis.set_ylim(analytic_y.min(), analytic_y.max())
        axis.set_xlabel(x_label)
        axis.set_ylabel(r"$\Delta m^2_{41}\;[\mathrm{eV}^2]$")
        axis.set_title(title)
        axis.grid(False)
        axis.legend(
            handles=[
                Line2D([0], [0], color="tab:red", linewidth=2.1,
                       label=r"Analytic (no Toy MC)"),
                Line2D([0], [0], color="tab:green", linewidth=2.1,
                       label=r"Adaptive profiled Toy MC"),
            ],
            loc="best",
        )
        figure.tight_layout()
        figure.savefig(
            arguments.output_directory / f"overlay_{plot_name}_analytic_vs_toy.png",
            dpi=180,
            bbox_inches="tight",
        )
        plt.close(figure)

    print(arguments.output_directory)


if __name__ == "__main__":
    main()
