"""Compare analytic CLs and CLs+b= p_4nu contours without Toy MC."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from sterile_fit.output import result_directory


ROOT = Path(__file__).resolve().parents[2]


def _contour_vertices(contour_set) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    segment_index = 0
    for path in contour_set.get_paths():
        vertices = np.asarray(path.vertices, dtype=float)
        if vertices.size == 0:
            continue
        for vertex_index, (x_value, y_value) in enumerate(vertices):
            rows.append({
                "segment_index": segment_index,
                "vertex_index": vertex_index,
                "x": float(x_value),
                "delta_m2_41_eV2": float(y_value),
            })
        segment_index += 1
    return pd.DataFrame(rows)


def _plot_one(
    source: Path,
    output_directory: Path,
    *,
    mode: str,
) -> dict[str, object]:
    table = pd.read_csv(source)
    if mode == "fig3a":
        x_name = "fixed_sin2_2theta_mue"
        x_label = r"$\sin^2(2\theta_{\mu e})$"
        x_limits = (1e-4, 1.0)
        y_limits = (1e-2, 1e2)
    elif mode == "fig3b":
        x_name = "fixed_sin2_2theta_ee"
        x_label = r"$\sin^2(2\theta_{ee})$"
        x_limits = (1e-2, 1.0)
        y_limits = (1e-1, 14.0)
    else:
        raise ValueError(f"unsupported mode {mode!r}")
    required = {
        "fixed_delta_m2_41_eV2",
        x_name,
        "p_value_4nu_asymptotic",
        "p_value_3nu_asymptotic",
        "cls_asymptotic",
    }
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"{source} is missing columns: {sorted(missing)}")

    mass_values = np.sort(table["fixed_delta_m2_41_eV2"].unique())
    x_values = np.sort(table[x_name].unique())
    if mass_values.size < 2 or x_values.size < 2:
        raise ValueError("both contour axes need at least two values")
    cls_surface = table.pivot(
        index="fixed_delta_m2_41_eV2", columns=x_name, values="cls_asymptotic"
    ).loc[mass_values, x_values].to_numpy(dtype=float)
    cls_plus_b_surface = table.pivot(
        index="fixed_delta_m2_41_eV2",
        columns=x_name,
        values="p_value_4nu_asymptotic",
    ).loc[mass_values, x_values].to_numpy(dtype=float)
    x_grid, mass_grid = np.meshgrid(x_values, mass_values)

    figure, axis = plt.subplots(figsize=(7.2, 5.6))
    cls_contour = axis.contour(
        x_grid,
        mass_grid,
        cls_surface,
        levels=[0.05],
        colors=["tab:red"],
        linewidths=2.0,
    )
    cls_plus_b_contour = axis.contour(
        x_grid,
        mass_grid,
        cls_plus_b_surface,
        levels=[0.05],
        colors=["tab:blue"],
        linestyles=["--"],
        linewidths=2.0,
    )
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlim(*x_limits)
    axis.set_ylim(*y_limits)
    axis.set_xlabel(x_label)
    axis.set_ylabel(r"$\Delta m^2_{41}\;[\mathrm{eV}^2]$")
    axis.set_title(r"Analytic non-Toy $3+1$ exclusion diagnostics")
    axis.legend(
        handles=[
            Line2D([0], [0], color="tab:red", linewidth=2.0, label=r"$CL_s=0.05$"),
            Line2D(
                [0],
                [0],
                color="tab:blue",
                linestyle="--",
                linewidth=2.0,
                label=r"$CL_{s+b}=p_{4\nu}=0.05$",
            ),
        ],
        loc="best",
    )
    axis.grid(which="both", alpha=0.18)
    figure.tight_layout()
    figure_path = output_directory / f"{mode}_cls_vs_cls_plus_b.png"
    figure.savefig(figure_path, dpi=200, bbox_inches="tight")
    plt.close(figure)

    cls_vertices = _contour_vertices(cls_contour)
    cls_plus_b_vertices = _contour_vertices(cls_plus_b_contour)
    cls_vertices.insert(0, "criterion", "CLs")
    cls_plus_b_vertices.insert(0, "criterion", "CLs_plus_b")
    pd.concat((cls_vertices, cls_plus_b_vertices), ignore_index=True).to_csv(
        output_directory / f"{mode}_contours.csv", index=False, float_format="%.17g"
    )
    pointwise = table[[
        "fixed_delta_m2_41_eV2",
        x_name,
        "p_value_4nu_asymptotic",
        "p_value_3nu_asymptotic",
        "cls_asymptotic",
    ]].copy()
    pointwise["excluded_by_cls"] = pointwise["cls_asymptotic"] <= 0.05
    pointwise["excluded_by_cls_plus_b"] = (
        pointwise["p_value_4nu_asymptotic"] <= 0.05
    )
    pointwise.to_csv(
        output_directory / f"{mode}_pointwise.csv", index=False, float_format="%.17g"
    )
    return {
        "mode": mode,
        "source": str(source.resolve()),
        "grid_shape": [int(mass_values.size), int(x_values.size)],
        "minimum_cls": float(np.min(cls_surface)),
        "minimum_cls_plus_b": float(np.min(cls_plus_b_surface)),
        "points_excluded_by_cls": int(np.count_nonzero(cls_surface <= 0.05)),
        "points_excluded_by_cls_plus_b": int(
            np.count_nonzero(cls_plus_b_surface <= 0.05)
        ),
        "points_with_different_decision": int(np.count_nonzero(
            (cls_surface <= 0.05) != (cls_plus_b_surface <= 0.05)
        )),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fig3a-result",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--fig3b-result",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=result_directory("studies", "cls_plus_b_non_toy_comparison", "results"),
    )
    arguments = parser.parse_args()
    output_directory = arguments.output_directory.resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    summaries = [
        _plot_one(arguments.fig3a_result.resolve(), output_directory, mode="fig3a"),
        _plot_one(arguments.fig3b_result.resolve(), output_directory, mode="fig3b"),
    ]
    metadata = {
        "model": "3+1",
        "calibration": "existing pointwise analytic Gaussian moments; no Toy MC",
        "tail": "right tail",
        "cls_definition": "p_4nu / p_3nu",
        "cls_plus_b_definition": "p_4nu",
        "threshold": 0.05,
        "core_code_changed": False,
        "results": summaries,
    }
    (output_directory / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    print(output_directory)


if __name__ == "__main__":
    main()
