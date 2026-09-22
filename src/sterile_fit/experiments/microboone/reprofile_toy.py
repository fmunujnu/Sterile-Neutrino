"""Active reader and renderer for the completed MicroBooNE reprofile-Toy grid."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from sterile_fit.output import result_directory, render_profiled_cls_surface, write_csv, write_json
from sterile_fit.paths import REPOSITORY_ROOT


DEFAULT_RESULT = (
    REPOSITORY_ROOT
    / "data/experiments/microboone/shared/derived/reprofile_toy_5000/point_calibration.csv"
)
EXPECTED = {"fig3a": 61 * 61, "fig3b": 74 * 61}


def render_completed_reprofile_toy() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figure", choices=("fig3a", "fig3b", "both"), default="both")
    parser.add_argument("--input", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--output-directory", type=Path)
    arguments = parser.parse_args()

    table = pd.read_csv(arguments.input)
    required = {
        "figure", "mass", "amplitude", "toys_per_hypothesis",
        "cls_corrected", "tested_point_index",
    }
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"reprofile-Toy table is missing columns: {sorted(missing)}")
    if len(table) != sum(EXPECTED.values()) or table.tested_point_index.nunique() != len(table):
        raise ValueError("completed reprofile-Toy table must contain all 8,235 unique grid points")
    toy_counts = table.toys_per_hypothesis.unique()
    if len(toy_counts) != 1 or int(toy_counts[0]) != 5000:
        raise ValueError("the active completed result requires 5,000 Toys per hypothesis")

    figures = tuple(EXPECTED) if arguments.figure == "both" else (arguments.figure,)
    output = arguments.output_directory or result_directory(
        "microboone_bnb_numi_joint", "three_plus_one", "reprofile_toy_5000"
    )
    output.mkdir(parents=True, exist_ok=False)
    selected_all = table[table.figure.isin(figures)].copy()
    write_csv(selected_all, output / "result.csv", index=False)

    for name in figures:
        selected = table[table.figure == name].copy()
        if len(selected) != EXPECTED[name] or not np.all(np.isfinite(selected.cls_corrected)):
            raise ValueError(f"{name} surface is incomplete")
        write_csv(selected, output / f"{name}.csv", index=False)
        x_label = r"$\sin^2(2\theta_{\mu e})$" if name == "fig3a" else r"$\sin^2(2\theta_{ee})$"
        title = rf"MicroBooNE BNB+NuMI {name}: per-Toy-profiled $CL_s$"
        render_profiled_cls_surface(selected, output / f"{name}_cls_heatmap.png", x_label=x_label, title=title, heatmap=True)
        render_profiled_cls_surface(selected, output / f"{name}_cls_contour_only.png", x_label=x_label, title=title, heatmap=False)

    write_json(output / "metadata.json", {
        "status": "primary_local_microboone_reprofile_toy_result",
        "source": str(arguments.input),
        "figures": list(figures),
        "grid_points": int(len(selected_all)),
        "toys_per_generating_hypothesis_per_point": 5000,
        "total_toys_per_point": 10000,
        "profile_rule": "repeat the coordinate-constrained profile independently inside every 3nu- and 4nu-generated Toy",
        "limitation": "public-input approximation; not the collaboration's internal event simulation",
    })
    print(output)
