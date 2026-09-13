"""Full Fig.3a/Fig.3b grids with per-Toy constrained reprofile."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
from time import perf_counter

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from studies.microboone_profiled_toy_cls_band200.run import (  # noqa: E402
    evaluate,
    initialize_worker,
)
from sterile_fit.scan import _stable_point_seed  # noqa: E402
from sterile_fit.output import write_json  # noqa: E402


FIGURE_SPECS = {
    "fig3a": {
        "mode": "appearance-profile",
        "mass_grid": np.geomspace(1.0e-2, 1.0e2, 61),
        "amplitude_grid": np.geomspace(1.0e-4, 1.0, 61),
        "amplitude_name": "sin2_2theta_mue",
    },
    "fig3b": {
        "mode": "electron-disappearance-profile",
        "mass_grid": np.geomspace(0.1, 40.0, 74),
        "amplitude_grid": np.geomspace(0.01, 1.0, 61),
        "amplitude_name": "sin2_2theta_ee",
    },
}


def selected_points(figures: str) -> list[tuple[str, int, float, float]]:
    names = tuple(FIGURE_SPECS) if figures == "both" else (figures,)
    return [
        (name, local_index, float(mass), float(amplitude))
        for name in names
        for local_index, (mass, amplitude) in enumerate(
            (mass, amplitude)
            for mass in FIGURE_SPECS[name]["mass_grid"]
            for amplitude in FIGURE_SPECS[name]["amplitude_grid"]
        )
    ]


def append_row(path: Path, row: dict[str, object]) -> None:
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)
        stream.flush()


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figures", choices=("fig3a", "fig3b", "both"), default="both")
    parser.add_argument("--toys", type=int, default=5000,
                        help="Toy count under each of the 3nu and 4nu generating hypotheses")
    parser.add_argument("--seed", type=int, default=20260913)
    parser.add_argument("--start-point", type=int, default=0)
    parser.add_argument("--stop-point", type=int)
    parser.add_argument("--output-directory", type=Path, required=True)
    result = parser.parse_args()
    if result.toys < 2:
        parser.error("--toys must be at least 2 per generating hypothesis")
    total_points = len(selected_points(result.figures))
    if result.stop_point is None:
        result.stop_point = total_points
    if not 0 <= result.start_point < result.stop_point <= total_points:
        parser.error(f"require 0 <= start-point < stop-point <= {total_points}")
    return result


def main() -> None:
    args = arguments()
    args.output_directory.mkdir(parents=True, exist_ok=False)
    summary_path = args.output_directory / "point_calibration.csv"
    toy_path = args.output_directory / "toy_statistics.csv"
    initialize_worker()
    started = perf_counter()
    points = selected_points(args.figures)
    selected = range(args.start_point, args.stop_point)

    for completed, point_index in enumerate(selected, start=1):
        figure, local_index, mass, amplitude = points[point_index]
        spec = FIGURE_SPECS[figure]
        seed_offset = 0 if figure == "fig3b" else 1_000_000
        seed = _stable_point_seed(args.seed + seed_offset, local_index)
        summary, toy_rows = evaluate((
            figure, spec["mode"],
            mass, amplitude, args.toys, seed,
        ))
        summary["tested_point_index"] = point_index
        summary["figure_point_index"] = local_index
        append_row(summary_path, summary)
        for row in toy_rows:
            row["tested_point_index"] = point_index
            row["figure_point_index"] = local_index
        if toy_rows:
            exists = toy_path.exists()
            with toy_path.open("a", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(toy_rows[0]))
                if not exists:
                    writer.writeheader()
                writer.writerows(toy_rows)

        elapsed = perf_counter() - started
        remaining = elapsed / completed * ((args.stop_point - args.start_point) - completed)
        print(
            f"point {completed}/{args.stop_point - args.start_point} "
            f"(global {point_index}); elapsed={elapsed:.1f}s; ETA~{remaining:.1f}s",
            flush=True,
        )

    write_json(args.output_directory / "metadata.json", {
        "scientific_status": "research_full_grid_per_toy_reprofile",
        "figures": args.figures,
        "figure_grids": {
            name: {
                "mass_grid_eV2": {"minimum": float(spec["mass_grid"][0]), "maximum": float(spec["mass_grid"][-1]), "points": len(spec["mass_grid"])},
                spec["amplitude_name"] + "_grid": {"minimum": float(spec["amplitude_grid"][0]), "maximum": float(spec["amplitude_grid"][-1]), "points": len(spec["amplitude_grid"])},
            }
            for name, spec in FIGURE_SPECS.items()
            if args.figures == "both" or args.figures == name
        },
        "toys_per_generating_hypothesis_per_point": args.toys,
        "total_toys_per_point": 2 * args.toys,
        "profile_rule": (
            "profile observed data at each plotted coordinate; then independently "
            "repeat the same constrained s24 profile on both physical s14 branches "
            "inside every 3nu-generated and 4nu-generated Toy"
        ),
        "seed": args.seed,
        "point_range": [args.start_point, args.stop_point],
        "elapsed_seconds": perf_counter() - started,
        "limitation": "Toy precision must be assessed from the empirical tail counts",
    })
    print(args.output_directory, flush=True)


if __name__ == "__main__":
    main()
