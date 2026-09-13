"""Full Fig.3b grid with per-Toy constrained reprofile (research calibration)."""
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


MASS_POINTS = 74
AMPLITUDE_POINTS = 61
TOTAL_POINTS = MASS_POINTS * AMPLITUDE_POINTS
MASS_GRID = np.geomspace(0.1, 40.0, MASS_POINTS)
AMPLITUDE_GRID = np.geomspace(0.01, 1.0, AMPLITUDE_POINTS)


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
    parser.add_argument("--toys", type=int, default=100,
                        help="Toy count under each of the 3nu and 4nu generating hypotheses")
    parser.add_argument("--seed", type=int, default=20260913)
    parser.add_argument("--start-point", type=int, default=0)
    parser.add_argument("--stop-point", type=int, default=TOTAL_POINTS)
    parser.add_argument("--output-directory", type=Path, required=True)
    result = parser.parse_args()
    if result.toys < 2:
        parser.error("--toys must be at least 2 per generating hypothesis")
    if not 0 <= result.start_point < result.stop_point <= TOTAL_POINTS:
        parser.error(f"require 0 <= start-point < stop-point <= {TOTAL_POINTS}")
    return result


def main() -> None:
    args = arguments()
    args.output_directory.mkdir(parents=True, exist_ok=False)
    summary_path = args.output_directory / "point_calibration.csv"
    toy_path = args.output_directory / "toy_statistics.csv"
    initialize_worker()
    started = perf_counter()
    selected = range(args.start_point, args.stop_point)

    for completed, point_index in enumerate(selected, start=1):
        mass_index, amplitude_index = divmod(point_index, AMPLITUDE_POINTS)
        mass = float(MASS_GRID[mass_index])
        amplitude = float(AMPLITUDE_GRID[amplitude_index])
        seed = _stable_point_seed(args.seed, point_index)
        summary, toy_rows = evaluate((
            "fig3b", "electron-disappearance-profile",
            mass, amplitude, args.toys, seed,
        ))
        summary["tested_point_index"] = point_index
        append_row(summary_path, summary)
        for row in toy_rows:
            row["tested_point_index"] = point_index
            append_row(toy_path, row)

        elapsed = perf_counter() - started
        remaining = elapsed / completed * ((args.stop_point - args.start_point) - completed)
        print(
            f"point {completed}/{args.stop_point - args.start_point} "
            f"(global {point_index}); elapsed={elapsed:.1f}s; ETA~{remaining:.1f}s",
            flush=True,
        )

    write_json(args.output_directory / "metadata.json", {
        "scientific_status": "research_full_grid_per_toy_reprofile",
        "figure": "MicroBooNE Fig.3b public-data approximation",
        "mass_grid_eV2": {"minimum": 0.1, "maximum": 40.0, "points": MASS_POINTS},
        "sin2_2theta_ee_grid": {"minimum": 0.01, "maximum": 1.0, "points": AMPLITUDE_POINTS},
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
        "limitation": (
            "100 Toys per generating hypothesis is a workflow/qualitative contour "
            "check; it is not a stable 0.05-tail calibration"
        ),
    })
    print(args.output_directory, flush=True)


if __name__ == "__main__":
    main()
