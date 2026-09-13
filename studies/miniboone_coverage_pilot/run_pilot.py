"""Non-official full-grid MiniBooNE Toy coverage calibration.

For each tested parameter point, generate Toy data under that point and
reprofile every Toy over the complete released 190 x 190 parameter grid.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from sterile_fit.experiments.miniboone.official_2020 import (
    RAW,
    load_release,
    prediction_and_covariance_from_signals,
    signal_counts,
)
from sterile_fit.output import result_directory, write_json

DEFAULT_TOYS_PER_POINT = 10_000
DEFAULT_BATCH_SIZE = 100
REPRESENTATIVE_TARGETS = (
    (0.5, 0.05),
    (0.2, 0.08),
    (0.05, 0.15),
    (0.02, 0.3),
    (0.008, 0.45),
    (0.02, 0.5),
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Non-official MiniBooNE per-Toy full-grid reprofile calibration."
    )
    parser.add_argument("--toys", type=int, default=DEFAULT_TOYS_PER_POINT)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--seed", type=int, default=20260913)
    parser.add_argument("--start-point", type=int, default=0)
    parser.add_argument("--stop-point", type=int)
    parser.add_argument(
        "--selection",
        choices=("full", "representative-90"),
        default="full",
        help="Use the full grid or six points spanning the official 90%% contour.",
    )
    parser.add_argument("--output-directory", type=Path)
    arguments = parser.parse_args()
    if arguments.toys < 1 or arguments.batch_size < 1:
        parser.error("--toys and --batch-size must be positive")
    if arguments.start_point < 0:
        parser.error("--start-point must be non-negative")
    return arguments


def _prepare_profile_grid(data, masses: np.ndarray, amplitudes: np.ndarray):
    """Prepare the common grid used for observed and per-Toy profiling."""
    point_count = len(masses) * len(amplitudes)
    means = np.empty((point_count, 38), dtype=float)
    precisions = np.empty((point_count, 38, 38), dtype=float)
    log_determinants = np.empty(point_count, dtype=float)
    observed_nll = np.empty(point_count, dtype=float)
    unit_signals = []
    cursor = 0

    for mass in masses:
        unit_nu = signal_counts(
            data.nu_full_transmutation, data.electron_edges_MeV, mass, 1.0
        )
        unit_nubar = signal_counts(
            data.nubar_full_transmutation, data.electron_edges_MeV, mass, 1.0
        )
        unit_signals.append((unit_nu, unit_nubar))
        for amplitude in amplitudes:
            observation, mean, covariance, _, _ = prediction_and_covariance_from_signals(
                data, amplitude * unit_nu, amplitude * unit_nubar
            )
            sign, log_determinant = np.linalg.slogdet(covariance)
            if sign <= 0:
                raise RuntimeError("non-positive covariance on released grid")
            precision = np.linalg.inv(covariance)
            residual = observation - mean
            means[cursor] = mean
            precisions[cursor] = precision
            log_determinants[cursor] = log_determinant
            observed_nll[cursor] = residual @ precision @ residual + log_determinant
            cursor += 1

    linear_terms = np.einsum("pij,pj->pi", precisions, means, optimize=True)
    constants = np.einsum("pi,pi->p", means, linear_terms, optimize=True) + log_determinants
    return unit_signals, means, precisions, linear_terms, constants, observed_nll


def _append_row(path: Path, row: dict[str, object]) -> None:
    """Save each completed point immediately for interrupted long jobs."""
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)
        stream.flush()


def _representative_indices(
    masses: np.ndarray, amplitudes: np.ndarray
) -> list[int]:
    contour = np.loadtxt(RAW / "cont_fake_oct19_contNunubar_90.txt")
    indices = []
    for target_amplitude, target_mass in REPRESENTATIVE_TARGETS:
        distance = np.sum(
            (np.log10(contour) - np.log10([target_amplitude, target_mass])) ** 2,
            axis=1,
        )
        amplitude, mass = contour[np.argmin(distance)]
        mass_index = int(np.argmin(np.abs(np.log(masses) - np.log(mass))))
        amplitude_index = int(
            np.argmin(np.abs(np.log(amplitudes) - np.log(amplitude)))
        )
        index = mass_index * len(amplitudes) + amplitude_index
        if index not in indices:
            indices.append(index)
    return indices


def main() -> None:
    arguments = _arguments()
    data = load_release()
    official = pd.read_csv(RAW / "likihood_surface_contNunubar.txt", sep=r"\s+")
    masses = np.sort(official.dm2.unique())
    amplitudes = np.sort(official.sintheta.unique())
    total_points = len(masses) * len(amplitudes)
    stop_point = total_points if arguments.stop_point is None else arguments.stop_point
    if arguments.selection == "full":
        if not arguments.start_point < stop_point <= total_points:
            raise ValueError(f"require 0 <= start-point < stop-point <= {total_points}")
        tested_indices = list(range(arguments.start_point, stop_point))
    else:
        if arguments.start_point != 0 or arguments.stop_point is not None:
            raise ValueError(
                "start/stop-point cannot be combined with representative-90"
            )
        tested_indices = _representative_indices(masses, amplitudes)

    output = arguments.output_directory or result_directory(
        "studies_miniboone", "three_plus_one", "full_grid_reprofile_toy"
    )
    output.mkdir(parents=True, exist_ok=False)
    summary_path = output / "point_calibration.csv"
    started = time.perf_counter()

    unit_signals, means, precisions, linear_terms, constants, observed_nll = (
        _prepare_profile_grid(data, masses, amplitudes)
    )

    # First profile: minimize the observed-data NLL over the complete grid.
    observed_best_nll = float(np.min(observed_nll))
    observed_best_index = int(np.argmin(observed_nll))
    completed = 0

    for selection_index, tested_index in enumerate(tested_indices):
        mass_index, amplitude_index = divmod(tested_index, len(amplitudes))
        mass = float(masses[mass_index])
        amplitude = float(amplitudes[amplitude_index])
        unit_nu, unit_nubar = unit_signals[mass_index]
        _, tested_mean, tested_covariance, _, _ = prediction_and_covariance_from_signals(
            data, amplitude * unit_nu, amplitude * unit_nubar
        )
        cholesky = np.linalg.cholesky(tested_covariance)

        # Independent deterministic stream per tested point keeps split jobs identical.
        rng = np.random.default_rng(
            np.random.SeedSequence([arguments.seed, tested_index])
        )
        toy_statistics = np.empty(arguments.toys, dtype=float)

        for start in range(0, arguments.toys, arguments.batch_size):
            stop = min(arguments.toys, start + arguments.batch_size)
            toys = tested_mean + rng.standard_normal(
                (stop - start, tested_mean.size)
            ) @ cholesky.T

            # Second profile: independently minimize every Toy over all 36100 points.
            quadratic_terms = np.einsum(
                "ti,pij,tj->tp", toys, precisions, toys, optimize=True
            )
            all_grid_nll = quadratic_terms - 2.0 * toys @ linear_terms.T + constants
            toy_statistics[start:stop] = (
                all_grid_nll[:, tested_index] - np.min(all_grid_nll, axis=1)
            )

        row = {
            "tested_point_index": tested_index,
            "delta_m2_eV2": mass,
            "sin2_2theta_mue": amplitude,
            "toys": arguments.toys,
            "observed_profiled_delta_nll": float(
                observed_nll[tested_index] - observed_best_nll
            ),
            "toy_90_critical": float(np.quantile(toy_statistics, 0.90, method="higher")),
            "toy_95_critical": float(np.quantile(toy_statistics, 0.95, method="higher")),
            "toy_99_critical": float(np.quantile(toy_statistics, 0.99, method="higher")),
            "toy_mean": float(np.mean(toy_statistics)),
            "toy_standard_deviation": float(np.std(toy_statistics, ddof=1)),
        }
        _append_row(summary_path, row)
        completed += 1
        elapsed = time.perf_counter() - started
        remaining = elapsed / completed * (len(tested_indices) - selection_index - 1)
        print(
            f"selected point {selection_index + 1}/{len(tested_indices)} "
            f"(grid index {tested_index}); elapsed={elapsed:.1f}s; "
            f"remaining~{remaining:.1f}s",
            flush=True,
        )

    write_json(output / "metadata.json", {
        "scientific_status": "non_official_reprofile_toy_study",
        "toys_per_tested_parameter_point": arguments.toys,
        "default_toys_per_point": DEFAULT_TOYS_PER_POINT,
        "selection": arguments.selection,
        "tested_point_indices": tested_indices,
        "complete_grid_points": total_points,
        "grid_shape": [len(masses), len(amplitudes)],
        "observed_profile": {
            "domain": "complete released 190x190 grid",
            "best_grid_index": observed_best_index,
            "best_nll": observed_best_nll,
        },
        "toy_generator": (
            "38-dimensional Gaussian at each tested point using its "
            "point-dependent released covariance"
        ),
        "toy_reprofile": (
            "each Toy independently minimizes NLL over the complete released "
            "190x190 grid"
        ),
        "seed_strategy": "SeedSequence([master_seed, flattened_tested_point_index])",
        "master_seed": arguments.seed,
        "limitation": (
            "research implementation using the public Gaussian model and finite "
            "released grid; not the collaboration's internal calibration"
        ),
        "elapsed_seconds": time.perf_counter() - started,
    })
    print(output, flush=True)


if __name__ == "__main__":
    main()
