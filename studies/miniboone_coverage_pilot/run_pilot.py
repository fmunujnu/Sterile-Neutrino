from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from sterile_fit.experiments.miniboone.official_2020 import (
    RAW,
    gaussian_negative_two_log_likelihood_from_signals,
    load_release,
    prediction_and_covariance_from_signals,
    signal_counts,
)
from sterile_fit.output import result_directory, write_csv, write_json


TARGETS = ((0.5, 0.05), (0.2, 0.08), (0.05, 0.15),
           (0.02, 0.3), (0.008, 0.45), (0.02, 0.5))


def _nearest_contour_points(contour: np.ndarray) -> np.ndarray:
    selected = []
    for amplitude, mass in TARGETS:
        distance = np.sum(
            (np.log10(contour) - np.log10([amplitude, mass])) ** 2, axis=1
        )
        selected.append(contour[np.argmin(distance)])
    return np.asarray(selected)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--toys", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=250)
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--output-directory", type=Path)
    args = parser.parse_args()
    if args.toys < 100 or args.batch_size < 1:
        parser.error("use at least 100 toys and a positive batch size")
    output = args.output_directory or result_directory(
        "studies_miniboone", "three_plus_one", "coverage_pilot"
    )
    output.mkdir(parents=True, exist_ok=False)

    started = time.perf_counter()
    data = load_release()
    official = pd.read_csv(RAW / "likihood_surface_contNunubar.txt", sep=r"\s+")
    masses = np.sort(official.dm2.unique())
    amplitudes = np.sort(official.sintheta.unique())
    contour = np.loadtxt(RAW / "cont_fake_oct19_contNunubar_90.txt")
    selected = _nearest_contour_points(contour)

    official_pivot = official.pivot(index="dm2", columns="sintheta", values="-2ln(L)")
    official_delta = official_pivot.to_numpy() - official_pivot.to_numpy().min()
    official_interpolator = RegularGridInterpolator(
        (np.log10(masses), np.log10(amplitudes)), official_delta,
        bounds_error=True,
    )

    unit_signals = []
    for mass in masses:
        unit_signals.append((
            signal_counts(data.nu_full_transmutation, data.electron_edges_MeV, mass, 1.0),
            signal_counts(data.nubar_full_transmutation, data.electron_edges_MeV, mass, 1.0),
        ))

    point_count = len(masses) * len(amplitudes)
    means = np.empty((point_count, 38), dtype=float)
    precisions = np.empty((point_count, 38, 38), dtype=float)
    logdets = np.empty(point_count, dtype=float)
    local_observed = np.empty(point_count, dtype=float)
    observation = None
    cursor = 0
    for mass_index, mass in enumerate(masses):
        unit_nu, unit_nubar = unit_signals[mass_index]
        for amplitude in amplitudes:
            obs, mean, covariance, _, _ = prediction_and_covariance_from_signals(
                data, amplitude * unit_nu, amplitude * unit_nubar
            )
            sign, logdet = np.linalg.slogdet(covariance)
            if sign <= 0:
                raise RuntimeError("non-positive covariance on released grid")
            precision = np.linalg.inv(covariance)
            means[cursor] = mean
            precisions[cursor] = precision
            logdets[cursor] = logdet
            residual = obs - mean
            local_observed[cursor] = residual @ precision @ residual + logdet
            observation = obs
            cursor += 1
    linear = np.einsum("pij,pj->pi", precisions, means, optimize=True)
    constants = np.einsum("pi,pi->p", means, linear) + logdets
    local_minimum = float(local_observed.min())
    print(f"Prepared {point_count} fit hypotheses in {time.perf_counter()-started:.1f} s", flush=True)

    rng = np.random.default_rng(args.seed)
    summaries = []
    toy_rows = []
    for point_index, (amplitude, mass) in enumerate(selected):
        unit_nu = signal_counts(
            data.nu_full_transmutation, data.electron_edges_MeV, mass, 1.0
        )
        unit_nubar = signal_counts(
            data.nubar_full_transmutation, data.electron_edges_MeV, mass, 1.0
        )
        _, true_mean, true_covariance, _, _ = prediction_and_covariance_from_signals(
            data, amplitude * unit_nu, amplitude * unit_nubar
        )
        true_precision = np.linalg.inv(true_covariance)
        true_cholesky = np.linalg.cholesky(true_covariance)
        true_sign, true_logdet = np.linalg.slogdet(true_covariance)
        if true_sign <= 0:
            raise RuntimeError("non-positive covariance at selected true point")
        fixed_observed = gaussian_negative_two_log_likelihood_from_signals(
            data, amplitude * unit_nu, amplitude * unit_nubar
        )
        toy_statistics = np.empty(args.toys, dtype=float)
        for start in range(0, args.toys, args.batch_size):
            stop = min(args.toys, start + args.batch_size)
            standard_normal = rng.standard_normal((stop-start, true_mean.size))
            toys = true_mean + standard_normal @ true_cholesky.T
            quadratic = np.einsum(
                "ti,pij,tj->tp", toys, precisions, toys, optimize=True
            )
            all_nll = quadratic - 2.0 * toys @ linear.T + constants
            best_grid_nll = np.min(all_nll, axis=1)
            true_residual = toys - true_mean
            fixed_nll = np.einsum(
                "ti,ij,tj->t", true_residual, true_precision, true_residual,
                optimize=True,
            ) + true_logdet
            # The selected official contour coordinate is not generally one
            # of the 190x190 released grid nodes. Include the fixed true point
            # itself in the fit domain so the likelihood-ratio statistic
            # cannot become negative solely because of grid discretisation.
            best_nll = np.minimum(best_grid_nll, fixed_nll)
            toy_statistics[start:stop] = fixed_nll - best_nll
        critical = float(np.quantile(toy_statistics, 0.90, method="higher"))
        bootstrap_rng = np.random.default_rng(args.seed + 1000 + point_index)
        bootstrap = np.quantile(
            bootstrap_rng.choice(toy_statistics, size=(2000, args.toys), replace=True),
            0.90, axis=1, method="higher",
        )
        official_required = float(official_interpolator(
            [[np.log10(mass), np.log10(amplitude)]]
        )[0])
        local_observed_delta = float(fixed_observed - local_minimum)
        summaries.append({
            "point": point_index,
            "sin2_2theta_mue": amplitude,
            "delta_m2_eV2": mass,
            "official_interpolated_observed_delta_nll": official_required,
            "local_observed_delta_nll": local_observed_delta,
            "fixed_chi2_2_90_threshold": 4.605170185988092,
            "toy_90_critical": critical,
            "toy_90_bootstrap_low": float(np.quantile(bootstrap, 0.025)),
            "toy_90_bootstrap_high": float(np.quantile(bootstrap, 0.975)),
            "absolute_error_fixed_vs_official": abs(4.605170185988092-official_required),
            "absolute_error_toy_vs_official": abs(critical-official_required),
        })
        toy_rows.extend({"point": point_index, "toy_index": i, "delta_nll": value}
                        for i, value in enumerate(toy_statistics))
        elapsed = time.perf_counter() - started
        remaining = elapsed / (point_index + 1) * (len(selected)-point_index-1)
        print(f"Point {point_index+1}/{len(selected)} done; elapsed {elapsed:.1f} s; estimated remaining {remaining:.1f} s", flush=True)

    summary = pd.DataFrame(summaries)
    write_csv(summary, output / "summary.csv", index=False)
    write_csv(pd.DataFrame(toy_rows), output / "toy_statistics.csv", index=False)
    write_json(output / "metadata.json", {
        "toys_per_point": args.toys,
        "selected_official_contour": "90% frequentist coverage",
        "fit_domain": "complete released 190x190 grid",
        "toy_generator": "38-dimensional Gaussian with point-dependent released covariance",
        "best_fit_definition": "minimum NLL over all 36100 grid hypotheses for each toy",
        "seed": args.seed,
        "elapsed_seconds": time.perf_counter()-started,
        "limitation": "finite Toy statistics and released-grid rather than continuous minimization",
    })

    figure, axis = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
    x = np.arange(len(summary))
    official_values = summary.official_interpolated_observed_delta_nll.to_numpy()
    toy_values = summary.toy_90_critical.to_numpy()
    axis.plot(x, official_values, "ko-", label="Official contour-required observed statistic")
    axis.axhline(4.605170185988092, color="tab:red", linestyle="--",
                label=r"Fixed $\chi^2_2$ 90% threshold")
    axis.errorbar(
        x, toy_values,
        yerr=[toy_values-summary.toy_90_bootstrap_low,
              summary.toy_90_bootstrap_high-toy_values],
        fmt="o-", color="tab:blue", capsize=3,
        label="Local full-grid Toy 90% critical value",
    )
    axis.set_xticks(x, [rf"{m:.3g} eV$^2$" for m in summary.delta_m2_eV2])
    axis.set_xlabel(r"Selected official 90% contour point: $\Delta m^2$")
    axis.set_ylabel(r"Critical $\Delta(-2\ln L)$")
    axis.set_title("MiniBooNE 90% coverage pilot")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    figure.savefig(output / "threshold_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(figure)
    print(output, flush=True)


if __name__ == "__main__":
    main()
