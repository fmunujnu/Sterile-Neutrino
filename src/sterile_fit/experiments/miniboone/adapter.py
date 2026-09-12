"""MiniBooNE 2020 public-release validation and two-flavour scan entry."""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from sterile_fit.experiments.miniboone.official_2020 import (
    RAW, gaussian_negative_two_log_likelihood_from_signals, load_release,
    prediction_and_covariance, signal_counts,
)
from sterile_fit.output import result_directory, write_csv, write_json


def run_miniboone() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("official", "scan"), default="official")
    parser.add_argument("--output-directory", type=Path)
    args = parser.parse_args()
    data = load_release()
    output = args.output_directory or result_directory(
        "miniboone_nu_nubar_combined", "two_flavour", args.kind
    )
    output.mkdir(parents=True, exist_ok=False)

    official = pd.read_csv(RAW / "likihood_surface_contNunubar.txt", sep=r"\s+")
    contours = {
        label: np.loadtxt(RAW / filename)
        for label, filename in {
            "1sigma": "cont_fake_oct19_contNunubar_1s.txt",
            "90percent": "cont_fake_oct19_contNunubar_90.txt",
            "99percent": "cont_fake_oct19_contNunubar_99.txt",
            "3sigma": "cont_fake_oct19_contNunubar_3s.txt",
        }.items()
    }
    if args.kind == "official":
        result = official.copy()
    else:
        masses = np.sort(official.dm2.unique())
        amplitudes = np.sort(official.sintheta.unique())
        rows = []
        for mass in masses:
            # P is exactly linear in the scanned amplitude. Histogramming the
            # 135k released events once per mass is algebraically identical to
            # repeating it for all 190 amplitudes.
            unit_nu = signal_counts(
                data.nu_full_transmutation, data.electron_edges_MeV, mass, 1.0
            )
            unit_nubar = signal_counts(
                data.nubar_full_transmutation, data.electron_edges_MeV, mass, 1.0
            )
            for amplitude in amplitudes:
                value = gaussian_negative_two_log_likelihood_from_signals(
                    data, amplitude * unit_nu, amplitude * unit_nubar
                )
                rows.append((mass, amplitude, value))
        result = pd.DataFrame(rows, columns=["dm2", "sintheta", "negative_2_log_likelihood"])
    write_csv(result, output / "result.csv", index=False)

    best = result.loc[result.iloc[:, 2].idxmin()]
    _, prediction, covariance, signal_nu, signal_nubar = prediction_and_covariance(
        data, float(best.iloc[0]), float(best.iloc[1])
    )
    write_csv(pd.DataFrame({
        "sample_bin": np.arange(len(prediction)),
        "observation": np.r_[data.nu_e_data, data.nu_mu_data,
                              data.nubar_e_data, data.nubar_mu_data],
        "prediction": prediction,
    }), output / "best_fit_prediction.csv", index=False)
    metadata = {
        "release": "MiniBooNE nue2020 combined, corrected 2021-02-23",
        "paper": "arXiv:2006.16883; Phys. Rev. D 103, 052002 (2021)",
        "mode": args.kind,
        "best_grid_point": {"delta_m2_eV2": float(best.iloc[0]),
                            "sin2_2theta": float(best.iloc[1]),
                            "statistic": float(best.iloc[2])},
        "signal_totals_at_selected_grid_point": {"nu_mode": float(signal_nu.sum()),
                          "nubar_mode": float(signal_nubar.sum())},
        "covariance_minimum_eigenvalue": float(np.linalg.eigvalsh(covariance).min()),
        "confidence_contours": "official released contours; coverage calibrated by MiniBooNE frequentist studies",
        "local_scan_statistic": "chi2 + log(det(covariance)); official likelihood surface is the validation reference",
    }
    if args.kind == "scan":
        official_pivot = official.pivot(index="dm2", columns="sintheta", values="-2ln(L)")
        local_pivot = result.pivot(
            index="dm2", columns="sintheta", values="negative_2_log_likelihood"
        )
        official_delta = official_pivot.to_numpy() - official_pivot.to_numpy().min()
        local_delta = local_pivot.to_numpy() - local_pivot.to_numpy().min()
        difference = local_delta - official_delta
        metadata["official_surface_comparison"] = {
            "pearson_correlation_of_delta_surfaces": float(
                np.corrcoef(official_delta.ravel(), local_delta.ravel())[0, 1]
            ),
            "root_mean_square_delta_difference": float(np.sqrt(np.mean(difference**2))),
            "maximum_absolute_delta_difference": float(np.max(np.abs(difference))),
            "note": "Grid-aligned diagnostic only; official confidence contours use collaboration coverage studies.",
        }
        metadata["three_way_contour_comparison"] = {
            "grid": {
                "delta_m2_points": int(official.dm2.nunique()),
                "sin2_2theta_mue_points": int(official.sintheta.nunique()),
                "total_points": int(len(official)),
            },
            "fixed_likelihood_thresholds": {
                "90_percent": 4.605,
                "99_percent": 9.210,
            },
            "curves": [
                "local Gaussian NLL fixed slices",
                "official released likelihood fixed slices",
                "official released frequentist calibrated contours",
            ],
            "toy_mc_in_local_scan": False,
        }
    write_json(output / "metadata.json", metadata)

    from sterile_fit.output import render_miniboone_parameter_space
    render_miniboone_parameter_space(result, contours, output / "parameter_space.png",
                                     official_surface=(args.kind == "official"))
    if args.kind == "scan":
        from sterile_fit.output import (
            render_miniboone_line_comparison,
            render_miniboone_three_way_comparison,
        )
        render_miniboone_line_comparison(
            result, contours, output / "parameter_space_line_overlay.png"
        )
        render_miniboone_three_way_comparison(
            result,
            official,
            contours,
            output / "three_way_comparison_heatmap.png",
            heatmap=True,
        )
        render_miniboone_three_way_comparison(
            result,
            official,
            contours,
            output / "three_way_comparison_lines.png",
            heatmap=False,
        )
    print(output)
