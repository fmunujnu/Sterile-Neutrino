"""MiniBooNE 2020 public-release validation and two-flavour scan entry."""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from sterile_fit.experiments.miniboone.official_2020 import (
    RAW, gaussian_negative_two_log_likelihood_from_signals, load_release,
    prediction_and_covariance, prediction_and_covariance_from_signals, signal_counts,
)
from sterile_fit.output import result_directory, write_csv, write_json


DEFAULT_REPROFILE_TOY_RESULT = (
    Path(__file__).resolve().parents[4]
    / "data/experiments/miniboone/shared/derived/reprofile_toy_10000/point_calibration.csv"
)


def _parse_mass_pairs(text: str) -> tuple[tuple[float, float], ...]:
    try:
        pairs = tuple(
            tuple(float(value) for value in item.split(":"))
            for item in text.split(",")
        )
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "mass pairs must look like 0.1:1,1:1,1:10"
        ) from error
    if not pairs or any(len(pair) != 2 or pair[0] <= 0.0 or pair[1] <= 0.0 for pair in pairs):
        raise argparse.ArgumentTypeError("each mass pair must contain two positive values")
    return pairs


def run_miniboone() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--kind",
        choices=("official", "scan", "toy", "one-plus-three-plus-one"),
        default="toy",
    )
    parser.add_argument("--toy-result", type=Path, default=DEFAULT_REPROFILE_TOY_RESULT)
    parser.add_argument("--output-directory", type=Path)
    parser.add_argument(
        "--mass-pairs",
        type=_parse_mass_pairs,
        default=_parse_mass_pairs("0.1:1,1:1,1:10"),
        help="1+3+1 fixed |dm2_41|:dm2_51 slices in eV^2",
    )
    parser.add_argument("--amplitude-min", type=float, default=3.0e-4)
    parser.add_argument("--amplitude-max", type=float, default=1.0)
    parser.add_argument("--amplitude-points", type=int, default=31)
    parser.add_argument("--phase-grid-points", type=int, default=25)
    parser.add_argument(
        "--one-plus-three-plus-one-mode",
        choices=("fixed-paper-masses", "profile-masses", "fixed-mass-slices"),
        default="fixed-paper-masses",
    )
    parser.add_argument("--mixing-product-min", type=float, default=1.0e-3)
    parser.add_argument("--mixing-product-max", type=float, default=1.5e-1)
    parser.add_argument("--mixing-product-points", type=int, default=61)
    parser.add_argument("--fixed-delta-m2-41-absolute", type=float, default=0.9)
    parser.add_argument("--fixed-delta-m2-51", type=float, default=0.5)
    parser.add_argument("--fixed-cp-phase", type=float, default=0.0)
    parser.add_argument("--mass-min", type=float, default=1.0e-2)
    parser.add_argument("--mass-max", type=float, default=1.0e2)
    parser.add_argument("--mass-profile-points", type=int, default=21)
    parser.add_argument("--coarse-phase-points", type=int, default=5)
    parser.add_argument("--refined-mass-candidates", type=int, default=10)
    parser.add_argument("--refined-phase-grid-points", type=int, default=13)
    args = parser.parse_args()
    data = load_release()
    if args.kind == "one-plus-three-plus-one":
        _run_one_plus_three_plus_one(data, args)
        return
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
    toy_result = None
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
        if args.kind == "toy":
            toy_result = pd.read_csv(args.toy_result)
            expected_columns = {
                "tested_point_index", "delta_m2_eV2", "sin2_2theta_mue",
                "toys", "observed_profiled_delta_nll", "toy_90_critical",
                "toy_95_critical", "toy_99_critical",
            }
            missing = expected_columns.difference(toy_result.columns)
            if missing:
                raise ValueError(f"Toy result is missing columns: {sorted(missing)}")
            if len(toy_result) != len(result) or toy_result["tested_point_index"].nunique() != len(result):
                raise ValueError("Toy result must contain every point of the 190x190 MiniBooNE grid exactly once")
            toy_result = toy_result.sort_values("tested_point_index").reset_index(drop=True)
            if not (
                np.allclose(toy_result["delta_m2_eV2"], result["dm2"], rtol=1e-12, atol=0.0)
                and np.allclose(toy_result["sin2_2theta_mue"], result["sintheta"], rtol=1e-12, atol=0.0)
            ):
                raise ValueError("Toy result coordinates do not match the active MiniBooNE grid ordering")
            write_csv(toy_result, output / "toy_calibration.csv", index=False)
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
    if args.kind in {"scan", "toy"}:
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
            "toy_mc_in_local_scan": toy_result is not None,
        }
        if toy_result is not None:
            metadata["reprofile_toy_calibration"] = {
                "source": str(args.toy_result),
                "toys_per_tested_point": int(toy_result["toys"].iloc[0]),
                "profile": "each Toy is independently minimized over the full released 190x190 grid",
                "status": "primary local MiniBooNE calibration; not the collaboration's internal calibration",
            }
    write_json(output / "metadata.json", metadata)

    from sterile_fit.output import render_miniboone_parameter_space
    render_miniboone_parameter_space(result, contours, output / "parameter_space.png",
                                     official_surface=(args.kind == "official"))
    if args.kind in {"scan", "toy"}:
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
            toy_result=toy_result,
        )
        render_miniboone_three_way_comparison(
            result,
            official,
            contours,
            output / "three_way_comparison_lines.png",
            heatmap=False,
            toy_result=toy_result,
        )
    print(output)


def _run_one_plus_three_plus_one(data, args: argparse.Namespace) -> None:
    """Run the declared no-Toy, appearance-only 1+3+1 scan."""
    if args.one_plus_three_plus_one_mode == "fixed-paper-masses":
        _run_fixed_mass_product_plane(data, args)
        return
    if not 0.0 < args.amplitude_min < args.amplitude_max <= 1.0:
        raise ValueError("amplitude limits must satisfy 0 < min < max <= 1")
    if args.amplitude_points < 3:
        raise ValueError("amplitude-points must be at least 3")
    if args.one_plus_three_plus_one_mode == "profile-masses":
        _run_profiled_one_plus_three_plus_one_mixing_plane(data, args)
        return
    from sterile_fit.experiments.miniboone.one_plus_three_plus_one import (
        build_fixed_mass_templates,
        profile_cp_phase,
        scan_fixed_mass_slices,
    )
    from sterile_fit.output import render_miniboone_one_plus_three_plus_one_slices

    output = args.output_directory or result_directory(
        "miniboone_nu_nubar_combined",
        "one_plus_three_plus_one",
        "gaussian_nll_fixed_mass_slices",
    )
    output.mkdir(parents=True, exist_ok=False)
    amplitudes = np.geomspace(
        args.amplitude_min, args.amplitude_max, args.amplitude_points
    )
    result = scan_fixed_mass_slices(
        data,
        args.mass_pairs,
        amplitudes,
        phase_grid_points=args.phase_grid_points,
    )
    write_csv(result, output / "result.csv", index=False)

    best_rows = []
    prediction_rows = []
    for q41, q51 in args.mass_pairs:
        selected = result[
            np.isclose(result["delta_m2_41_absolute_eV2"], q41)
            & np.isclose(result["delta_m2_51_eV2"], q51)
        ]
        best = selected.loc[selected["negative_2_log_likelihood"].idxmin()]
        best_rows.append(best)
        templates = build_fixed_mass_templates(data, q41, q51)
        phase, _, signal_nu, signal_nubar = profile_cp_phase(
            data,
            templates,
            float(best["appearance_amplitude_state4"]),
            float(best["appearance_amplitude_state5"]),
            phase_grid_points=args.phase_grid_points,
        )
        observation, prediction, _, _, _ = prediction_and_covariance_from_signals(
            data, signal_nu, signal_nubar
        )
        prediction_rows.append(pd.DataFrame({
            "delta_m2_41_absolute_eV2": q41,
            "delta_m2_51_eV2": q51,
            "profiled_cp_phase_mue_rad": phase,
            "sample_bin": np.arange(len(prediction)),
            "observation": observation,
            "prediction": prediction,
        }))
    write_csv(pd.DataFrame(best_rows), output / "best_fit_by_mass_slice.csv", index=False)
    write_csv(pd.concat(prediction_rows, ignore_index=True),
              output / "best_fit_predictions.csv", index=False)

    render_miniboone_one_plus_three_plus_one_slices(
        result, output / "fixed_mass_slices_heatmap.png", heatmap=True
    )
    render_miniboone_one_plus_three_plus_one_slices(
        result, output / "fixed_mass_slices_lines.png", heatmap=False
    )
    for q41, q51 in args.mass_pairs:
        selected = result[
            np.isclose(result["delta_m2_41_absolute_eV2"], q41)
            & np.isclose(result["delta_m2_51_eV2"], q51)
        ]
        label = f"q41_{q41:g}_q51_{q51:g}".replace(".", "p")
        render_miniboone_one_plus_three_plus_one_slices(
            selected, output / f"{label}_heatmap.png", heatmap=True
        )
        render_miniboone_one_plus_three_plus_one_slices(
            selected, output / f"{label}_lines.png", heatmap=False
        )

    write_json(output / "metadata.json", {
        "release": "MiniBooNE nue2020 combined, corrected 2021-02-23",
        "paper": "arXiv:2006.16883; Phys. Rev. D 103, 052002 (2021)",
        "model": "1+3+1 appearance-only short-baseline vacuum model",
        "mass_ordering": "delta_m2_41 < 0 < delta_m2_51",
        "fixed_mass_pairs_eV2": [
            {"abs_delta_m2_41": q41, "delta_m2_51": q51}
            for q41, q51 in args.mass_pairs
        ],
        "scan_coordinates": {
            "A4": "4*|Ue4|^2*|Umu4|^2",
            "A5": "4*|Ue5|^2*|Umu5|^2",
            "minimum": args.amplitude_min,
            "maximum": args.amplitude_max,
            "points_per_axis": args.amplitude_points,
            "scale": "logarithmic",
        },
        "profiled_parameter": "cp_phase_mue_rad",
        "phase_profile": {
            "coarse_grid_points": args.phase_grid_points,
            "continuous_local_refinement": True,
            "range_rad": [-float(np.pi), float(np.pi)],
            "unitary_embedding": "symmetric minimum-row-norm representative enforced",
        },
        "statistic": "chi2 + log(det(covariance)); same 38-bin MiniBooNE local Gaussian NLL as the validated 3+1 scan",
        "calibration": {
            "toy_mc": False,
            "thresholds": {"90_percent_two_coordinate": 4.605,
                           "99_percent_two_coordinate": 9.210},
            "interpretation": "fixed-threshold asymptotic contours relative to the minimum of each fixed-mass slice; not official MiniBooNE coverage",
        },
        "scope": "Only the released full-transmutation appearance events are reweighted. Published backgrounds and muon-control predictions remain unchanged because the public package does not decompose them for 1+3+1 disappearance reweighting.",
    })
    print(output)


def _run_fixed_mass_product_plane(data, args: argparse.Namespace) -> None:
    """Reproduce the fixed-mass, CP-conserving product-coordinate scan."""
    if not 0.0 < args.mixing_product_min < args.mixing_product_max <= 0.5:
        raise ValueError("mixing-product limits must satisfy 0 < min < max <= 0.5")
    if args.mixing_product_points < 3:
        raise ValueError("mixing-product-points must be at least 3")
    if args.fixed_delta_m2_41_absolute <= 0.0 or args.fixed_delta_m2_51 <= 0.0:
        raise ValueError("fixed mass-splitting magnitudes must be positive")
    from sterile_fit.experiments.miniboone.one_plus_three_plus_one import (
        build_fixed_mass_templates,
        scan_fixed_mass_product_plane,
        signals_from_templates,
    )
    from sterile_fit.output import render_miniboone_fixed_mass_product_plane

    output = args.output_directory or result_directory(
        "miniboone_nu_nubar_combined",
        "one_plus_three_plus_one",
        "chi_square_fixed_masses_cp_zero",
    )
    output.mkdir(parents=True, exist_ok=False)
    products = np.geomspace(
        args.mixing_product_min,
        args.mixing_product_max,
        args.mixing_product_points,
    )
    result = scan_fixed_mass_product_plane(
        data,
        products,
        delta_m2_41_absolute_eV2=args.fixed_delta_m2_41_absolute,
        delta_m2_51_eV2=args.fixed_delta_m2_51,
        cp_phase_mue_rad=args.fixed_cp_phase,
    )
    write_csv(result, output / "result.csv", index=False)
    best = result.loc[result["chi_square"].idxmin()]
    templates = build_fixed_mass_templates(
        data,
        args.fixed_delta_m2_41_absolute,
        args.fixed_delta_m2_51,
    )
    signal_nu, signal_nubar = signals_from_templates(
        templates,
        float(best["appearance_amplitude_state4"]),
        float(best["appearance_amplitude_state5"]),
        args.fixed_cp_phase,
    )
    observation, prediction, covariance, _, _ = prediction_and_covariance_from_signals(
        data, signal_nu, signal_nubar
    )
    write_csv(pd.DataFrame({
        "sample_bin": np.arange(len(prediction)),
        "observation": observation,
        "prediction": prediction,
    }), output / "best_fit_prediction.csv", index=False)
    write_csv(pd.DataFrame([best]), output / "best_fit.csv", index=False)
    render_miniboone_fixed_mass_product_plane(
        result, output / "fixed_mass_product_plane_heatmap.png", heatmap=True,
        delta_m2_41_absolute_eV2=args.fixed_delta_m2_41_absolute,
        delta_m2_51_eV2=args.fixed_delta_m2_51,
        cp_phase_mue_rad=args.fixed_cp_phase,
    )
    render_miniboone_fixed_mass_product_plane(
        result, output / "fixed_mass_product_plane_lines.png", heatmap=False,
        delta_m2_41_absolute_eV2=args.fixed_delta_m2_41_absolute,
        delta_m2_51_eV2=args.fixed_delta_m2_51,
        cp_phase_mue_rad=args.fixed_cp_phase,
    )
    write_json(output / "metadata.json", {
        "release": "MiniBooNE nue2020 combined, corrected 2021-02-23",
        "model": "1+3+1 appearance-only short-baseline vacuum model",
        "fixed_parameters": {
            "delta_m2_41_eV2": -float(args.fixed_delta_m2_41_absolute),
            "delta_m2_51_eV2": float(args.fixed_delta_m2_51),
            "cp_phase_mue_rad": float(args.fixed_cp_phase),
        },
        "scan_coordinates": {
            "x": "|Ue4 Umu4|",
            "y": "|Ue5 Umu5|",
            "minimum": args.mixing_product_min,
            "maximum": args.mixing_product_max,
            "points_per_axis": args.mixing_product_points,
            "scale": "logarithmic",
        },
        "profiled_unidentifiable_coordinates": {
            "individual_matrix_element_factorisations": "analytically flat in the released appearance-only likelihood",
            "effect_on_chi_square": "none",
            "additional_nuisance_parameters": "none are exposed by the active public MiniBooNE likelihood",
        },
        "statistic": "chi2 = (data-prediction)^T V(prediction)^-1 (data-prediction)",
        "covariance_minimum_eigenvalue_at_best_fit": float(
            np.linalg.eigvalsh(covariance).min()
        ),
        "calibration": {
            "toy_mc": False,
            "thresholds": {"90_percent_two_coordinate": 4.605,
                           "99_percent_two_coordinate": 9.210},
            "interpretation": "fixed two-coordinate chi-square thresholds; not official MiniBooNE coverage",
        },
        "scope": "Only full-transmutation appearance events are reweighted; backgrounds and muon-control samples keep their released treatment.",
    })
    print(output)


def _run_profiled_one_plus_three_plus_one_mixing_plane(
    data, args: argparse.Namespace
) -> None:
    """Scan the two appearance amplitudes and profile both masses and CP phase."""
    if not 0.0 < args.mass_min < args.mass_max:
        raise ValueError("mass limits must satisfy 0 < min < max")
    if args.mass_profile_points < 2:
        raise ValueError("mass-profile-points must be at least 2")
    from sterile_fit.experiments.miniboone.one_plus_three_plus_one import (
        build_fixed_mass_templates,
        scan_profiled_mixing_plane,
        signals_from_templates,
    )
    from sterile_fit.output import render_miniboone_profiled_mixing_plane

    output = args.output_directory or result_directory(
        "miniboone_nu_nubar_combined",
        "one_plus_three_plus_one",
        "chi_square_profiled_mixing_plane",
    )
    output.mkdir(parents=True, exist_ok=False)
    amplitudes = np.geomspace(
        args.amplitude_min, args.amplitude_max, args.amplitude_points
    )
    masses = np.geomspace(args.mass_min, args.mass_max, args.mass_profile_points)
    result = scan_profiled_mixing_plane(
        data,
        amplitudes,
        masses,
        coarse_phase_points=args.coarse_phase_points,
        refined_mass_candidates=args.refined_mass_candidates,
        refined_phase_grid_points=args.refined_phase_grid_points,
    )
    write_csv(result, output / "result.csv", index=False)

    best = result.loc[result["chi_square"].idxmin()]
    templates = build_fixed_mass_templates(
        data,
        float(best["profiled_delta_m2_41_absolute_eV2"]),
        float(best["profiled_delta_m2_51_eV2"]),
    )
    signal_nu, signal_nubar = signals_from_templates(
        templates,
        float(best["appearance_amplitude_state4"]),
        float(best["appearance_amplitude_state5"]),
        float(best["profiled_cp_phase_mue_rad"]),
    )
    observation, prediction, covariance, _, _ = prediction_and_covariance_from_signals(
        data, signal_nu, signal_nubar
    )
    write_csv(pd.DataFrame({
        "sample_bin": np.arange(len(prediction)),
        "observation": observation,
        "prediction": prediction,
    }), output / "best_fit_prediction.csv", index=False)
    write_csv(pd.DataFrame([best]), output / "best_fit.csv", index=False)
    render_miniboone_profiled_mixing_plane(
        result, output / "profiled_mixing_plane_heatmap.png", heatmap=True
    )
    render_miniboone_profiled_mixing_plane(
        result, output / "profiled_mixing_plane_lines.png", heatmap=False
    )
    write_json(output / "metadata.json", {
        "release": "MiniBooNE nue2020 combined, corrected 2021-02-23",
        "model": "1+3+1 appearance-only short-baseline vacuum model",
        "scan_coordinates": {
            "A4": "4*|Ue4|^2*|Umu4|^2",
            "A5": "4*|Ue5|^2*|Umu5|^2",
            "minimum": args.amplitude_min,
            "maximum": args.amplitude_max,
            "points_per_axis": args.amplitude_points,
            "scale": "logarithmic",
        },
        "profiled_parameters": {
            "abs_delta_m2_41_eV2": {
                "minimum": args.mass_min,
                "maximum": args.mass_max,
                "grid_points": args.mass_profile_points,
                "scale": "logarithmic",
            },
            "delta_m2_51_eV2": {
                "minimum": args.mass_min,
                "maximum": args.mass_max,
                "grid_points": args.mass_profile_points,
                "scale": "logarithmic",
            },
            "cp_phase_mue_rad": {
                "range": [-float(np.pi), float(np.pi)],
                "coarse_points": args.coarse_phase_points,
                "continuous_refinement_for_best_mass_candidates": True,
                "refined_mass_candidates": args.refined_mass_candidates,
                "refined_phase_grid_points": args.refined_phase_grid_points,
            },
        },
        "mass_ordering": "delta_m2_41 < 0 < delta_m2_51",
        "statistic": "chi2 = (data-prediction)^T V(prediction)^-1 (data-prediction); log(det(V)) deliberately omitted for this first chi-square study",
        "covariance_minimum_eigenvalue_at_best_fit": float(
            np.linalg.eigvalsh(covariance).min()
        ),
        "calibration": {
            "toy_mc": False,
            "thresholds": {"90_percent_two_coordinate": 4.605,
                           "99_percent_two_coordinate": 9.210},
            "interpretation": "two-coordinate fixed chi-square thresholds after discrete mass-grid and continuous CP-phase profiling; not official MiniBooNE coverage",
        },
        "scope": "Only full-transmutation appearance events are reweighted; published backgrounds and muon-control samples remain unchanged.",
    })
    print(output)
