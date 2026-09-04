"""Validate active fixed-profile quadratic CLs near the two exclusion contours."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from sterile_fit.adapter import _hypothesis_pairs, build_three_plus_one_analysis, load_analysis_selection
from sterile_fit.core.calibration import (
    _quadratic_difference_law,
    prepare_fixed_test_statistic,
    quadratic_cls,
    toy_cls,
)
from sterile_fit.core.profile_three_plus_one import (
    profile_s14_s24_at_fixed_sin2_2theta_ee,
    profile_s14_s24_at_fixed_sin2_2theta_mue,
)
from sterile_fit.core.three_plus_one import ThreePlusOneParameters
from sterile_fit.output import begin_output_batch, finish_output_batch, plot_statistic_calibration, result_directory, write_csv, write_json
from sterile_fit.scan import _stable_point_seed


def profile_observation(objective, mass: float, amplitude: float, mode: str):
    if mode == "appearance-profile":
        return profile_s14_s24_at_fixed_sin2_2theta_mue(
            objective, delta_m2_41_eV2=mass, sin2_2theta_mue=amplitude
        ).best_fit
    return profile_s14_s24_at_fixed_sin2_2theta_ee(
        objective, delta_m2_41_eV2=mass, sin2_2theta_ee=amplitude
    ).best_fit


def select_boundary_triplets(frame: pd.DataFrame, mode: str, mass_points: int) -> pd.DataFrame:
    amplitude_name = (
        "fixed_sin2_2theta_mue"
        if mode == "appearance-profile"
        else "fixed_sin2_2theta_ee"
    )
    spans = frame.groupby("fixed_delta_m2_41_eV2").cls_quadratic.agg(["min", "max"])
    masses = spans.index[(spans["min"] <= 0.05) & (spans["max"] >= 0.05)].to_numpy(dtype=float)
    if masses.size < 3:
        raise ValueError("fewer than three mass slices contain a quadratic CLs=0.05 crossing")
    targets = np.geomspace(masses.min(), masses.max(), mass_points)
    selected_masses = []
    for target in targets:
        candidate = masses[np.argmin(np.abs(np.log10(masses / target)))]
        if not selected_masses or candidate != selected_masses[-1]:
            selected_masses.append(candidate)
    rows = []
    for mass in selected_masses:
        group = frame[frame.fixed_delta_m2_41_eV2 == mass].sort_values(amplitude_name).reset_index(drop=True)
        center = int(np.argmin(np.abs(group.cls_quadratic.to_numpy() - 0.05)))
        center = min(max(center, 1), len(group) - 2)
        for position in (center - 1, center, center + 1):
            row = group.iloc[position]
            rows.append({
                "mass": float(mass),
                "amplitude": float(row[amplitude_name]),
                "scan_cls_quadratic": float(row.cls_quadratic),
            })
    return pd.DataFrame(rows)


def empirical_ks(sample: np.ndarray, law) -> float:
    ordered = np.sort(np.asarray(sample, dtype=float))
    candidate_cdf = 1.0 - law.survival_probabilities(ordered)
    n = ordered.size
    return float(max(
        np.max(np.arange(1, n + 1) / n - candidate_cdf),
        np.max(candidate_cdf - np.arange(n) / n),
    ))


def downward_crossing(amplitudes, q_values) -> float:
    x = np.log10(np.asarray(amplitudes, dtype=float))
    q = np.asarray(q_values, dtype=float)
    for index in range(len(x) - 1):
        if q[index] > 0.0 and q[index + 1] <= 0.0:
            fraction = q[index] / (q[index] - q[index + 1])
            return float(10.0 ** (x[index] + fraction * (x[index + 1] - x[index])))
    return float("nan")


def run_mode(scan_csv: Path, mode: str, toys: int, seed: int, mass_points: int, output: Path):
    frame = pd.read_csv(scan_csv)
    selected = select_boundary_triplets(frame, mode, mass_points)
    selection = load_analysis_selection(
        ROOT / "configs/analyses/microboone_bnb_numi.yaml", repository_root=ROOT
    )
    analysis = build_three_plus_one_analysis(selection, repository_root=ROOT)
    null = ThreePlusOneParameters(1.0, 0.0, 0.0)
    chi2_null = analysis.objective.chi2(null)
    rows = []
    for point_index, point in selected.iterrows():
        fit = profile_observation(analysis.objective.chi2, point.mass, point.amplitude, mode)
        tested = fit.parameters
        observed = fit.chi2 - chi2_null
        pairs = _hypothesis_pairs(analysis, null, tested)
        quadratic = quadratic_cls(observed, pairs)
        null_hypotheses = tuple(pair[0] for pair in pairs)
        tested_hypotheses = tuple(pair[1] for pair in pairs)
        empirical = toy_cls(
            observed,
            null_hypotheses,
            tested_hypotheses,
            prepare_fixed_test_statistic(null_hypotheses, tested_hypotheses),
            number_of_toys=toys,
            seed=_stable_point_seed(seed, int(point_index)),
            workers=1,
        )
        law_3nu = _quadratic_difference_law(pairs, generated_under_tested=False)
        law_4nu = _quadratic_difference_law(pairs, generated_under_tested=True)
        samples = {
            "3nu": empirical.test_statistics_under_3nu,
            "4nu": empirical.test_statistics_under_4nu,
        }
        panels = []
        for generator, law in (("3nu", law_3nu), ("4nu", law_4nu)):
            sample = samples[generator]
            low = min(sample.min(), law.mean - 4.5 * law.standard_deviation, observed)
            high = max(sample.max(), law.mean + 4.5 * law.standard_deviation, observed)
            grid = np.linspace(low, high, 900)
            survival = law.survival_probabilities(grid)
            density = np.maximum(0.0, -np.gradient(survival, grid))
            panels.append({
                "label": f"{generator} generation; N={toys}",
                "profiled_T": sample,
                "fixed_T": None,
                "toy_label": "Fixed-hypothesis Toy",
                "tail_label": "Fixed-hypothesis empirical tail",
                "residual_label": "Fixed Toy",
                "gaussian_residual_label": "Fixed Toy CDF - Gaussian CDF",
                "mean": law.mean,
                "sigma": law.standard_deviation,
                "observed_T": observed,
                "candidate": pd.DataFrame({"T": grid, "pdf": density, "sf": survival}),
            })
            write_csv(
                pd.DataFrame({
                    "point_index": int(point_index),
                    "generator": generator,
                    "toy_index": np.arange(toys),
                    "test_statistic": sample,
                }),
                output / "samples" / f"point_{point_index:02d}_{generator}.csv",
            )
        symbol = r"\mu e" if mode == "appearance-profile" else "ee"
        plot_statistic_calibration(
            panels,
            output / "distributions" / f"point_{point_index:02d}.png",
            title=(
                rf"$\Delta m^2_{{41}}={point.mass:.5g}\,\mathrm{{eV}}^2$, "
                rf"$\sin^2(2\theta_{{{symbol}}})={point.amplitude:.5g}$"
            ),
        )
        actual_amplitude = (
            tested.sin2_2theta_mue_exact
            if mode == "appearance-profile"
            else tested.sin2_2theta_ee_exact
        )
        rows.append({
            **point.to_dict(),
            "observed_T": observed,
            "profiled_sin2_theta14": tested.sin2_theta14,
            "profiled_sin2_theta24": tested.sin2_theta24,
            "profile_amplitude_error": actual_amplitude - point.amplitude,
            "p_3nu_quadratic": quadratic.p_value_3nu,
            "p_4nu_quadratic": quadratic.p_value_4nu,
            "cls_quadratic_recomputed": quadratic.cls,
            "p_3nu_toy": empirical.p_value_3nu,
            "p_4nu_toy": empirical.p_value_4nu,
            "cls_toy": empirical.cls,
            "cls_toy_minus_quadratic": empirical.cls - quadratic.cls,
            "ks_3nu": empirical_ks(empirical.test_statistics_under_3nu, law_3nu),
            "ks_4nu": empirical_ks(empirical.test_statistics_under_4nu, law_4nu),
            "ks_gaussian_3nu": empirical_ks(
                empirical.test_statistics_under_3nu,
                type("GaussianLaw", (), {"survival_probabilities": lambda self, x: norm.sf(x, law_3nu.mean, law_3nu.standard_deviation)})(),
            ),
            "ks_gaussian_4nu": empirical_ks(
                empirical.test_statistics_under_4nu,
                type("GaussianLaw", (), {"survival_probabilities": lambda self, x: norm.sf(x, law_4nu.mean, law_4nu.standard_deviation)})(),
            ),
            "toy_tail_count_3nu": empirical.right_tail_count_under_3nu,
            "toy_tail_count_4nu": empirical.right_tail_count_under_4nu,
        })
        print(
            f"{mode} point {point_index + 1}/{len(selected)}: dm2={point.mass:g}, "
            f"A={point.amplitude:.6g}, quadratic={quadratic.cls:.4g}, toy={empirical.cls:.4g}",
            flush=True,
        )
    results = pd.DataFrame(rows)
    write_csv(results, output / "point_comparison.csv")
    boundaries = []
    for mass, group in results.groupby("mass", sort=True):
        group = group.sort_values("amplitude")
        quadratic_boundary = downward_crossing(
            group.amplitude,
            group.p_4nu_quadratic - 0.05 * group.p_3nu_quadratic,
        )
        toy_boundary = downward_crossing(
            group.amplitude,
            group.p_4nu_toy - 0.05 * group.p_3nu_toy,
        )
        boundaries.append({
            "mass": mass,
            "quadratic_boundary": quadratic_boundary,
            "toy_boundary": toy_boundary,
            "toy_minus_quadratic_dex": np.log10(toy_boundary / quadratic_boundary)
            if np.isfinite(toy_boundary * quadratic_boundary) else np.nan,
            "toy_minus_quadratic_percent": 100.0 * (toy_boundary / quadratic_boundary - 1.0)
            if np.isfinite(toy_boundary * quadratic_boundary) else np.nan,
        })
    boundary_frame = pd.DataFrame(boundaries)
    write_csv(boundary_frame, output / "boundary_shift.csv")

    number_of_masses = results.mass.nunique()
    figure, axes = plt.subplots(
        1, number_of_masses, figsize=(4.5 * number_of_masses, 4.0), sharey=True,
        squeeze=False,
    )
    axes = axes[0]
    for axis, (mass, group) in zip(axes, results.groupby("mass", sort=True), strict=True):
        group = group.sort_values("amplitude")
        axis.plot(group.amplitude, group.cls_quadratic_recomputed, "o-", label="Quadratic form")
        axis.plot(group.amplitude, group.cls_toy, "s--", label=f"Fixed Toy, N={toys}")
        axis.axhline(0.05, color="tab:red", linewidth=1.5)
        axis.set_xscale("log")
        axis.set_title(rf"$\Delta m^2_{{41}}={mass:g}\,\mathrm{{eV}}^2$")
        axis.set_xlabel(
            r"$\sin^2(2\theta_{\mu e})$"
            if mode == "appearance-profile"
            else r"$\sin^2(2\theta_{ee})$"
        )
        axis.grid(alpha=0.2)
    axes[0].set_ylabel(r"$CL_s$")
    axes[0].legend(fontsize=8)
    figure.suptitle("Observed-data profile followed by fixed-hypothesis calibration")
    figure.tight_layout()
    figure.savefig(output / "boundary_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(figure)
    return results, boundary_frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fig3a", required=True, type=Path)
    parser.add_argument("--fig3b", required=True, type=Path)
    parser.add_argument("--toys", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20250821)
    parser.add_argument("--mass-points", type=int, default=5)
    parser.add_argument("--batch", required=True)
    arguments = parser.parse_args()
    begin_output_batch(arguments.batch)
    base = result_directory("studies", "active_quadratic_validation", "comparison")
    a_results, a_boundaries = run_mode(
        arguments.fig3a, "appearance-profile", arguments.toys, arguments.seed, arguments.mass_points, base / "fig3a"
    )
    b_results, b_boundaries = run_mode(
        arguments.fig3b, "electron-disappearance-profile", arguments.toys, arguments.seed + 1000, arguments.mass_points, base / "fig3b"
    )
    write_json(base / "metadata.json", {
        "method": "observed-data profile once, then fixed-hypothesis quadratic inversion versus fixed-hypothesis Toy MC",
        "toys_per_hypothesis_per_point": arguments.toys,
        "points_per_profile_mode": int(3 * arguments.mass_points),
        "representative_mass_slices": arguments.mass_points,
        "selection": "three adjacent scan amplitudes nearest quadratic CLs=0.05 at each mass",
        "maximum_absolute_profile_amplitude_error": float(max(
            a_results.profile_amplitude_error.abs().max(), b_results.profile_amplitude_error.abs().max()
        )),
    })
    finish_output_batch(sys.argv[1:])
    print(base)


if __name__ == "__main__":
    main()
