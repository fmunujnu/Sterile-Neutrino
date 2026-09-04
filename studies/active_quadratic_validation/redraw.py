"""Redraw saved active quadratic-validation Toys without regenerating them."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from sterile_fit.adapter import _hypothesis_pairs, build_three_plus_one_analysis, load_analysis_selection
from sterile_fit.core.calibration import _quadratic_difference_law
from sterile_fit.core.three_plus_one import ThreePlusOneParameters
from sterile_fit.output import plot_statistic_calibration
from studies.active_quadratic_validation.run import profile_observation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    arguments = parser.parse_args()
    selection = load_analysis_selection(
        ROOT / "configs/analyses/microboone_bnb_numi.yaml", repository_root=ROOT
    )
    analysis = build_three_plus_one_analysis(selection, repository_root=ROOT)
    null = ThreePlusOneParameters(1.0, 0.0, 0.0)
    for folder, mode in (("fig3a", "appearance-profile"), ("fig3b", "electron-disappearance-profile")):
        directory = arguments.source / folder
        points = pd.read_csv(directory / "point_comparison.csv")
        for point_index, point in points.iterrows():
            fit = profile_observation(analysis.objective.chi2, point.mass, point.amplitude, mode)
            pairs = _hypothesis_pairs(analysis, null, fit.parameters)
            panels = []
            for generator_index, generator in enumerate(("3nu", "4nu")):
                sample = pd.read_csv(
                    directory / "samples" / f"point_{point_index:02d}_{generator}.csv"
                ).test_statistic.to_numpy()
                law = _quadratic_difference_law(
                    pairs, generated_under_tested=bool(generator_index)
                )
                low = min(sample.min(), law.mean - 4.5 * law.standard_deviation, point.observed_T)
                high = max(sample.max(), law.mean + 4.5 * law.standard_deviation, point.observed_T)
                grid = np.linspace(low, high, 900)
                survival = law.survival_probabilities(grid)
                panels.append({
                    "label": f"{generator} generation; N={len(sample)}",
                    "profiled_T": sample,
                    "fixed_T": None,
                    "toy_label": "Fixed-hypothesis Toy",
                    "tail_label": "Fixed-hypothesis empirical tail",
                    "mean": law.mean,
                    "sigma": law.standard_deviation,
                    "observed_T": point.observed_T,
                    "candidate": pd.DataFrame({
                        "T": grid,
                        "pdf": np.maximum(0.0, -np.gradient(survival, grid)),
                        "sf": survival,
                    }),
                })
            symbol = r"\mu e" if mode == "appearance-profile" else "ee"
            plot_statistic_calibration(
                panels,
                directory / "distributions" / f"point_{point_index:02d}.png",
                title=(
                    rf"$\Delta m^2_{{41}}={point.mass:.5g}\,\mathrm{{eV}}^2$, "
                    rf"$\sin^2(2\theta_{{{symbol}}})={point.amplitude:.5g}$"
                ),
            )
            print(f"redrew {folder} point {point_index + 1}/{len(points)}", flush=True)


if __name__ == "__main__":
    main()
