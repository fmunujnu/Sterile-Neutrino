"""Reuse archived profiled Toys; replay only Gaussian draws for fixed-T control."""
from __future__ import annotations
import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy.linalg import cholesky
from scipy.stats import norm, kstest, beta

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from sterile_fit.adapter import load_analysis_selection, build_three_plus_one_analysis, _hypothesis_pairs
from sterile_fit.core.three_plus_one import ThreePlusOneParameters
from sterile_fit.core.calibration import asymptotic_cls, _draw_gaussian_toys, prepare_fixed_hypothesis_chi2
from sterile_fit.scan import _stable_point_seed
from sterile_fit.output import begin_output_batch, result_directory, finish_output_batch, write_csv, write_json, plot_gaussian_toy_grid, plot_statistic_calibration


def render_comparison(samples, scores, destination, layout="individual"):
    """Only saved statistics enter this rendering adapter."""
    if layout == "combined":
        plot_gaussian_toy_grid(samples, scores, destination/"all_points_gaussian_vs_toy.png")
        return
    for point, group in samples.groupby("point_index", sort=True):
        panels = []
        for i, generator in enumerate(("3nu", "4nu")):
            values = group[group.generator == generator]
            score = scores[(scores.point_index == point) & (scores.generator == generator) & (scores.variant == "reprofiled")].iloc[0]
            panels.append(dict(label=rf"Generated under ${i+3}\nu$; N={len(values):,}",
                profiled_T=values.test_statistic.to_numpy(), fixed_T=values.fixed_T_same_draw.to_numpy(),
                mean=score.gaussian_mean, sigma=score.gaussian_sigma, observed_T=score.observed_T))
        plot_statistic_calibration(panels, destination/f"point_{point:02d}_gaussian_vs_toy.png",
            title=rf"Point {point}: $\Delta m^2_{{41}}={score.delta_m2_41_eV2:.5g}$ eV$^2$, $\sin^2(2\theta_{{\mu e}})={score.sin2_2theta_mue:.5g}$")
    deviations = scores[scores.variant == "reprofiled"].copy()
    deviations["mean_bias_Toy_minus_Gaussian"] = deviations.toy_mean - deviations.gaussian_mean
    deviations["sigma_ratio_Toy_over_Gaussian"] = deviations.toy_sigma / deviations.gaussian_sigma
    deviations["tail_bias_raw_Toy_minus_Gaussian"] = deviations.toy_p_raw-deviations.gaussian_p_at_observed
    deviations["dkw_95_single_distribution_halfwidth"] = np.sqrt(np.log(40)/(2*deviations.sample_size))
    write_csv(deviations, destination/"gaussian_deviations.csv")


def score_distribution(values, mean, sigma, observed):
    """Fully specified normal KS test: no parameters estimated from these Toys."""
    values = np.asarray(values, dtype=float)
    if values.size < 2 or sigma <= 0 or not np.all(np.isfinite(values)):
        raise ValueError("Need finite samples and positive Gaussian width")
    distribution = norm(mean, sigma)
    ks = kstest(values, distribution.cdf)
    count = int(np.count_nonzero(values >= observed))
    n = values.size
    return dict(sample_size=n, gaussian_mean=mean, gaussian_sigma=sigma, gaussian_variance=sigma**2,
                toy_mean=float(values.mean()), toy_sigma=float(values.std(ddof=1)),
                ks_D=float(ks.statistic), ks_p=float(ks.pvalue),
                gaussian_p_at_observed=float(distribution.sf(observed)),
                tail_count=count, toy_p_raw=count/n, toy_p_smoothed=(count+1)/(n+1),
                tail_ci95_low=0.0 if count == 0 else float(beta.ppf(.025, count, n-count+1)),
                tail_ci95_high=1.0 if count == n else float(beta.ppf(.975, count+1, n-count)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--analysis-config", type=Path)
    parser.add_argument("--plot-only", type=Path, help="Replot a completed comparison directory using saved CSVs only")
    parser.add_argument("--batch")
    parser.add_argument("--layout", choices=("individual", "combined"), default="individual")
    args = parser.parse_args()
    if args.plot_only:
        render_comparison(pd.read_csv(args.plot_only/"samples_with_gaussian_p.csv", float_precision="round_trip"),
            pd.read_csv(args.plot_only/"distribution_comparison.csv", float_precision="round_trip"),
            args.plot_only, args.layout)
        print(args.plot_only)
        return
    if any(getattr(args, name) is None for name in ("samples", "metadata", "analysis_config")):
        parser.error("Provide --samples, --metadata and --analysis-config, or --plot-only")
    samples = pd.read_csv(args.samples, float_precision="round_trip")
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    if metadata["scan_mode"] != "appearance-profile":
        raise ValueError("This archived-sample adapter supports Fig3a appearance-profile only")
    analysis = build_three_plus_one_analysis(load_analysis_selection(args.analysis_config, repository_root=ROOT), repository_root=ROOT)
    if analysis.analysis_name != metadata["analysis_name"]:
        raise ValueError("Historical sample and current analysis differ")
    null = ThreePlusOneParameters(1., 0., 0.)
    null_chi2 = analysis.objective.chi2(null)
    begin_output_batch(args.batch)
    destination = result_directory("studies", "three_plus_one_toy_distribution_fit", "gaussian_comparison")
    if destination.exists():
        raise FileExistsError("Use a new batch; refusing to overwrite a diagnostic")
    rows, augmented, observed_rows = [], [], []
    for point_index, point in samples.groupby("point_index", sort=True):
        first = point.iloc[0]
        tested = ThreePlusOneParameters(float(first.fixed_delta_m2_41_eV2), float(first.profiled_observed_sin2_theta14), float(first.derived_observed_sin2_theta24))
        observed_chi4 = analysis.objective.chi2(tested)
        np.testing.assert_allclose(observed_chi4, first.observed_profiled_chi2_4nu, rtol=1e-10, atol=1e-8)
        observed = observed_chi4 - null_chi2
        pairs = _hypothesis_pairs(analysis, null, tested)
        approx = asymptotic_cls(observed, pairs)
        hypotheses = [tuple(pair[i] for pair in pairs) for i in (0, 1)]
        fixed_chi = [prepare_fixed_hypothesis_chi2(h) for h in hypotheses]
        seeds = np.random.SeedSequence(_stable_point_seed(metadata["toy_seed"], int(point_index))).spawn(2)
        point_scores = {}
        for i, generator in enumerate(("3nu", "4nu")):
            group = point[point.generator == generator].sort_values("toy_index").copy()
            n = len(group)
            np.testing.assert_array_equal(group.toy_index, np.arange(n))
            if n != metadata["toys_per_hypothesis_per_point"]:
                raise ValueError("Incomplete archived samples")
            draws = _draw_gaussian_toys(hypotheses[i], tuple(cholesky(h.covariance, lower=True) for h in hypotheses[i]), n,
                tuple(np.random.default_rng(s) for s in seeds[i].spawn(len(hypotheses[i]))))
            chi0, chi1 = [], []
            for index in range(n):
                data = tuple(component[index] for component in draws)
                chi0.append(fixed_chi[0](data)); chi1.append(fixed_chi[1](data))
            # Checks reproduction of the saved sample, not merely similar histograms.
            np.testing.assert_allclose(chi0, group.chi2_3nu, rtol=1e-10, atol=1e-8)
            np.testing.assert_allclose(group.chi2_4nu-group.chi2_3nu, group.test_statistic, rtol=1e-10, atol=1e-8)
            fixed = np.asarray(chi1)-np.asarray(chi0)
            mean = getattr(approx, "mean_under_"+generator)
            sigma = getattr(approx, "standard_deviation_under_"+generator)
            for variant, values in (("reprofiled", group.test_statistic.to_numpy()), ("fixed", fixed)):
                score = score_distribution(values, mean, sigma, observed)
                rows.append(dict(point_index=int(point_index), generator=generator, variant=variant,
                    delta_m2_41_eV2=tested.delta_m2_41_eV2, sin2_2theta_mue=tested.sin2_2theta_mue_exact,
                    observed_T=observed, **score))
                if variant == "reprofiled": point_scores[generator] = score
            group["fixed_T_same_draw"] = fixed
            group["gaussian_p_profiled_T"] = norm.sf(group.test_statistic, loc=mean, scale=sigma)
            group["gaussian_p_fixed_T"] = norm.sf(fixed, loc=mean, scale=sigma)
            augmented.append(group)
        observed_rows.append(dict(point_index=int(point_index), gaussian_cls=approx.cls,
            toy_cls=min(1., point_scores['4nu']['toy_p_smoothed']/point_scores['3nu']['toy_p_smoothed'])))
        print(f"Point {point_index}: replay matched saved null chi2; comparison complete", flush=True)
    write_csv(pd.DataFrame(rows), destination/"distribution_comparison.csv")
    write_csv(pd.concat(augmented, ignore_index=True), destination/"samples_with_gaussian_p.csv")
    write_csv(pd.DataFrame(observed_rows), destination/"observed_cls_comparison.csv")
    render_comparison(pd.concat(augmented, ignore_index=True), pd.DataFrame(rows), destination, args.layout)
    write_json(destination/"metadata.json", dict(
        input_paths={k: str(getattr(args,k).resolve()) for k in ('samples','metadata','analysis_config')},
        input_sha256={k: sha256(getattr(args,k).read_bytes()).hexdigest() for k in ('samples','metadata','analysis_config')},
        profiled_toys_reused=True, new_profile_fits=0, gaussian_parameters_fitted_to_toys=False,
        replay_validation="All saved null chi2 matched at rtol=1e-10, atol=1e-8; stored observed alternative chi2 matched",
        points=int(samples.point_index.nunique()), toys_per_hypothesis_per_point=metadata["toys_per_hypothesis_per_point"],
        warning="KS tests unadjusted for multiple comparisons. Analytic moments describe fixed hypotheses, not reprofiled T."))
    finish_output_batch(sys.argv[1:])
    print(destination)

if __name__ == "__main__": main()
