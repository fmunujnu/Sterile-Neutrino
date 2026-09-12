"""Saved-Toy assessment: CDF errors, observed tails, T cuts and amplitude shifts.

No new Toys. Optional analytic boundary polishing reuses canonical profile.
Bootstrap uncertainty is descriptive, not an
exact coverage claim. Exact binomial intervals cover fixed observed thresholds.
"""
import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
import argparse
import json
from pathlib import Path
import sys
from hashlib import sha256
import numpy as np
import pandas as pd
from scipy.stats import norm, beta
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/"src"), str(ROOT)]
from sterile_fit.output import begin_output_batch, result_directory, write_csv, write_json, finish_output_batch, plot_statistic_calibration, plot_calibration_boundaries


def cp_interval(k, n, alpha=.05):
    return (0. if k == 0 else float(beta.ppf(alpha/2, k, n-k+1)),
            1. if k == n else float(beta.ppf(1-alpha/2, k+1, n-k)))


def crossings(x, q):
    """All downward zero crossings; never extrapolate missing boundaries."""
    x, q = np.asarray(x), np.asarray(q)
    found = []
    for j in range(len(x)-1):
        if np.isfinite(q[j:j+2]).all() and q[j] > 0 and q[j+1] <= 0:
            f = q[j]/(q[j]-q[j+1])
            found.append((float(x[j]+f*(x[j+1]-x[j])), float(x[j]), float(x[j+1])))
    return found


def ks_distance(cdf, weights=None):
    n = len(cdf)
    w = np.ones(n)/n if weights is None else weights/n
    right = np.cumsum(w)
    return float(max(np.max(right-cdf), np.max(cdf-(right-w))))


def polish_gaussian_boundaries(frame, boundary_rows, source):
    """Remove three-node interpolation error from the Gaussian comparator only."""
    if not boundary_rows:
        return
    from studies.quadratic_toy_check.run import _build_analysis
    from sterile_fit.core.three_plus_one import ThreePlusOneParameters
    from studies.quadratic_toy_check.run import profile_observation
    from sterile_fit.experiments.microboone.adapter import _hypothesis_pairs
    from sterile_fit.core.calibration import asymptotic_cls
    config = ROOT/"configs/analyses/microboone_bnb_numi.yaml"
    source_metadata = json.loads((source/"metadata.json").read_text(encoding="utf-8"))
    mode = source_metadata.get("scan_mode","appearance-profile")
    if source_metadata.get("config_sha256") != sha256(config.read_bytes()).hexdigest():
        raise ValueError("Analytic polishing requires the original matching BNB+NuMI configuration")
    analysis = _build_analysis(config)
    null = ThreePlusOneParameters(1.,0.,0.)
    chi3 = analysis.objective.chi2(null)
    for b in boundary_rows:
        lo,hi = b["boundary_gaussian_bracket_low"],b["boundary_gaussian_bracket_high"]
        b["boundary_gaussian_interpolated"] = b["boundary_gaussian"]
        def f(log_amplitude):
            fit = profile_observation(analysis.objective.chi2,b["mass"],10**log_amplitude,mode)
            return asymptotic_cls(fit.chi2-chi3,_hypothesis_pairs(analysis,null,fit.parameters)).cls-.05
        b["gaussian_bracket_extended"] = False
        if not np.isfinite(lo+hi):
            # Evaluate real theory outside the Toy slice; never extrapolate a missing root.
            grid = np.linspace(np.log10(b["amplitude_min"]), 0., 16)
            roots = crossings(grid,[f(x) for x in grid])
            if len(roots) != 1:
                continue
            lo,hi = 10**roots[0][1],10**roots[0][2]
            b["gaussian_bracket_extended"] = True
            b["boundary_gaussian_bracket_low"],b["boundary_gaussian_bracket_high"] = lo,hi
        b["boundary_gaussian"] = 10**brentq(f,np.log10(lo),np.log10(hi),xtol=1e-5)
        b["gaussian_shift_dex"] = np.log10(b["boundary_gaussian"]/b["boundary_toy"])
        b["gaussian_shift_percent"] = 100*(b["boundary_gaussian"]/b["boundary_toy"]-1)


def distribution_metrics(values, curve, mean, sigma, observed, rng, repeats, numerical_resolution=1e-12):
    ordered = np.sort(values)
    fq = 1-np.interp(ordered, curve.T_values, curve.sf)
    fg = norm.cdf(ordered, mean, sigma)
    dq, dg = ks_distance(fq), ks_distance(fg)
    improvements = []
    for _ in range(repeats):
        weights = rng.multinomial(len(values), np.full(len(values), 1/len(values)))
        improvements.append(ks_distance(fg, weights)-ks_distance(fq, weights))
    lo, hi = np.quantile(improvements, [.025,.975])
    count = int(np.count_nonzero(values >= observed))
    raw_quadratic = float(np.interp(observed, curve.T_values, curve.sf))
    return dict(N=len(values), tail_count=count, p_toy_raw=count/len(values), p_toy=(count+1)/(len(values)+1),
        p_quadratic=raw_quadratic if raw_quadratic > numerical_resolution else np.nan,
        p_quadratic_raw_integral=raw_quadratic, p_quadratic_numerical_resolution=numerical_resolution,
        p_quadratic_numerical_upper=max(0., raw_quadratic)+numerical_resolution,
        p_gaussian=float(norm.sf(observed,mean,sigma)),
        ks_quadratic=dq, ks_gaussian=dg, ks_improvement=dg-dq,
        ks_improvement_bootstrap95_low=lo, ks_improvement_bootstrap95_high=hi,
        profile_gaussian_bump=dg, profile_quadratic_bump=dq)


def profile_coordinate_audit(values, amplitude, mode, generation_s14):
    """Check stored Toy coordinates without refitting or changing any sample."""
    s14, s24 = values.toy_sin2_theta14.to_numpy(), values.toy_sin2_theta24.to_numpy()
    aee = 4*s14*(1-s14)
    actual = aee*s24 if mode == "appearance-profile" else aee
    np.testing.assert_allclose(actual, amplitude, rtol=1e-8, atol=1e-12)
    if np.any((s14 < 0)|(s14 > 1)|(s24 < 0)|(s24 > 1)):
        raise ValueError("Toy profile outside physical mixing bounds")
    return dict(max_amplitude_error=float(np.max(np.abs(actual-amplitude))),
        low_s14_branch_count=int(np.sum(s14 < .5)),
        high_s14_branch_count=int(np.sum(s14 > .5)),
        opposite_generation_s14_half_count=int(np.sum((s14 > .5)!=(generation_s14 > .5))),
        toy_s24_min=float(s24.min()), toy_s24_max=float(s24.max()),
        toy_s24_mean=float(s24.mean()), toy_s24_std=float(s24.std()))


def assess(source, destination, repeats=300):
    source_metadata=json.loads((source/"metadata.json").read_text(encoding="utf-8"))
    mode=source_metadata.get("scan_mode","appearance-profile")
    symbol=r"\mu e" if mode=="appearance-profile" else "ee"
    scores = pd.read_csv(source/"scores.csv")
    samples = pd.read_csv(source/"samples.csv")
    rng = np.random.default_rng(20260904)  # Analysis bootstrap only; not physical Toy RNG.
    metrics, cls_rows, cut_rows = [], [], []
    point_count = samples.point_index.nunique()
    for index, sample in samples.groupby("point_index", sort=True):
        panels, tails, curves, quantities = [], [], [], []
        for generator in ("3nu", "4nu"):
            score = scores[(scores.point_index == index)&(scores.generator == generator)&(scores.variant == "reprofiled")].iloc[0]
            values = sample[sample.generator == generator]
            curve = pd.read_csv(source/f"point_{index:02d}_{generator}_curve.csv").rename(columns={"T":"T_values"})
            resolution = max(1e-12, float(score.cdf_truncation_bound)+float(score.mesh_difference))
            row = distribution_metrics(values.test_statistic.to_numpy(), curve, score["mean"], score.sigma, score.observed_T, rng, repeats, resolution)
            ordered_fixed = np.sort(values.fixed_T.to_numpy())
            row["ks_fixed_quadratic"] = ks_distance(1-np.interp(ordered_fixed,curve.T_values,curve.sf))
            difference = values.test_statistic.to_numpy()-values.fixed_T.to_numpy()
            row["mean_profile_minus_fixed"] = float(difference.mean())
            row["rms_profile_minus_fixed"] = float(np.sqrt(np.mean(difference**2)))
            row["tail_fixed_count"] = int(np.count_nonzero(values.fixed_T.to_numpy() >= score.observed_T))
            if "toy_sin2_theta14" in values.columns:
                row.update(profile_coordinate_audit(values, score.amplitude, mode, score.generation_sin2_theta14))
            lo, hi = cp_interval(row["tail_count"], row["N"], alpha=.05/(2*point_count))
            row.update(point_index=index, generator=generator, tail_simultaneous95_low=lo, tail_simultaneous95_high=hi)
            metrics.append(row)
            tails.append(row)
            curves.append(curve)
            quantities.append(score)
            panels.append(dict(label=f"{generator}: N={len(values):,}", profiled_T=values.test_statistic,
                fixed_T=values.fixed_T, mean=score["mean"], sigma=score.sigma, observed_T=score.observed_T,
                candidate=curve.rename(columns={"T_values":"T"})))
        first = quantities[0]
        row = dict(point_index=index, mass=first.mass, amplitude=first.amplitude, region=first.region,
            cls_simultaneous95_low=tails[1]["tail_simultaneous95_low"]/tails[0]["tail_simultaneous95_high"],
            cls_simultaneous95_high=min(1., tails[1]["tail_simultaneous95_high"]/tails[0]["tail_simultaneous95_low"]) if tails[0]["tail_simultaneous95_low"] > 0 else 1.)
        for method in ("toy", "gaussian", "quadratic"):
            p3, p4 = tails[0][f"p_{method}"], tails[1][f"p_{method}"]
            row[f"cls_{method}"] = min(1., p4/p3) if p3 > 0 and np.isfinite(p4) else np.nan
            row[f"q_{method}"] = p4-.05*p3
        for h in (0,1):
            row[f"count_{h+3}"] = tails[h]["tail_count"]
            row[f"N_{h+3}"] = tails[h]["N"]
        denominator_lower = tails[0]["p_quadratic_raw_integral"]-tails[0]["p_quadratic_numerical_resolution"]
        row["cls_quadratic_numerical_upper"] = min(1., tails[1]["p_quadratic_numerical_upper"]/denominator_lower) if denominator_lower > 0 else np.nan
        cls_rows.append(row)
        # Cuts compare distributions at the SAME physical point. They are not an amplitude boundary.
        low = min(c.T_values.min() for c in curves)
        high = max(c.T_values.max() for c in curves)
        grid = np.linspace(low, high, 2500)
        cut = dict(point_index=index, observed_T=first.observed_T, mass=first.mass, amplitude=first.amplitude)
        for method in ("toy", "gaussian", "quadratic"):
            ps = []
            for h, generator in enumerate(("3nu", "4nu")):
                if method == "toy":
                    ordered = np.sort(sample[sample.generator == generator].test_statistic)
                    ps.append((len(ordered)-np.searchsorted(ordered, grid, side="left")+1)/(len(ordered)+1))
                elif method == "gaussian":
                    ps.append(norm.sf(grid, quantities[h]["mean"], quantities[h].sigma))
                else:
                    ps.append(np.interp(grid, curves[h].T_values, curves[h].sf, left=1., right=0.))
            # Require >=20 expected/empirical denominator events; mark absent roots, do not extrapolate.
            q = np.where(ps[0]*tails[0]["N"] >= 20, ps[1]-.05*ps[0], np.nan)
            roots = crossings(grid, q)
            cut[f"{method}_crossing_count"] = len(roots)
            chosen = min(roots, key=lambda r: abs(r[0]-first.observed_T)) if roots else (np.nan,)*3
            cut[f"Tcut_{method}"] = chosen[0]
            cut[f"Tcut_{method}_grid_low"], cut[f"Tcut_{method}_grid_high"] = chosen[1:]
        cut["gaussian_minus_toy_in_sigma3"] = (cut["Tcut_gaussian"]-cut["Tcut_toy"])/first.sigma
        cut["quadratic_minus_toy_in_sigma3"] = (cut["Tcut_quadratic"]-cut["Tcut_toy"])/first.sigma
        boot_cuts = []
        bin_probabilities = []
        for generator in ("3nu", "4nu"):
            vals = sample[sample.generator == generator].test_statistic.to_numpy()
            bins = np.searchsorted(grid, vals, side="right")
            bin_probabilities.append(np.bincount(bins, minlength=len(grid)+1)/len(vals))
        for _ in range(repeats):
            bp = []
            for h in (0,1):
                counts = rng.multinomial(tails[h]["N"], bin_probabilities[h])
                bp.append((tails[h]["N"]-np.cumsum(counts)[:-1]+1)/(tails[h]["N"]+1))
            q = np.where(bp[0]*tails[0]["N"] >= 20, bp[1]-.05*bp[0], np.nan)
            roots = crossings(grid,q)
            if roots:
                boot_cuts.append(min(roots,key=lambda r:abs(r[0]-first.observed_T))[0])
        cut["toy_cut_bootstrap_in_range_fraction"] = len(boot_cuts)/repeats
        cut["toy_cut_bootstrap95_low"], cut["toy_cut_bootstrap95_high"] = np.quantile(boot_cuts,[.025,.975]) if boot_cuts else (np.nan,np.nan)
        cut_rows.append(cut)
        plot_statistic_calibration(panels, destination/f"point_{index:02d}_comparison.png",
            title=rf"Point {index}: $\Delta m^2_{{41}}={first.mass:.5g}$ eV$^2$, $\sin^2(2\theta_{{{symbol}}})={first.amplitude:.5g}$")
        print(f"assessed point {index}; KS quadratic/Gaussian: {tails[0]['ks_quadratic']:.4g}/{tails[0]['ks_gaussian']:.4g}, {tails[1]['ks_quadratic']:.4g}/{tails[1]['ks_gaussian']:.4g}", flush=True)
    frame = pd.DataFrame(cls_rows)
    boundary_rows = []
    for mass, group in frame[frame.region == "near"].groupby("mass"):
        if len(group) < 3:
            continue
        group = group.sort_values("amplitude")
        x = np.log10(group.amplitude.to_numpy())
        b = dict(mass=mass, points=len(group), amplitude_min=10**x[0], amplitude_max=10**x[-1])
        for method in ("toy", "gaussian", "quadratic"):
            roots = crossings(x, group[f"q_{method}"].to_numpy())
            root = roots[0] if len(roots) == 1 else (np.nan,)*3
            b[f"boundary_{method}"] = 10**root[0]
            b[f"boundary_{method}_bracket_low"], b[f"boundary_{method}_bracket_high"] = 10**root[1], 10**root[2]
            b[f"{method}_crossings"] = len(roots)
        boot_roots = []
        for _ in range(repeats):
            k3 = rng.binomial(group.N_3.to_numpy(int), group.count_3/group.N_3)
            k4 = rng.binomial(group.N_4.to_numpy(int), group.count_4/group.N_4)
            q = (k4+1)/(group.N_4.to_numpy()+1)-.05*(k3+1)/(group.N_3.to_numpy()+1)
            roots = crossings(x, q)
            if len(roots) == 1:
                boot_roots.append(10**roots[0][0])
        b["bootstrap_in_range_fraction"] = len(boot_roots)/repeats
        b["toy_bootstrap95_low"], b["toy_bootstrap95_high"] = np.quantile(boot_roots,[.025,.975]) if boot_roots else (np.nan,np.nan)
        for method in ("gaussian", "quadratic"):
            b[f"{method}_shift_dex"] = np.log10(b[f"boundary_{method}"]/b["boundary_toy"])
            b[f"{method}_shift_percent"] = 100*(b[f"boundary_{method}"]/b["boundary_toy"]-1)
        boundary_rows.append(b)
    polish_gaussian_boundaries(frame, boundary_rows, source)
    for b in boundary_rows:
        for method in ("gaussian", "quadratic"):
            b[f"{method}_shift_percent_bootstrap95_low"] = 100*(b[f"boundary_{method}"]/b["toy_bootstrap95_high"]-1)
            b[f"{method}_shift_percent_bootstrap95_high"] = 100*(b[f"boundary_{method}"]/b["toy_bootstrap95_low"]-1)
    write_csv(pd.DataFrame(metrics), destination/"distribution_metrics.csv")
    write_csv(frame, destination/"tail_cls_comparison.csv")
    write_csv(pd.DataFrame(cut_rows), destination/"T_cut_comparison.csv")
    write_csv(pd.DataFrame(boundary_rows), destination/"amplitude_boundary_shifts.csv")
    if boundary_rows:
        plot_calibration_boundaries(frame, pd.DataFrame(boundary_rows), destination/"boundary_slices.png",
            amplitude_label=rf"$\sin^2(2\theta_{{{symbol}}})$")
    write_json(destination/"metadata.json", dict(source=str(source.resolve()), scan_mode=mode,bootstrap_repeats=repeats,
        bootstrap_seed=20260904, bootstrap="Nonparametric percentile, descriptive, not an exact error guarantee; boundary intervals conditional on in-range single crossings",
        tail_intervals="Clopper-Pearson with Bonferroni over 2*points, >=95% simultaneous coverage at fixed observed thresholds",
        cut="All downward crossings counted; closest observed-T crossing reported; denominator >=20/N; bootstrap cut intervals conditional on root availability, not formal coverage",
        boundary="Toy and CF: linear interpolation of q=p4-.05*p3 in log10 amplitude; no extrapolation; brackets retained. Gaussian comparator polished with canonical profile/brentq; a missing sampled bracket triggers actual theory evaluations up to amplitude 1, recorded by gaussian_bracket_extended; not full contour.",
        source_hashes={p.name:sha256(p.read_bytes()).hexdigest() for p in (source/"scores.csv",source/"samples.csv")},
        note="Normal used solely as comparator, not in quadratic or empirical calculations. Numerical upper estimates combine a truncation bound with mesh diagnostics, NOT a rigorous total quadrature bound; unresolved tails have blank p/CLs estimates."))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--batch", required=True)
    parser.add_argument("--bootstrap", type=int, default=300)
    args = parser.parse_args()
    begin_output_batch(args.batch)
    out = result_directory("studies","quadratic_toy_check","assessment")
    if out.exists(): raise FileExistsError("Use a new assessment batch")
    assess(args.source, out, args.bootstrap)
    finish_output_batch(sys.argv[1:])


if __name__ == "__main__": main()
