"""Six-point pilot: fixed-quadratic inversion versus canonical profiled Toys."""
import os
# Single-process, single BLAS thread; do not alter any core numerical settings.
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
import argparse
from hashlib import sha256
from pathlib import Path
import sys
from time import perf_counter
import numpy as np
import pandas as pd
from scipy.linalg import cholesky
from scipy.stats import beta
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
from studies.quadratic_toy_check.quadratic import from_hypotheses
from studies.three_plus_one_toy_distribution_fit.run import _build_analysis
from sterile_fit.adapter import _hypothesis_pairs
from sterile_fit.core.calibration import _draw_gaussian_toys, prepare_fixed_hypothesis_chi2
from sterile_fit.core.profile_three_plus_one import profile_s14_s24_at_fixed_sin2_2theta_mue, profile_s14_s24_at_fixed_sin2_2theta_ee
from sterile_fit.core.three_plus_one import ThreePlusOneParameters
from sterile_fit.scan import _stable_point_seed, _profile_toy_at_scan_point
from sterile_fit.output import begin_output_batch, result_directory, finish_output_batch, write_csv, write_json, plot_statistic_calibration


def profile_observation(objective, mass, amplitude, mode):
    """Same coordinate definitions and core functions as the actual scan grids."""
    if mode == "appearance-profile":
        return profile_s14_s24_at_fixed_sin2_2theta_mue(objective,
            delta_m2_41_eV2=mass, sin2_2theta_mue=amplitude).best_fit
    if mode == "electron-disappearance-profile":
        return profile_s14_s24_at_fixed_sin2_2theta_ee(objective,
            delta_m2_41_eV2=mass, sin2_2theta_ee=amplitude).best_fit
    raise ValueError("Unsupported profile mode")


def select_points(frame, mode="appearance-profile"):
    """Old Toy results only nominate coordinates; no historical predictions reused."""
    pool = frame[frame.cls_toy.notna()].copy()
    amplitude_column = "fixed_sin2_2theta_mue" if mode == "appearance-profile" else "fixed_sin2_2theta_ee"
    chosen = []
    for label, target in (("near", .045), ("near", .05), ("near", .06),
                          ("far_retained", .7), ("far_retained", .35), ("far_excluded", .01)):
        ranking = (np.log(pool.cls_toy.clip(lower=1e-12)/target)).abs().sort_values()
        for idx in ranking.index:
            row = pool.loc[idx]
            mass, amplitude = float(row.fixed_delta_m2_41_eV2), float(row[amplitude_column])
            if all(np.hypot(np.log10(mass/p["mass"]), np.log10(amplitude/p["amplitude"])) > .35 for p in chosen):
                chosen.append(dict(point_index=len(chosen), region=label, mass=mass, amplitude=amplitude,
                                   selection_cls_toy=float(row.cls_toy)))
                break
        else:
            raise ValueError("Insufficient separated historical Toy points")
    return chosen


def tail_score(sample, observed):
    n, count = len(sample), int(np.count_nonzero(sample >= observed))
    return dict(tail_count=count, toy_p=(count+1)/(n+1),
        ci95_low=0. if count == 0 else float(beta.ppf(.025, count, n-count+1)),
        ci95_high=1. if count == n else float(beta.ppf(.975, count+1, n-count)))


def refine_near_points(points, analysis, null, mode="appearance-profile"):
    """Locate candidate boundary with CF tails, never with normal tails or new Toys.

    This nominates validation coordinates only. Toy coverage is NOT assumed.
    """
    chi3 = analysis.objective.chi2(null)
    targets = [(p, p.get("target_cls", (.035, .05, .07)[j%3]))
               for j, p in enumerate(points) if p["region"] == "near"]
    for point, target in targets:
        cache = {}
        def evaluate(log_amplitude):
            key = float(log_amplitude)
            if key not in cache:
                fit = profile_observation(analysis.objective.chi2, point["mass"], 10**key, mode)
                pairs = _hypothesis_pairs(analysis, null, fit.parameters)
                tails = [from_hypotheses(pairs, h).evaluate([fit.chi2-chi3])[1][0] for h in (0, 1)]
                if tails[0] <= 1e-7:
                    raise ArithmeticError("Unresolved denominator during boundary nomination")
                cache[key] = float(tails[1]/tails[0]-target)
            return cache[key]
        center = np.log10(point["amplitude"])
        found = None
        for radius in (.2, .5, 1., 2., 4.):
            grid = np.linspace(max(-4., center-radius), min(0., center+radius), 7)
            vals = [evaluate(x) for x in grid]
            brackets = [(grid[j], grid[j+1]) for j in range(6) if vals[j]*vals[j+1] <= 0]
            if brackets:
                found = min(brackets, key=lambda pair: abs(np.mean(pair)-center))
                break
        if found is None:
            raise ArithmeticError("No candidate crossing; do not label a far point as near")
        root = brentq(evaluate, *found, xtol=1e-5)
        point["historical_amplitude"] = point["amplitude"]
        point["amplitude"] = 10**root
        point["selection_quadratic_cls"] = evaluate(root)+target
        print(f"Refined near point {point['point_index']}: amplitude={point['amplitude']:.7g} CF CLs={point['selection_quadratic_cls']:.6g}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan-csv", type=Path, required=True)
    parser.add_argument("--analysis-config", type=Path, default=ROOT/"configs/analyses/microboone_bnb_numi.yaml")
    parser.add_argument("--toys", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20250821)
    parser.add_argument("--batch", required=True)
    parser.add_argument("--refine-near", action="store_true", help="Refine 3 candidate boundary points using CF tails before validation")
    parser.add_argument("--boundary-triplets", action="store_true", help="Three amplitudes per mass at candidate CLs .015/.05/.15; plus three far points")
    parser.add_argument("--mode", choices=("appearance-profile","electron-disappearance-profile"), default="appearance-profile")
    args = parser.parse_args()
    if args.toys < 20:
        parser.error("At least 20 Toys per hypothesis")
    begin_output_batch(args.batch)
    out = result_directory("studies", "quadratic_toy_check", "results")
    if out.exists():
        raise FileExistsError("Use a new batch; existing samples will not be overwritten")
    points = select_points(pd.read_csv(args.scan_csv), args.mode)
    if args.boundary_triplets:
        points = [dict(p, target_cls=target) for p in points[:3] for target in (.15, .05, .015)]+points[3:]
        for index, point in enumerate(points):
            point["point_index"] = index
    analysis = _build_analysis(args.analysis_config)
    null = ThreePlusOneParameters(1., 0., 0.)
    if args.refine_near or args.boundary_triplets:
        refine_near_points(points, analysis, null, args.mode)
    write_csv(pd.DataFrame(points), out/"selected_points.csv")
    hashes = {str(p.relative_to(ROOT)):sha256(p.read_bytes()).hexdigest()
              for p in (ROOT/"src/sterile_fit/core").glob("*.py")}
    write_json(out/"metadata.json", dict(status="running", toys_per_hypothesis=args.toys,
        seed=args.seed, workers=1, core_sha256=hashes, scan_mode=args.mode,
        scan_csv=str(args.scan_csv.resolve()), scan_sha256=sha256(args.scan_csv.read_bytes()).hexdigest(),
        config_sha256=sha256(args.analysis_config.read_bytes()).hexdigest(),
        method="Fixed Gaussian quadratic CF inversion; no Toy fitting; Gaussian display only",
        selection="CF-refined boundary triplets and 3 far points" if args.boundary_triplets else ("3 CF-refined near points and 3 historical far points" if args.refine_near else "3 near and 3 far relative to old finite-Toy CLs; not a certified true boundary"),
        caveat="Profiles unchanged. A fixed quadratic law is a candidate approximation for profiled T."))
    started, done = perf_counter(), 0
    all_samples, scores, cls_rows = [], [], []
    for point in points:
        index = point["point_index"]
        fitted = profile_observation(analysis.objective.chi2, point["mass"], point["amplitude"], args.mode)
        tested = fitted.parameters
        observed = fitted.chi2-analysis.objective.chi2(null)
        pairs = _hypothesis_pairs(analysis, null, tested)
        hs = [tuple(p[i] for p in pairs) for i in (0, 1)]
        evaluators = [prepare_fixed_hypothesis_chi2(h) for h in hs]
        seeds = np.random.SeedSequence(_stable_point_seed(args.seed, index)).spawn(2)
        panels, tails = [], []
        for h in (0, 1):
            law = from_hypotheses(pairs, h)
            write_csv(pd.DataFrame(dict(eigenvalue=law.eigenvalues, linear=law.linear)), out/f"point_{index:02d}_{h+3}nu_coefficients.csv")
            generators = tuple(np.random.default_rng(s) for s in seeds[h].spawn(len(hs[h])))
            factors = tuple(cholesky(x.covariance, lower=True) for x in hs[h])
            draws = _draw_gaussian_toys(hs[h], factors, args.toys, generators)
            rows = []
            for j in range(args.toys):
                dataset = tuple(x[j] for x in draws)
                chi3 = evaluators[0](dataset)
                fixed4 = evaluators[1](dataset)
                toy_fit = _profile_toy_at_scan_point(analysis, dataset, mode=args.mode, tested_parameters=tested)
                chi4 = toy_fit.chi2
                amplitude_check = (toy_fit.parameters.sin2_2theta_mue_exact if args.mode == "appearance-profile"
                                   else toy_fit.parameters.sin2_2theta_ee_exact)
                np.testing.assert_allclose(amplitude_check, point["amplitude"], rtol=1e-8, atol=1e-12)
                rows.append(dict(point_index=index, generator=f"{h+3}nu", toy_index=j,
                    chi2_3nu=chi3, chi2_4nu=chi4, test_statistic=chi4-chi3, fixed_T=fixed4-chi3,
                    toy_sin2_theta14=toy_fit.parameters.sin2_theta14, toy_sin2_theta24=toy_fit.parameters.sin2_theta24))
                done += 1
                if (j+1) % (200 if args.toys >= 1000 else 50) == 0:
                    elapsed = perf_counter()-started
                    total = 2*len(points)*args.toys
                    print(f"point={index} H={h+3}nu {j+1}/{args.toys}; total={done}/{total}; elapsed={elapsed:.1f}s ETA={(total-done)*elapsed/done:.1f}s", flush=True)
            sample = pd.DataFrame(rows)
            write_csv(sample, out/f"point_{index:02d}_{h+3}nu_samples.csv")
            all_samples.append(sample)
            low = min(sample.test_statistic.min(), sample.fixed_T.min(), law.mean-4*law.sigma, observed)
            high = max(sample.test_statistic.max(), sample.fixed_T.max(), law.mean+4*law.sigma, observed)
            plot_grid = np.linspace(low, high, 500)
            # Direct inversion at every Toy order statistic gives KS without CDF interpolation.
            values = np.unique(np.r_[plot_grid, sample.test_statistic, sample.fixed_T, observed])
            pdf, sf, diagnostics = law.evaluate(values)
            observed_p = float(sf[np.searchsorted(values, observed)])
            curve = pd.DataFrame(dict(T=values, pdf=pdf, sf=sf))
            write_csv(curve, out/f"point_{index:02d}_{h+3}nu_curve.csv")
            for variant, column in (("reprofiled", "test_statistic"), ("fixed", "fixed_T")):
                ordered = np.sort(sample[column].to_numpy())
                cdf = 1-sf[np.searchsorted(values, ordered)]
                ks = max(np.max(np.arange(1,len(ordered)+1)/len(ordered)-cdf), np.max(cdf-np.arange(len(ordered))/len(ordered)))
                score = dict(**point, generator=f"{h+3}nu", variant=variant, observed_T=observed,
                    quadratic_p=observed_p, mean=law.mean, sigma=law.sigma, constant=law.constant,
                    generation_sin2_theta14=tested.sin2_theta14, generation_sin2_theta24=tested.sin2_theta24,
                    ks_D=float(ks), dkw95_halfwidth=float(np.sqrt(np.log(40)/(2*args.toys))),
                    **diagnostics, **tail_score(sample[column].to_numpy(), observed))
                scores.append(score)
                if variant == "reprofiled": tails.append(score)
            panels.append(dict(label=f"{h+3}nu generation; N={args.toys}", profiled_T=sample.test_statistic,
                fixed_T=sample.fixed_T, mean=law.mean, sigma=law.sigma, observed_T=observed,
                candidate=curve))
        ratio = tails[1]["quadratic_p"]/tails[0]["quadratic_p"] if tails[0]["quadratic_p"] > 0 else float("nan")
        cls_rows.append(dict(**point, quadratic_cls=min(1., ratio) if np.isfinite(ratio) else ratio,
            toy_cls=min(1., tails[1]["toy_p"]/tails[0]["toy_p"])))
        symbol = r"\mu e" if args.mode == "appearance-profile" else "ee"
        plot_statistic_calibration(panels, out/f"point_{index:02d}_comparison.png",
            title=rf"BNB+NuMI point {index} ({point['region']}): $\Delta m^2_{{41}}={point['mass']:.4g}$ eV$^2$, $\sin^2(2\theta_{{{symbol}}})={point['amplitude']:.4g}$")
        write_csv(pd.DataFrame(scores), out/"scores.csv")
        write_csv(pd.DataFrame(cls_rows), out/"cls_comparison.csv")
    write_csv(pd.concat(all_samples, ignore_index=True), out/"samples.csv")
    import json
    metadata = json.loads((out/"metadata.json").read_text(encoding="utf-8"))
    metadata.update(status="complete", elapsed_seconds=perf_counter()-started,
        code_sha256={p.name:sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob("*.py")})
    write_json(out/"metadata.json", metadata)
    finish_output_batch(sys.argv[1:])
    print(out, flush=True)


if __name__ == "__main__":
    main()
