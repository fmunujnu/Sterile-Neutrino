from __future__ import annotations

import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"

from pathlib import Path
import sys
from time import perf_counter
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import cholesky

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from studies.three_plus_one_toy_distribution_fit.run import _build_analysis
from sterile_fit.experiments.microboone.adapter import _hypothesis_pairs
from sterile_fit.core.calibration import _draw_gaussian_toys, prepare_fixed_hypothesis_chi2
from sterile_fit.core.profile_three_plus_one import (
    profile_s14_s24_at_fixed_sin2_2theta_ee,
    profile_s14_s24_at_fixed_sin2_2theta_mue,
)
from sterile_fit.core.three_plus_one import ThreePlusOneParameters
from sterile_fit.output import write_csv, write_json
from sterile_fit.scan import _profile_toy_at_scan_point, _stable_point_seed


SPECS = {
    "fig3a": {
        "path": ROOT / "outputs/microboone_bnb_numi_joint/three_plus_one/fullgrid_fixed_toy5000_20260904/scan_fig3a_toy/result.csv",
        "mode": "appearance-profile", "x": "fixed_sin2_2theta_mue",
        "xlabel": r"$\sin^2(2\theta_{\mu e})$",
    },
    "fig3b": {
        "path": ROOT / "outputs/microboone_bnb_numi_joint/three_plus_one/fullgrid_fixed_toy5000_20260904/scan_electron-disfig3a_toy/result.csv",
        "mode": "electron-disappearance-profile", "x": "fixed_sin2_2theta_ee",
        "xlabel": r"$\sin^2(2\theta_{ee})$",
    },
}


def _baseline(frame, x_column):
    rows = []
    for mass, group in frame.groupby("fixed_delta_m2_41_eV2"):
        group = group.sort_values(x_column)
        q = group.cls_toy.to_numpy()-0.05
        x = group[x_column].to_numpy(dtype=float)
        crossings = np.flatnonzero(q[:-1]*q[1:] <= 0)
        if crossings.size:
            i = crossings[np.argmin(abs(q[crossings])+abs(q[crossings+1]))]
            fraction = -q[i]/(q[i+1]-q[i]) if q[i+1] != q[i] else 0.5
            amplitude = 10**(np.log10(x[i])+fraction*(np.log10(x[i+1])-np.log10(x[i])))
        elif q.min() <= 0 <= q.max():
            amplitude = x[np.argmin(abs(q))]
        else:
            continue
        rows.append((float(mass), float(amplitude)))
    return pd.DataFrame(rows, columns=["mass", "baseline_amplitude"])


def _profile_observed(analysis, mode, mass, amplitude):
    if mode == "appearance-profile":
        return profile_s14_s24_at_fixed_sin2_2theta_mue(
            analysis.objective.chi2, delta_m2_41_eV2=mass,
            sin2_2theta_mue=amplitude).best_fit
    return profile_s14_s24_at_fixed_sin2_2theta_ee(
        analysis.objective.chi2, delta_m2_41_eV2=mass,
        sin2_2theta_ee=amplitude).best_fit


def _evaluate_point(analysis, null, mode, mass, amplitude, toys, seed):
    fitted = _profile_observed(analysis, mode, mass, amplitude)
    observed_t = fitted.chi2-analysis.objective.chi2(null)
    pairs = _hypothesis_pairs(analysis, null, fitted.parameters)
    hypotheses = [tuple(pair[i] for pair in pairs) for i in (0, 1)]
    null_chi2 = prepare_fixed_hypothesis_chi2(hypotheses[0])
    seeds = np.random.SeedSequence(seed).spawn(2)
    counts = []
    for h in (0, 1):
        generators = tuple(np.random.default_rng(s) for s in seeds[h].spawn(len(hypotheses[h])))
        factors = tuple(cholesky(item.covariance, lower=True, check_finite=False)
                        for item in hypotheses[h])
        draws = _draw_gaussian_toys(hypotheses[h], factors, toys, generators)
        count = 0
        for i in range(toys):
            dataset = tuple(draw[i] for draw in draws)
            chi3 = null_chi2(dataset)
            toy_fit = _profile_toy_at_scan_point(
                analysis, dataset, mode=mode, tested_parameters=fitted.parameters)
            count += toy_fit.chi2-chi3 >= observed_t
        counts.append(count)
    p3, p4 = (counts[0]+1)/(toys+1), (counts[1]+1)/(toys+1)
    return counts[0], counts[1], p3, p4, p4-0.05*p3, min(1.0, p4/p3)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--toys", type=int, default=50)
    parser.add_argument("--blocks", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260906)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    args.output_directory.mkdir(parents=True, exist_ok=False)
    analysis = _build_analysis(ROOT / "configs/analyses/microboone_bnb_numi.yaml")
    null = ThreePlusOneParameters(1.0, 0.0, 0.0)
    all_samples, all_shifts = [], []
    started = perf_counter()
    total = len(SPECS)*args.blocks*3*5
    done = 0
    for figure_name, spec in SPECS.items():
        frame = pd.read_csv(spec["path"])
        baseline = _baseline(frame, spec["x"])
        log_edges = np.linspace(np.log10(baseline.mass.min()),
                                np.log10(baseline.mass.max()), args.blocks+1)
        shifted_segments = []
        block_boxes = []
        for block in range(args.blocks):
            low, high = log_edges[block:block+2]
            segment = baseline[(np.log10(baseline.mass) >= low) &
                               (np.log10(baseline.mass) <= high)].copy()
            if segment.empty:
                continue
            target_logs = np.linspace(low, high, 5)[1:4]
            sample_rows = []
            for mass_slot, target_log in enumerate(target_logs):
                base_row = segment.iloc[np.argmin(abs(np.log10(segment.mass)-target_log))]
                mass, base = float(base_row.mass), float(base_row.baseline_amplitude)
                for amplitude_slot, offset in enumerate(np.linspace(-0.18, 0.18, 5)):
                    amplitude = float(np.clip(base*10**offset,
                                              frame[spec["x"]].min(), frame[spec["x"]].max()))
                    seed = _stable_point_seed(args.seed + (0 if figure_name == "fig3a" else 100000),
                                              block*100+mass_slot*10+amplitude_slot)
                    c3,c4,p3,p4,q,cls = _evaluate_point(
                        analysis, null, spec["mode"], mass, amplitude, args.toys, seed)
                    row = {"figure": figure_name, "block": block, "mass": mass,
                           "baseline_amplitude": base, "amplitude": amplitude,
                           "toys_per_hypothesis": args.toys, "tail_3nu": c3,
                           "tail_4nu": c4, "p3": p3, "p4": p4, "q": q, "cls": cls}
                    sample_rows.append(row); all_samples.append(row); done += 1
                    elapsed = perf_counter()-started
                    print(f"points={done}/{total} elapsed={elapsed:.1f}s ETA={(total-done)*elapsed/done:.1f}s", flush=True)
            samples = pd.DataFrame(sample_rows)
            design = np.c_[np.ones(len(samples)), np.log10(samples.mass), np.log10(samples.amplitude)]
            coefficients, *_ = np.linalg.lstsq(design, samples.q, rcond=None)
            if abs(coefficients[2]) < 1e-12:
                shift = 0.0
            else:
                predicted_log_boundary = -(coefficients[0]+coefficients[1]*np.log10(segment.mass.to_numpy()))/coefficients[2]
                shift = float(np.median(predicted_log_boundary-np.log10(segment.baseline_amplitude)))
            shifted = segment.copy()
            shifted["shifted_amplitude"] = np.clip(segment.baseline_amplitude*10**shift,
                                                    frame[spec["x"]].min(), frame[spec["x"]].max())
            shifted["block"] = block
            shifted_segments.append(shifted)
            block_boxes.append((10**low, 10**high, samples.amplitude.min(), samples.amplitude.max()))
            all_shifts.append({"figure": figure_name, "block": block,
                               "mass_low": 10**low, "mass_high": 10**high,
                               "log10_amplitude_shift": shift,
                               "amplitude_factor": 10**shift,
                               "fit_q_rms": float(np.sqrt(np.mean((design@coefficients-samples.q)**2)))})

        figure, axis = plt.subplots(figsize=(7.5, 5.8), constrained_layout=True)
        axis.plot(baseline.baseline_amplitude, baseline.mass, color="tab:red", linewidth=2,
                  label="Existing fixed-Toy contour")
        for i, shifted in enumerate(shifted_segments):
            axis.plot(shifted.shifted_amplitude, shifted.mass, color="tab:blue", linewidth=2,
                      label="Coarse block-shift profiled-Toy estimate" if i == 0 else None)
        sample_table = pd.DataFrame(all_samples)
        shown = sample_table[sample_table.figure == figure_name]
        axis.scatter(shown.amplitude, shown.mass, s=8, color="black", alpha=.55,
                     label=f"Profiled-Toy samples ({args.toys}/hypothesis)")
        for mass_low,mass_high,amp_low,amp_high in block_boxes:
            axis.plot([amp_low,amp_high,amp_high,amp_low,amp_low],
                      [mass_low,mass_low,mass_high,mass_high,mass_low],
                      color="0.5", linewidth=.7, alpha=.7)
        axis.set_xscale("log"); axis.set_yscale("log")
        axis.set_xlabel(spec["xlabel"]); axis.set_ylabel(r"$\Delta m^2_{41}\;[\mathrm{eV}^2]$")
        axis.set_title(f"MicroBooNE {figure_name}: coarse profiled-Toy block shifts")
        axis.legend(fontsize=8); axis.grid(False)
        figure.savefig(args.output_directory/f"{figure_name}_block_shift.png", dpi=180, bbox_inches="tight")
        plt.close(figure)

    write_csv(pd.DataFrame(all_samples), args.output_directory/"sample_points.csv", index=False)
    write_csv(pd.DataFrame(all_shifts), args.output_directory/"block_shifts.csv", index=False)
    write_json(args.output_directory/"metadata.json", {
        "status":"complete", "blocks_per_figure":args.blocks,
        "masses_per_block":3, "amplitudes_per_mass":5,
        "toys_per_generating_hypothesis":args.toys,
        "total_profiled_toys":2*len(all_samples)*args.toys,
        "elapsed_seconds":perf_counter()-started,
        "warning":"rough blockwise displacement estimate; discontinuities intentional; not a coverage-certified contour",
    })
    print(args.output_directory)


if __name__ == "__main__":
    main()
