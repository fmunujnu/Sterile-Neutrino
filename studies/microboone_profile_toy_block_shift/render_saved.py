from __future__ import annotations

from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SPECS = {
    "fig3a": {
        "path": ROOT / "outputs/microboone_bnb_numi_joint/three_plus_one/fullgrid_fixed_toy5000_20260904/scan_fig3a_toy/result.csv",
        "x": "fixed_sin2_2theta_mue", "xlabel": r"$\sin^2(2\theta_{\mu e})$",
    },
    "fig3b": {
        "path": ROOT / "outputs/microboone_bnb_numi_joint/three_plus_one/fullgrid_fixed_toy5000_20260904/scan_electron-disfig3a_toy/result.csv",
        "x": "fixed_sin2_2theta_ee", "xlabel": r"$\sin^2(2\theta_{ee})$",
    },
}


def baseline(frame: pd.DataFrame, x_column: str) -> pd.DataFrame:
    rows = []
    for mass, group in frame.groupby("fixed_delta_m2_41_eV2"):
        group = group.sort_values(x_column)
        q = group.cls_toy.to_numpy() - 0.05
        x = group[x_column].to_numpy(dtype=float)
        crossings = np.flatnonzero(q[:-1] * q[1:] <= 0)
        if not crossings.size:
            continue
        i = crossings[np.argmin(abs(q[crossings]) + abs(q[crossings + 1]))]
        fraction = -q[i] / (q[i + 1] - q[i]) if q[i + 1] != q[i] else 0.5
        amplitude = 10 ** (np.log10(x[i]) + fraction * (np.log10(x[i + 1]) - np.log10(x[i])))
        rows.append((float(mass), float(amplitude)))
    return pd.DataFrame(rows, columns=["mass", "baseline_amplitude"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    samples_all = pd.read_csv(args.directory / "sample_points.csv")
    shifts = []

    for name, spec in SPECS.items():
        frame = pd.read_csv(spec["path"])
        base = baseline(frame, spec["x"])
        figure, axis = plt.subplots(figsize=(7.5, 5.8), constrained_layout=True)
        axis.plot(base.baseline_amplitude, base.mass, color="tab:red", linewidth=2,
                  label="Existing fixed-Toy contour")
        shown = samples_all[samples_all.figure == name]
        first_valid = True
        first_invalid = True
        for block, points in shown.groupby("block"):
            lo, hi = points.mass.min(), points.mass.max()
            segment = base[(base.mass >= lo) & (base.mass <= hi)].copy()
            design = np.c_[np.ones(len(points)), np.log10(points.mass), np.log10(points.amplitude)]
            coefficients, *_ = np.linalg.lstsq(design, points.q, rcond=None)
            predicted = -(coefficients[0] + coefficients[1] * np.log10(segment.mass)) / coefficients[2]
            shifts_at_mass = predicted - np.log10(segment.baseline_amplitude)
            shift = float(np.median(shifts_at_mass))
            sampled_limit = float(np.max(np.abs(np.log10(points.amplitude / points.baseline_amplitude))))
            valid = np.isfinite(shift) and abs(shift) <= sampled_limit
            rms = float(np.sqrt(np.mean((design @ coefficients - points.q) ** 2)))
            shifts.append({"figure": name, "block": int(block), "mass_low": lo, "mass_high": hi,
                           "log10_amplitude_shift": shift if valid else np.nan,
                           "amplitude_factor": 10**shift if valid else np.nan,
                           "fit_q_rms": rms, "status": "estimated" if valid else "unresolved_outside_sample_band"})
            if valid:
                axis.plot(segment.baseline_amplitude * 10**shift, segment.mass,
                          color="tab:blue", linewidth=2,
                          label="Coarse block-shift estimate" if first_valid else None)
                first_valid = False
            else:
                axis.plot(segment.baseline_amplitude, segment.mass, color="tab:orange",
                          linewidth=3, linestyle=":",
                          label="Unresolved block" if first_invalid else None)
                first_invalid = False
            amp_lo, amp_hi = points.amplitude.min(), points.amplitude.max()
            axis.plot([amp_lo, amp_hi, amp_hi, amp_lo, amp_lo], [lo, lo, hi, hi, lo],
                      color="0.5", linewidth=.7, alpha=.7)

        axis.scatter(shown.amplitude, shown.mass, s=8, color="black", alpha=.55,
                     label=f"Profiled-Toy samples ({int(shown.toys_per_hypothesis.iloc[0])}/hypothesis)")
        axis.set_xscale("log"); axis.set_yscale("log")
        axis.set_xlabel(spec["xlabel"]); axis.set_ylabel(r"$\Delta m^2_{41}\;[\mathrm{eV}^2]$")
        axis.set_title(f"MicroBooNE {name}: contour-following block shifts")
        axis.legend(fontsize=8); axis.grid(False)
        figure.savefig(args.directory / f"{name}_block_shift.png", dpi=180, bbox_inches="tight")
        plt.close(figure)

    pd.DataFrame(shifts).to_csv(args.directory / "block_shifts.csv", index=False)


if __name__ == "__main__":
    main()
