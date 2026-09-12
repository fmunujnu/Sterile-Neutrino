from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm


ROOT = Path(__file__).resolve().parents[2]
BNB_ROOT = ROOT / "outputs/microboone_bnb/three_plus_one/bnb_only_error_analysis_20260906"
JOINT_ROOT = ROOT / "outputs/microboone_bnb_numi_joint/three_plus_one/quadratic_non_toy_20260904"
OUT = BNB_ROOT / "bnb_vs_joint_impact"

SPECS = (
    ("Fig. 3a", "scan_fig3a_analytic", "scan_fig3a_analytic", "fixed_sin2_2theta_mue", r"$\sin^2(2\theta_{\mu e})$"),
    ("Fig. 3b", "scan_fig3b_analytic", "scan_electron-disfig3a_analytic", "fixed_sin2_2theta_ee", r"$\sin^2(2\theta_{ee})$"),
)


def grid(frame: pd.DataFrame, xcol: str, value: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    table = frame.pivot(index="fixed_delta_m2_41_eV2", columns=xcol, values=value)
    return table.columns.to_numpy(float), table.index.to_numpy(float), table.to_numpy(float)


def first_crossing(x: np.ndarray, values: np.ndarray, level: float = 0.05) -> float:
    order = np.argsort(x)
    x, values = x[order], values[order]
    for index in range(len(x) - 1):
        y0, y1 = values[index] - level, values[index + 1] - level
        if y0 == 0:
            return float(x[index])
        if y0 * y1 < 0 or y1 == 0:
            lx0, lx1 = np.log10(x[index]), np.log10(x[index + 1])
            fraction = -y0 / (y1 - y0)
            return float(10 ** (lx0 + fraction * (lx1 - lx0)))
    return float("nan")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)
    overlay_fig, overlay_axes = plt.subplots(1, 2, figsize=(13, 5.2), constrained_layout=True)
    lines_fig, lines_axes = plt.subplots(1, 2, figsize=(13, 5.2), constrained_layout=True)
    summaries: dict[str, dict] = {}
    contour_rows = []
    for row, (label, bnb_dir, joint_dir, xcol, xlabel) in enumerate(SPECS):
        bnb = pd.read_csv(BNB_ROOT / bnb_dir / "result.csv")
        joint = pd.read_csv(JOINT_ROOT / joint_dir / "result.csv")
        keys = ["fixed_delta_m2_41_eV2", xcol]
        merged = bnb.merge(joint, on=keys, suffixes=("_bnb", "_joint"), validate="one_to_one")
        x, mass, cls_bnb = grid(bnb, xcol, "cls_quadratic")
        xj, massj, cls_joint = grid(joint, xcol, "cls_quadratic")
        if not (np.allclose(x, xj) and np.allclose(mass, massj)):
            raise RuntimeError(f"{label}: BNB and joint grids differ")

        left = axes[row, 0]
        mesh = left.pcolormesh(x, mass, np.clip(cls_bnb, 0, 0.1), shading="nearest", cmap="viridis", vmin=0, vmax=0.1)
        left.contour(x, mass, cls_bnb, levels=[0.05], colors="red", linewidths=2)
        fig.colorbar(mesh, ax=left, label=r"BNB-only $CL_s$ (clipped at 0.1)")
        left.set_title(f"{label}: BNB-only")

        delta = np.log10(np.clip(cls_joint, 1e-8, 1) / np.clip(cls_bnb, 1e-8, 1))
        limit = max(0.1, float(np.nanpercentile(np.abs(delta), 95)))
        right = axes[row, 1]
        impact = right.pcolormesh(x, mass, delta, shading="nearest", cmap="coolwarm", norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit))
        right.contour(x, mass, cls_bnb, levels=[0.05], colors="dodgerblue", linewidths=2)
        right.contour(x, mass, cls_joint, levels=[0.05], colors="red", linewidths=2)
        fig.colorbar(impact, ax=right, label=r"$\log_{10}(CL_s^{\rm joint}/CL_s^{\rm BNB})$")
        right.plot([], [], color="dodgerblue", linewidth=2, label="BNB-only 95%")
        right.plot([], [], color="red", linewidth=2, label="BNB+NuMI 95%")
        right.legend()
        right.set_title(f"{label}: effective NuMI + cross-covariance impact")

        overlay = overlay_axes[row]
        overlay_mesh = overlay.pcolormesh(
            x, mass, np.clip(cls_bnb, 0, 0.1), shading="nearest", cmap="viridis", vmin=0, vmax=0.1
        )
        overlay.contour(x, mass, cls_bnb, levels=[0.05], colors="dodgerblue", linewidths=2.2)
        overlay.contour(x, mass, cls_joint, levels=[0.05], colors="red", linewidths=2.2)
        overlay.plot([], [], color="dodgerblue", linewidth=2.2, label="BNB-only 95%")
        overlay.plot([], [], color="red", linewidth=2.2, label="Saved BNB+NuMI 95%")
        overlay.legend()
        overlay.set_title(f"{label}: BNB-only map with saved joint contour")
        overlay_fig.colorbar(overlay_mesh, ax=overlay, label=r"BNB-only $CL_s$ (clipped at 0.1)")

        line_axis = lines_axes[row]
        line_axis.contour(x, mass, cls_bnb, levels=[0.05], colors="dodgerblue", linewidths=2.2)
        line_axis.contour(x, mass, cls_joint, levels=[0.05], colors="red", linewidths=2.2)
        line_axis.plot([], [], color="dodgerblue", linewidth=2.2, label="BNB-only 95%")
        line_axis.plot([], [], color="red", linewidth=2.2, label="Saved BNB+NuMI 95%")
        line_axis.legend()
        line_axis.set_title(f"{label}: contour-only comparison")

        for axis in (left, right):
            axis.set_xscale("log")
            axis.set_yscale("log")
            axis.set_xlabel(xlabel)
            axis.set_ylabel(r"$\Delta m^2_{41}\ [{\rm eV}^2]$")
        for axis in (overlay, line_axis):
            axis.set_xscale("log")
            axis.set_yscale("log")
            axis.set_xlabel(xlabel)
            axis.set_ylabel(r"$\Delta m^2_{41}\ [{\rm eV}^2]$")

        bnb_excluded = cls_bnb <= 0.05
        joint_excluded = cls_joint <= 0.05
        crossings = []
        for mass_value in mass:
            b = bnb[bnb.fixed_delta_m2_41_eV2 == mass_value]
            j = joint[joint.fixed_delta_m2_41_eV2 == mass_value]
            cb = first_crossing(b[xcol].to_numpy(float), b.cls_quadratic.to_numpy(float))
            cj = first_crossing(j[xcol].to_numpy(float), j.cls_quadratic.to_numpy(float))
            contour_rows.append({"figure": label, "mass": mass_value, "bnb_crossing": cb, "joint_crossing": cj,
                                 "joint_over_bnb": cj / cb if np.isfinite(cb) and np.isfinite(cj) else np.nan})
            if np.isfinite(cb) and np.isfinite(cj):
                crossings.append(cj / cb)
        ratios = np.asarray(crossings)
        summaries[label] = {
            "grid_points": int(cls_bnb.size),
            "bnb_excluded_fraction": float(bnb_excluded.mean()),
            "joint_excluded_fraction": float(joint_excluded.mean()),
            "classification_changed_fraction": float((bnb_excluded != joint_excluded).mean()),
            "joint_newly_excluded_fraction": float((~bnb_excluded & joint_excluded).mean()),
            "joint_newly_allowed_fraction": float((bnb_excluded & ~joint_excluded).mean()),
            "matched_contour_mass_slices": int(ratios.size),
            "median_joint_over_bnb_crossing": float(np.median(ratios)) if ratios.size else None,
            "crossing_ratio_16_84_percentiles": [float(v) for v in np.percentile(ratios, [16, 84])] if ratios.size else None,
            "median_delta_chi2_joint_minus_bnb_at_matched_profile_coordinates": float(np.median(
                merged.test_statistic_chi2_4nu_minus_chi2_3nu_joint - merged.test_statistic_chi2_4nu_minus_chi2_3nu_bnb
            )),
        }

    fig.savefig(OUT / "bnb_only_and_joint_impact.png", dpi=190)
    plt.close(fig)
    overlay_fig.savefig(OUT / "bnb_only_with_saved_joint_contours.png", dpi=190)
    plt.close(overlay_fig)
    lines_fig.savefig(OUT / "bnb_only_vs_saved_joint_lines.png", dpi=190)
    plt.close(lines_fig)
    pd.DataFrame(contour_rows).to_csv(OUT / "contour_crossing_comparison.csv", index=False)
    (OUT / "summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
