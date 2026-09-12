"""Convert saved NuMI-only observed profiles to fixed-hypothesis quadratic CLs.

No observed-data profile or observed chi-square is recomputed.  The saved
profiled parameters and chi-square values are reused exactly; predictions and
covariances are constructed only to obtain the fixed-hypothesis T laws.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from sterile_fit.core.calibration import GaussianHypothesis, quadratic_cls
from sterile_fit.core.three_plus_one import ThreePlusOneParameters
from sterile_fit.experiments.microboone.numi import build_diagnostic_numi_workflow

SAVED = ROOT / "outputs/studies/numi_only_official_comparison/20260906T052358388810Z/profiled_delta_chi2"
BNB = ROOT / "outputs/microboone_bnb/three_plus_one/bnb_only_error_analysis_20260906"
JOINT = ROOT / "outputs/microboone_bnb_numi_joint/three_plus_one/quadratic_non_toy_20260904"
OUT = SAVED / "saved_profile_quadratic_cls"
CONFIG = ROOT / "configs/experiments/microboone/numi/analysis.yaml"


def surface(frame: pd.DataFrame, xcol: str, zcol: str):
    pivot = frame.pivot(index="fixed_delta_m2_41_eV2", columns=xcol, values=zcol)
    return pivot.columns.to_numpy(float), pivot.index.to_numpy(float), pivot.to_numpy(float)


def numi_surface(frame: pd.DataFrame, xcol: str):
    pivot = frame.pivot(index="delta_m2_41_eV2", columns=xcol, values="cls_quadratic")
    return pivot.columns.to_numpy(float), pivot.index.to_numpy(float), pivot.to_numpy(float)


def main() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    null = ThreePlusOneParameters(**{k: float(v) for k, v in config["reference_parameters"].items()})
    workflow = build_diagnostic_numi_workflow(
        ROOT / config["diagnostic_four_channel_events"]["kernel_directory"],
        null,
        float(config["baseline_km"]),
    )
    null_mean = workflow.predictor.predict_total_counts(null)
    null_hypothesis = GaussianHypothesis(null_mean, workflow.likelihood.covariance_for_prediction(null_mean))
    null_chi2 = workflow.likelihood.chi2(null_mean)

    def convert(row, mode: str):
        if mode == "fig3a":
            parameters = ThreePlusOneParameters(
                float(row.delta_m2_41_eV2), float(row.profiled_sin2_theta14), float(row.derived_sin2_theta24)
            )
        else:
            parameters = ThreePlusOneParameters(
                float(row.delta_m2_41_eV2), float(row.selected_sin2_theta14_branch), float(row.profiled_sin2_theta24)
            )
        mean = workflow.predictor.predict_total_counts(parameters)
        tested = GaussianHypothesis(mean, workflow.likelihood.covariance_for_prediction(mean))
        observed_t = float(row.local_profile_chi2) - null_chi2
        result = quadratic_cls(observed_t, [(null_hypothesis, tested)])
        return result.p_value_3nu, result.p_value_4nu, result.cls, observed_t

    specs = (
        ("fig3a", "fig3a_local_numi_only.csv", "sin2_2theta_mue"),
        ("fig3b", "fig3b_local_numi_only.csv", "sin2_2theta_ee"),
    )
    outputs = {}
    OUT.mkdir(parents=True, exist_ok=True)
    for mode, filename, xcol in specs:
        saved_cls = OUT / f"{mode}_numi_only_cls.csv"
        if saved_cls.exists():
            outputs[mode] = (pd.read_csv(saved_cls), xcol)
            continue
        frame = pd.read_csv(SAVED / filename)
        with ThreadPoolExecutor(max_workers=8) as executor:
            converted = list(executor.map(lambda row: convert(row, mode), frame.itertuples(index=False)))
        values = np.asarray(converted)
        frame[["p_value_3nu_quadratic", "p_value_4nu_quadratic", "cls_quadratic", "observed_T_saved_chi2"]] = values
        frame.to_csv(OUT / f"{mode}_numi_only_cls.csv", index=False, float_format="%.17g")
        outputs[mode] = (frame, xcol)

    figure, axes = plt.subplots(1, 2, figsize=(13.5, 5.4), constrained_layout=True)
    plot_specs = (
        ("fig3a", "scan_fig3a_analytic", "scan_fig3a_analytic", r"$\sin^2(2\theta_{\mu e})$"),
        ("fig3b", "scan_fig3b_analytic", "scan_electron-disfig3a_analytic", r"$\sin^2(2\theta_{ee})$"),
    )
    for axis, (mode, bnb_dir, joint_dir, xlabel) in zip(axes, plot_specs):
        bnb = pd.read_csv(BNB / bnb_dir / "result.csv")
        joint = pd.read_csv(JOINT / joint_dir / "result.csv")
        xcol = "fixed_sin2_2theta_mue" if mode == "fig3a" else "fixed_sin2_2theta_ee"
        bx, by, bz = surface(bnb, xcol, "cls_quadratic")
        jx, jy, jz = surface(joint, xcol, "cls_quadratic")
        numi, numi_xcol = outputs[mode]
        nx, ny, nz = numi_surface(numi, numi_xcol)
        colour = axis.pcolormesh(jx, jy, jz, shading="nearest", cmap="viridis", vmin=0.0, vmax=1.0)
        axis.contour(bx, by, bz, levels=[0.05], colors="dodgerblue", linewidths=2.2)
        axis.contour(nx, ny, nz, levels=[0.05], colors="darkorange", linewidths=2.2, linestyles="--")
        axis.contour(jx, jy, jz, levels=[0.05], colors="red", linewidths=2.2)
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlabel(xlabel)
        axis.set_ylabel(r"$\Delta m^2_{41}\,[\mathrm{eV}^2]$")
        axis.set_title("Fig. 3a" if mode == "fig3a" else "Fig. 3b")
        axis.set_xlim((1e-4, 1.0) if mode == "fig3a" else (1e-2, 1.0))
        axis.set_ylim((1e-2, 1e2) if mode == "fig3a" else (1e-1, 14.0))
        axis.legend(handles=[
            Line2D([0], [0], color="dodgerblue", lw=2.2, label="BNB-only"),
            Line2D([0], [0], color="darkorange", lw=2.2, ls="--", label="NuMI-only approximation"),
            Line2D([0], [0], color="red", lw=2.2, label="Saved BNB+NuMI joint"),
        ], fontsize=8)
        figure.colorbar(colour, ax=axis, label=r"Saved joint $CL_s$ (linear 0--1)")
    figure.savefig(OUT / "bnb_numi_joint_three_cls_contours_linear_heatmap.png", dpi=190)
    plt.close(figure)
    (OUT / "metadata.json").write_text(json.dumps({
        "observed_profiles_recomputed": False,
        "observed_chi2_recomputed": False,
        "numi_input": str(SAVED),
        "calibration": "fixed-hypothesis generalized quadratic-form CLs",
        "heatmap": "saved BNB+NuMI joint CLs, linear range 0 to 1",
        "warning": "NuMI-only remains a diagnostic approximation using the borrowed BNB response.",
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
