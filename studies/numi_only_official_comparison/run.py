"""Compare fixed-L and public-dk2nu E-L NuMI models with the released grid.

This study deliberately uses profiled observed-data delta chi-square and the
fixed two-parameter 95% threshold 5.99.  It does not calculate CLs and does not
register the approximate NuMI detector model as a production likelihood.
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from sterile_fit.core.profile_three_plus_one import (  # noqa: E402
    profile_s14_s24_at_fixed_sin2_2theta_ee,
    profile_s14_s24_at_fixed_sin2_2theta_mue,
)
from sterile_fit.core.three_plus_one import ThreePlusOneParameters  # noqa: E402
from sterile_fit.experiments.microboone.numi import (  # noqa: E402
    build_diagnostic_numi_workflow, build_energy_baseline_numi_workflow,
)
from sterile_fit.output import result_directory  # noqa: E402


OFFICIAL_DIRECTORY = ROOT / "outputs" / "studies" / "official_grid_wilks" / "legacy" / "results"
NUMI_CONFIG = ROOT / "configs" / "experiments" / "microboone" / "numi" / "analysis.yaml"
ENERGY_BASELINE_INPUT = ROOT / "data/experiments/microboone/numi/derived/public_dk2nu_energy_baseline/psi_exposure_weighted_four_flavours.csv"
THRESHOLD_95 = 5.99


def _map(function, coordinates, workers: int):
    coordinates = list(coordinates)
    if workers == 1:
        return [function(item) for item in coordinates]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(function, coordinates))


def _surface(table: pd.DataFrame, x_name: str, value_name: str):
    pivot = table.pivot(index="delta_m2_41_eV2", columns=x_name, values=value_name)
    return pivot.columns.to_numpy(float), pivot.index.to_numpy(float), pivot.to_numpy(float)


def _draw_panel(axis, fixed, distributed, official, x_name: str, x_label: str, *, heatmap: bool):
    lx, ly, lz = _surface(fixed, x_name, "local_profile_delta_chi2")
    dx, dy, dz = _surface(distributed, x_name, "local_profile_delta_chi2")
    ox, oy, oz = _surface(official, x_name, "official_profile_delta_chi2")
    if heatmap:
        colour = axis.pcolormesh(lx, ly, lz, shading="auto", cmap="viridis", vmin=0.0, vmax=25.0)
    else:
        colour = None
    axis.contour(lx, ly, lz, levels=[THRESHOLD_95], colors=["tab:blue"], linewidths=2.2)
    axis.contour(dx, dy, dz, levels=[THRESHOLD_95], colors=["tab:green"], linewidths=2.2, linestyles="-.")
    axis.contour(ox, oy, oz, levels=[THRESHOLD_95], colors=["tab:orange"], linewidths=2.2, linestyles="--")
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel(x_label)
    axis.set_ylabel(r"$\Delta m^2_{41}\;[\mathrm{eV}^2]$")
    axis.legend(handles=[
        Line2D([0], [0], color="tab:blue", lw=2.2, label="Local NuMI-only approximation"),
        Line2D([0], [0], color="tab:green", lw=2.2, ls="-.", label=r"Local public-dk2nu $E$--$L$ average"),
        Line2D([0], [0], color="tab:orange", lw=2.2, ls="--", label="Official NuMI-only grid"),
    ], fontsize=8)
    return colour


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output-directory", type=Path)
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("workers must be positive")

    config = yaml.safe_load(NUMI_CONFIG.read_text(encoding="utf-8"))
    reference = ThreePlusOneParameters(**{
        key: float(value) for key, value in config["reference_parameters"].items()
    })
    fixed_workflow = build_diagnostic_numi_workflow(
        ROOT / config["diagnostic_four_channel_events"]["kernel_directory"],
        reference,
        float(config["baseline_km"]),
    )
    distributed_workflow = build_energy_baseline_numi_workflow(
        ROOT / config["diagnostic_four_channel_events"]["kernel_directory"],
        reference,
        ENERGY_BASELINE_INPUT,
    )

    official_a = pd.read_csv(OFFICIAL_DIRECTORY / "fig3a_official_profile.csv")
    official_b = pd.read_csv(OFFICIAL_DIRECTORY / "fig3b_official_profile.csv")
    mass_axis = np.sort(official_a["delta_m2_41_eV2"].unique())
    appearance_axis = np.sort(official_a["sin2_2theta_mue"].unique())
    electron_axis = np.sort(official_b["sin2_2theta_ee"].unique())

    def scan(workflow, label):
        def objective(parameters: ThreePlusOneParameters) -> float:
            return workflow.likelihood.chi2(workflow.predictor.predict_total_counts(parameters))
        print(f"{label} Fig. 3a profile: {mass_axis.size} x {appearance_axis.size}", flush=True)
        profiled_a = _map(lambda coordinate: profile_s14_s24_at_fixed_sin2_2theta_mue(
            objective, delta_m2_41_eV2=coordinate[0], sin2_2theta_mue=coordinate[1]
        ), product(mass_axis, appearance_axis), args.workers)
        print(f"{label} Fig. 3b profile: {mass_axis.size} x {electron_axis.size}", flush=True)
        profiled_b = _map(lambda coordinate: profile_s14_s24_at_fixed_sin2_2theta_ee(
            objective, delta_m2_41_eV2=coordinate[0], sin2_2theta_ee=coordinate[1]
        ), product(mass_axis, electron_axis), args.workers)
        minimum = min(min(item.best_fit.chi2 for item in profiled_a), min(item.best_fit.chi2 for item in profiled_b))
        local_a = pd.DataFrame({
        "delta_m2_41_eV2": [item.delta_m2_41_eV2 for item in profiled_a],
        "sin2_2theta_mue": [item.sin2_2theta_mue for item in profiled_a],
        "profiled_sin2_theta14": [item.best_fit.parameters.sin2_theta14 for item in profiled_a],
        "derived_sin2_theta24": [item.best_fit.parameters.sin2_theta24 for item in profiled_a],
        "local_profile_chi2": [item.best_fit.chi2 for item in profiled_a],
        })
        local_b = pd.DataFrame({
        "delta_m2_41_eV2": [item.delta_m2_41_eV2 for item in profiled_b],
        "sin2_2theta_ee": [item.sin2_2theta_ee for item in profiled_b],
        "selected_sin2_theta14_branch": [item.best_fit.parameters.sin2_theta14 for item in profiled_b],
        "profiled_sin2_theta24": [item.best_fit.parameters.sin2_theta24 for item in profiled_b],
        "local_profile_chi2": [item.best_fit.chi2 for item in profiled_b],
        })
        local_a["local_profile_delta_chi2"] = local_a["local_profile_chi2"] - minimum
        local_b["local_profile_delta_chi2"] = local_b["local_profile_chi2"] - minimum
        return local_a, local_b, minimum

    fixed_a, fixed_b, fixed_minimum = scan(fixed_workflow, "Fixed-L")
    distributed_a, distributed_b, distributed_minimum = scan(distributed_workflow, "E-L")

    output = args.output_directory or result_directory(
        "studies", "numi_only_official_comparison", "profiled_delta_chi2"
    )
    output.mkdir(parents=True, exist_ok=False)
    fixed_a.to_csv(output / "fig3a_fixed_baseline.csv", index=False, float_format="%.17g")
    fixed_b.to_csv(output / "fig3b_fixed_baseline.csv", index=False, float_format="%.17g")
    distributed_a.to_csv(output / "fig3a_energy_baseline.csv", index=False, float_format="%.17g")
    distributed_b.to_csv(output / "fig3b_energy_baseline.csv", index=False, float_format="%.17g")

    for heatmap, filename in ((True, "comparison_heatmap.png"), (False, "comparison_lines.png")):
        figure, axes = plt.subplots(1, 2, figsize=(13.5, 5.4))
        colour = _draw_panel(
            axes[0], fixed_a, distributed_a, official_a, "sin2_2theta_mue", r"$\sin^2(2\theta_{\mu e})$", heatmap=heatmap
        )
        axes[0].set_xlim(1e-4, 1.0)
        axes[0].set_ylim(1e-2, 1e2)
        axes[0].set_title("(a) Appearance profile")
        _draw_panel(
            axes[1], fixed_b, distributed_b, official_b, "sin2_2theta_ee", r"$\sin^2(2\theta_{ee})$", heatmap=heatmap
        )
        axes[1].set_xlim(1e-2, 1.0)
        axes[1].set_ylim(1e-1, 14.0)
        axes[1].set_title("(b) Electron-disappearance profile")
        if colour is not None:
            figure.colorbar(colour, ax=axes, label=r"Local profiled $\Delta\chi^2$", shrink=0.88)
        figure.suptitle(r"MicroBooNE NuMI-only, fixed $\Delta\chi^2=5.99$ comparison")
        figure.subplots_adjust(left=0.08, right=0.94, bottom=0.12, top=0.88, wspace=0.25)
        figure.savefig(output / filename, dpi=180, bbox_inches="tight")
        plt.close(figure)

    metadata = {
        "local_status": "diagnostic approximation; borrowed BNB response, fixed aggregate background, no 3--5 GeV response",
        "official_source": "Zenodo 10.5281/zenodo.17161263, NuMI-only observed delta-chi-square grid",
        "statistic": "profiled observed-data delta chi-square",
        "criterion": "fixed delta-chi-square = 5.99 (95%, two-parameter Wilks diagnostic)",
        "cls_used": False,
        "toy_mc_used": False,
        "local_delta_reference": "minimum sampled across both complete local profile tables",
        "fixed_baseline_minimum_chi2": fixed_minimum,
        "energy_baseline_minimum_chi2": distributed_minimum,
        "energy_baseline_input": str(ENERGY_BASELINE_INPUT),
        "grid": {"mass_points": int(mass_axis.size), "fig3a_x_points": int(appearance_axis.size), "fig3b_x_points": int(electron_axis.size)},
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(output, flush=True)


if __name__ == "__main__":
    main()
