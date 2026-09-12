"""Render the fitted equivalent response on two 100 x 100 profile planes."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import sys
import time

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from studies.numi_reco_inverse import run as inverse  # noqa: E402
from sterile_fit.core.likelihood import PredictionScaledGaussianLikelihood  # noqa: E402
from sterile_fit.core.profile_three_plus_one import (  # noqa: E402
    profile_s14_s24_at_fixed_sin2_2theta_ee,
    profile_s14_s24_at_fixed_sin2_2theta_mue,
)
from sterile_fit.experiments.microboone.numi import NumiFourChannelEmpiricalKernel  # noqa: E402
from sterile_fit.experiments.microboone.public_data import NUMI_FOUR_CHANNELS, load_numi_four_channel_inputs  # noqa: E402

_OBJECTIVE = None


def _initialize(coefficients_path: str, modes: int, maximum_change: float):
    global _OBJECTIVE
    kernel = NumiFourChannelEmpiricalKernel.from_directory(inverse.KERNEL_DIRECTORY)
    inputs = load_numi_four_channel_inputs()
    items = [inverse._make_channel_perturbation(
        index, channel.identifier, kernel,
        modes_reco=modes, modes_true=modes,
        maximum_fractional_change=maximum_change,
    ) for index, channel in enumerate(NUMI_FOUR_CHANNELS)]
    coefficients = np.loadtxt(coefficients_path, delimiter=",").ravel()
    predict, _ = inverse._prediction_function(kernel, items, coefficients, 0.680)
    likelihood = PredictionScaledGaussianLikelihood(
        inputs.observed_counts, inputs.published_total_prediction_counts, inputs.systematic_covariance
    )
    _OBJECTIVE = lambda parameters: likelihood.chi2(predict(parameters))


def _point(task):
    panel, mass, amplitude = task
    if panel == "a":
        result = profile_s14_s24_at_fixed_sin2_2theta_mue(
            _OBJECTIVE, delta_m2_41_eV2=mass, sin2_2theta_mue=amplitude
        )
    else:
        result = profile_s14_s24_at_fixed_sin2_2theta_ee(
            _OBJECTIVE, delta_m2_41_eV2=mass, sin2_2theta_ee=amplitude
        )
    return result.best_fit.chi2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fit-directory", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--modes", type=int, default=2)
    parser.add_argument("--maximum-fractional-change", type=float, default=0.20)
    args = parser.parse_args()
    ma, xa = np.geomspace(1e-2, 1e2, 100), np.geomspace(1e-4, np.nextafter(1.0, 0.0), 100)
    mb, xb = np.geomspace(1e-1, 14.0, 100), np.geomspace(1e-2, np.nextafter(1.0, 0.0), 100)
    tasks_a = [("a", m, x) for m in ma for x in xa]
    tasks_b = [("b", m, x) for m in mb for x in xb]
    started = time.perf_counter()
    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=_initialize,
        initargs=(str(args.fit_directory / "coefficients_checkpoint.csv"), args.modes, args.maximum_fractional_change),
    ) as pool:
        va = list(pool.map(_point, tasks_a, chunksize=20))
        print(f"Fig3a 100x100 complete; elapsed={(time.perf_counter()-started)/60:.1f} min", flush=True)
        vb = list(pool.map(_point, tasks_b, chunksize=20))
    za, zb = np.asarray(va).reshape(100, 100), np.asarray(vb).reshape(100, 100)
    minimum = min(float(za.min()), float(zb.min()))
    za -= minimum; zb -= minimum
    pd.DataFrame({"delta_m2_41_eV2": np.repeat(ma, 100), "sin2_2theta_mue": np.tile(xa, 100), "candidate_profile_delta_chi2": za.ravel()}).to_csv(args.fit_directory / "fig3a_candidate_100x100.csv", index=False)
    pd.DataFrame({"delta_m2_41_eV2": np.repeat(mb, 100), "sin2_2theta_ee": np.tile(xb, 100), "candidate_profile_delta_chi2": zb.ravel()}).to_csv(args.fit_directory / "fig3b_candidate_100x100.csv", index=False)
    oxa, oma, oza = inverse._surface(inverse.OFFICIAL_DIRECTORY / "fig3a_official_profile.csv", "sin2_2theta_mue", "official_profile_delta_chi2")
    oxb, omb, ozb = inverse._surface(inverse.OFFICIAL_DIRECTORY / "fig3b_official_profile.csv", "sin2_2theta_ee", "official_profile_delta_chi2")
    for heatmap, name in ((True, "candidate_vs_official_100x100_heatmap.png"), (False, "candidate_vs_official_100x100_lines.png")):
        fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4))
        colour = None
        for axis, x, m, z, ox, om, oz, label in (
            (axes[0], xa, ma, za, oxa, oma, oza, r"$\sin^2(2\theta_{\mu e})$"),
            (axes[1], xb, mb, zb, oxb, omb, ozb, r"$\sin^2(2\theta_{ee})$"),
        ):
            if heatmap:
                colour = axis.pcolormesh(x, m, z, shading="auto", cmap="viridis", vmin=0, vmax=25)
            axis.contour(x, m, z, levels=[5.99], colors="tab:blue", linewidths=2.2)
            axis.contour(ox, om, oz, levels=[5.99], colors="tab:orange", linestyles="--", linewidths=2.2)
            axis.set_xscale("log"); axis.set_yscale("log"); axis.set_xlabel(label)
            axis.set_ylabel(r"$\Delta m^2_{41}\,[\mathrm{eV}^2]$")
            axis.legend(handles=[Line2D([0],[0],color="tab:blue",lw=2.2,label="Constrained equivalent Reco"), Line2D([0],[0],color="tab:orange",ls="--",lw=2.2,label="Official NuMI-only grid")], fontsize=8)
        axes[0].set_xlim(1e-4,1); axes[0].set_ylim(1e-2,1e2)
        axes[1].set_xlim(1e-2,1); axes[1].set_ylim(1e-1,14)
        if colour is not None: fig.colorbar(colour, ax=axes, label=r"profiled $\Delta\chi^2$")
        fig.subplots_adjust(left=.08,right=.94,bottom=.12,top=.9,wspace=.26)
        fig.savefig(args.fit_directory / name, dpi=180, bbox_inches="tight")
        plt.close(fig)
    print(f"complete; elapsed={(time.perf_counter()-started)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
