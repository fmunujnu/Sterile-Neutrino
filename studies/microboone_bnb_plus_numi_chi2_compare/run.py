"""Compare BNB+official-NuMI and BNB+local-NuMI saved chi-square surfaces."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

BNB = ROOT / "outputs/microboone_bnb/three_plus_one/bnb_only_error_analysis_20260906"
NUMI_LOCAL = ROOT / "outputs/studies/numi_only_official_comparison/energy_baseline_current"
NUMI_OFFICIAL = ROOT / "outputs/studies/official_grid_wilks/legacy/results"
OUT = ROOT / "outputs/studies/microboone_bnb_plus_numi_chi2_compare/final_saved_surfaces"
LEVEL = 5.99


def _surface(frame: pd.DataFrame, mass: str, amplitude: str, value: str):
    table = frame.pivot(index=mass, columns=amplitude, values=value).sort_index().sort_index(axis=1)
    return table.index.to_numpy(float), table.columns.to_numpy(float), table.to_numpy(float)


def _interpolate_bnb(frame: pd.DataFrame, x_name: str, target_mass, target_x):
    mass, x, values = _surface(frame, "fixed_delta_m2_41_eV2", x_name, "chi2")
    interpolator = RegularGridInterpolator(
        (np.log10(mass), np.log10(x)), values, bounds_error=True
    )
    mm, xx = np.meshgrid(target_mass, target_x, indexing="ij")
    points = np.column_stack((np.log10(mm.ravel()), np.log10(xx.ravel())))
    return interpolator(points).reshape(mm.shape)


def _draw(panel: str, x_name: str, xlabel: str, bnb_directory: str,
          local_file: str, official_file: str, ylim: tuple[float, float]):
    bnb = pd.read_csv(BNB / bnb_directory / "result.csv")
    local = pd.read_csv(NUMI_LOCAL / local_file)
    official = pd.read_csv(NUMI_OFFICIAL / official_file)
    mass, x, local_values = _surface(
        local, "delta_m2_41_eV2", x_name, "local_profile_delta_chi2"
    )
    official_mass, official_x, official_values = _surface(
        official, "delta_m2_41_eV2", x_name, "official_profile_delta_chi2"
    )
    if not np.allclose(mass, official_mass, rtol=1e-12, atol=0.0) or not np.allclose(
        x, official_x, rtol=1e-12, atol=0.0
    ):
        raise ValueError(f"{panel}: local and official NuMI grids differ")
    selected = (mass >= ylim[0]) & (mass <= ylim[1])
    mass = mass[selected]
    local_values = local_values[selected]
    official_values = official_values[selected]
    bnb_values = _interpolate_bnb(bnb, f"fixed_{x_name}", mass, x)
    combined_local = bnb_values + local_values
    combined_official = bnb_values + official_values
    combined_local -= np.nanmin(combined_local)
    combined_official -= np.nanmin(combined_official)

    rows = []
    for source, values in (("BNB_plus_local_NuMI", combined_local),
                           ("BNB_plus_official_NuMI", combined_official)):
        rows.append(pd.DataFrame({
            "panel": panel, "source": source,
            "delta_m2_41_eV2": np.repeat(mass, len(x)),
            x_name: np.tile(x, len(mass)), "combined_delta_chi2": values.ravel(),
        }))

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.4), constrained_layout=True)
    for axis, title, values in zip(
        axes,
        ("BNB + official NuMI", r"BNB + local NuMI $q(L|E,\nu)$ weighted"),
        (combined_official, combined_local),
    ):
        colour = axis.pcolormesh(x, mass, values, shading="nearest", cmap="viridis", vmin=0, vmax=25)
        axis.contour(x, mass, values, levels=[LEVEL], colors="red", linewidths=2.1)
        axis.set(xscale="log", yscale="log", xlabel=xlabel,
                 ylabel=r"$\Delta m^2_{41}\,[\mathrm{eV}^2]$", title=title)
        axis.set_xlim(x[0], x[-1]); axis.set_ylim(*ylim)
        fig.colorbar(colour, ax=axis, label=r"combined $\Delta\chi^2$")
    fig.savefig(OUT / f"{panel}_two_combined_chi2_heatmaps.png", dpi=190)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7.4, 5.6), constrained_layout=True)
    axis.contour(x, mass, combined_official, levels=[LEVEL], colors="tab:blue", linewidths=2.2)
    axis.contour(x, mass, combined_local, levels=[LEVEL], colors="tab:orange", linewidths=2.2, linestyles="--")
    axis.set(xscale="log", yscale="log", xlabel=xlabel,
             ylabel=r"$\Delta m^2_{41}\,[\mathrm{eV}^2]$",
             title=f"MicroBooNE Fig. 3{panel[-1]} saved-surface comparison")
    axis.set_xlim(x[0], x[-1]); axis.set_ylim(*ylim)
    axis.legend(handles=[
        Line2D([0], [0], color="tab:blue", lw=2.2, label=rf"BNB + official NuMI, $\Delta\chi^2={LEVEL}$"),
        Line2D([0], [0], color="tab:orange", ls="--", lw=2.2, label=rf"BNB + local NuMI $q(L|E,\nu)$, $\Delta\chi^2={LEVEL}$"),
    ], fontsize=8)
    axis.grid(alpha=.15)
    fig.savefig(OUT / f"{panel}_combined_chi2_contours.png", dpi=190)
    plt.close(fig)
    return pd.concat(rows, ignore_index=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tables = [
        _draw("fig3a", "sin2_2theta_mue", r"$\sin^2(2\theta_{\mu e})$",
              "scan_fig3a_analytic", "fig3a_energy_baseline.csv",
              "fig3a_official_profile.csv", (1e-2, 1e2)),
        _draw("fig3b", "sin2_2theta_ee", r"$\sin^2(2\theta_{ee})$",
              "scan_fig3b_analytic", "fig3b_energy_baseline.csv",
              "fig3b_official_profile.csv", (1e-1, 14.0)),
    ]
    pd.concat(tables, ignore_index=True).to_csv(OUT / "combined_surfaces.csv", index=False)
    (OUT / "metadata.json").write_text(json.dumps({
        "calculation": "saved BNB chi2 plus saved q(L|E,nu)-weighted NuMI profile surface; subtract each combined global minimum",
        "contour_level": LEVEL,
        "new_fit_or_toy": False,
        "numi_weighting_note": "the local q(L|E,nu) table is exactly the NuMI-only energy_baseline_current surface; this script introduces no second baseline average",
        "bnb_interpolation": "linear in log10 mass and log10 amplitude onto the common 100x61 NuMI grid",
        "warning": "apparent agreement can improve because a common BNB surface is added and each result is re-zeroed; diagnostic chi-square comparison only, not CLs and not the official BNB+NuMI joint likelihood",
    }, indent=2) + "\n", encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
