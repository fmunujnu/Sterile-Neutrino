"""Render saved official/local NuMI-only chi2 surfaces with the proven layout."""

from __future__ import annotations

from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
LOCAL = ROOT / "outputs/studies/numi_only_official_comparison/energy_baseline_current"
FIXED = ROOT / "outputs/studies/numi_only_official_comparison/20260912T083600638799Z/profiled_delta_chi2"
OFFICIAL = ROOT / "outputs/studies/official_grid_wilks/legacy/results"
OUT = ROOT / "outputs/studies/numi_only_official_comparison/numi_only_saved_surfaces"
LEVEL = 5.99


def _surface(frame: pd.DataFrame, amplitude: str, value: str):
    table = frame.pivot(
        index="delta_m2_41_eV2", columns=amplitude, values=value
    ).sort_index().sort_index(axis=1)
    return table.index.to_numpy(float), table.columns.to_numpy(float), table.to_numpy(float)


def _draw(panel: str, x_name: str, xlabel: str, local_file: str,
          official_file: str, ylim: tuple[float, float]) -> None:
    local = pd.read_csv(LOCAL / local_file)
    official = pd.read_csv(OFFICIAL / official_file)
    mass, x, local_values = _surface(local, x_name, "local_profile_delta_chi2")
    official_mass, official_x, official_values = _surface(
        official, x_name, "official_profile_delta_chi2"
    )
    if not np.allclose(mass, official_mass, rtol=1e-12, atol=0.0) or not np.allclose(
        x, official_x, rtol=1e-12, atol=0.0
    ):
        raise ValueError(f"{panel}: local and official NuMI grids differ")
    selected = (mass >= ylim[0]) & (mass <= ylim[1])
    mass = mass[selected]
    local_values = local_values[selected]
    official_values = official_values[selected]

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.4), constrained_layout=True)
    for axis, title, values in zip(
        axes,
        ("Official NuMI-only", r"Local NuMI-only $q(L|E,\nu)$ weighted"),
        (official_values, local_values),
    ):
        colour = axis.pcolormesh(
            x, mass, values, shading="nearest", cmap="viridis", vmin=0, vmax=25
        )
        axis.contour(x, mass, values, levels=[LEVEL], colors="red", linewidths=2.1)
        axis.set(
            xscale="log", yscale="log", xlabel=xlabel,
            ylabel=r"$\Delta m^2_{41}\,[\mathrm{eV}^2]$", title=title,
        )
        axis.set_xlim(x[0], x[-1])
        axis.set_ylim(*ylim)
        fig.colorbar(colour, ax=axis, label=r"NuMI-only profiled $\Delta\chi^2$")
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{panel}_two_numi_only_chi2_heatmaps.png", dpi=190)
    plt.close(fig)


def _load_panel(panel: str, x_name: str, local_file: str, official_file: str,
                ylim: tuple[float, float], local_root: Path = LOCAL):
    local = pd.read_csv(local_root / local_file)
    official = pd.read_csv(OFFICIAL / official_file)
    mass, x, local_values = _surface(local, x_name, "local_profile_delta_chi2")
    official_mass, official_x, official_values = _surface(
        official, x_name, "official_profile_delta_chi2"
    )
    if not np.allclose(mass, official_mass, rtol=1e-12, atol=0.0) or not np.allclose(
        x, official_x, rtol=1e-12, atol=0.0
    ):
        raise ValueError(f"{panel}: local and official NuMI grids differ")
    selected = (mass >= ylim[0]) & (mass <= ylim[1])
    return x, mass[selected], official_values[selected], local_values[selected]


def _local_values_on_official_contour(x: np.ndarray, official: np.ndarray,
                                      local: np.ndarray) -> np.ndarray:
    values: list[float] = []
    for official_row, local_row in zip(official, local):
        difference = official_row - LEVEL
        for index in np.flatnonzero(difference[:-1] * difference[1:] <= 0.0):
            denominator = official_row[index + 1] - official_row[index]
            if denominator == 0.0:
                continue
            weight = (LEVEL - official_row[index]) / denominator
            values.append(local_row[index] + weight * (local_row[index + 1] - local_row[index]))
    return np.asarray(values, dtype=float)


def _draw_four_panel(panels, filename: str, *, local_shift: float = 0.0,
                     title: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 10.0), constrained_layout=True)
    for row, (panel, x_name, xlabel, _, _, ylim) in enumerate(panels):
        x, mass, official, local = _load_panel(panel, x_name, panels[row][3], panels[row][4], ylim)
        for column, panel_title, values in (
            (0, "Official NuMI-only", official),
            (1, r"Local NuMI-only $q(L|E,\nu)$ weighted", local + local_shift),
        ):
            axis = axes[row, column]
            colour = axis.pcolormesh(
                x, mass, values, shading="nearest", cmap="viridis", vmin=0, vmax=25
            )
            axis.contour(x, mass, values, levels=[LEVEL], colors="red", linewidths=2.1)
            axis.set(
                xscale="log", yscale="log", xlabel=xlabel,
                ylabel=r"$\Delta m^2_{41}\,[\mathrm{eV}^2]$",
                title=f"{panel}: {panel_title}",
            )
            axis.set_xlim(x[0], x[-1])
            axis.set_ylim(*ylim)
            fig.colorbar(colour, ax=axis, label=r"NuMI-only profiled $\Delta\chi^2$")
    fig.suptitle(title)
    fig.savefig(OUT / filename, dpi=190)
    plt.close(fig)


def _draw_six_panel(panels, filename: str, *, local_shift: float,
                    local_root: Path = LOCAL,
                    local_label: str = r"Local NuMI-only $q(L|E,\nu)$ weighted") -> None:
    fig, axes = plt.subplots(2, 3, figsize=(18.2, 9.8), constrained_layout=True)
    for row, (panel, x_name, xlabel, local_file, official_file, ylim) in enumerate(panels):
        x, mass, official, local = _load_panel(
            panel, x_name, local_file, official_file, ylim, local_root
        )
        columns = (
            ("Official NuMI-only", official),
            (f"{local_label}: original", local),
            (f"{local_label}: common shift {local_shift:+.3f}", local + local_shift),
        )
        for column, (panel_title, values) in enumerate(columns):
            axis = axes[row, column]
            colour = axis.pcolormesh(
                x, mass, values, shading="nearest", cmap="viridis", vmin=0, vmax=25
            )
            axis.contour(x, mass, values, levels=[LEVEL], colors="red", linewidths=2.1)
            axis.set(
                xscale="log", yscale="log", xlabel=xlabel,
                ylabel=r"$\Delta m^2_{41}\,[\mathrm{eV}^2]$",
                title=f"{panel}: {panel_title}",
            )
            axis.set_xlim(x[0], x[-1])
            axis.set_ylim(*ylim)
            fig.colorbar(colour, ax=axis, label=r"NuMI-only profiled $\Delta\chi^2$")
    fig.suptitle(
        r"NuMI-only profiled $\Delta\chi^2$: official, local, and common-zero diagnostic"
    )
    fig.savefig(OUT / filename, dpi=190)
    plt.close(fig)


def main() -> None:
    panels = (
        ("fig3a", "sin2_2theta_mue", r"$\sin^2(2\theta_{\mu e})$",
         "fig3a_energy_baseline.csv", "fig3a_official_profile.csv", (1e-2, 1e2)),
        ("fig3b", "sin2_2theta_ee", r"$\sin^2(2\theta_{ee})$",
         "fig3b_energy_baseline.csv", "fig3b_official_profile.csv", (1e-1, 40.0)),
    )
    for arguments in panels:
        _draw(*arguments)
    _draw_four_panel(
        panels, "four_numi_only_chi2_heatmaps_original.png",
        title=r"NuMI-only profiled $\Delta\chi^2$: original saved surfaces",
    )
    contour_values = []
    per_panel = {}
    for panel, x_name, _, local_file, official_file, ylim in panels:
        x, _, official, local = _load_panel(panel, x_name, local_file, official_file, ylim)
        values = _local_values_on_official_contour(x, official, local)
        contour_values.append(values)
        per_panel[panel] = {
            "crossings": int(values.size),
            "median_local_delta_chi2_on_official_contour": float(np.median(values)),
        }
    all_contour_values = np.concatenate(contour_values)
    local_shift = float(LEVEL - np.median(all_contour_values))
    _draw_four_panel(
        panels, "four_numi_only_chi2_heatmaps_common_zero_shift.png",
        local_shift=local_shift,
        title=(r"NuMI-only diagnostic common-zero shift: "
               rf"$\Delta\chi^2_{{local}}{local_shift:+.3f}$"),
    )
    _draw_six_panel(
        panels, "six_numi_only_chi2_heatmaps.png", local_shift=local_shift
    )
    (OUT / "common_zero_shift.json").write_text(json.dumps({
        "status": "diagnostic_only_not_a_valid_replacement_for_the_likelihood",
        "criterion": "one common additive shift for both profile planes",
        "construction": "median local delta-chi2 interpolated at every official 5.99 contour crossing in both panels",
        "official_contour_level": LEVEL,
        "combined_crossings": int(all_contour_values.size),
        "median_local_delta_chi2_on_official_contour": float(np.median(all_contour_values)),
        "common_additive_shift_applied_to_local": local_shift,
        "per_panel": per_panel,
        "warning": "The shift aligns the median contour level only; it does not correct shape, kernel, channel, covariance, or coverage differences.",
    }, indent=2) + "\n", encoding="utf-8")

    fixed_panels = (
        ("fig3a", "sin2_2theta_mue", r"$\sin^2(2\theta_{\mu e})$",
         "fig3a_fixed_baseline.csv", "fig3a_official_profile.csv", (1e-2, 1e2)),
        ("fig3b", "sin2_2theta_ee", r"$\sin^2(2\theta_{ee})$",
         "fig3b_fixed_baseline.csv", "fig3b_official_profile.csv", (1e-1, 40.0)),
    )
    fixed_contour_values = []
    fixed_per_panel = {}
    for panel, x_name, _, local_file, official_file, ylim in fixed_panels:
        x, _, official, local = _load_panel(
            panel, x_name, local_file, official_file, ylim, FIXED
        )
        values = _local_values_on_official_contour(x, official, local)
        fixed_contour_values.append(values)
        fixed_per_panel[panel] = {
            "crossings": int(values.size),
            "median_fixed_baseline_delta_chi2_on_official_contour": float(np.median(values)),
        }
    all_fixed_contour_values = np.concatenate(fixed_contour_values)
    fixed_shift = float(LEVEL - np.median(all_fixed_contour_values))
    _draw_six_panel(
        fixed_panels, "six_numi_only_fixed_baseline_chi2_heatmaps.png",
        local_shift=fixed_shift, local_root=FIXED,
        local_label="Local NuMI-only fixed L=0.680 km",
    )
    (OUT / "fixed_baseline_common_zero_shift.json").write_text(json.dumps({
        "status": "diagnostic_only_not_a_valid_replacement_for_the_likelihood",
        "baseline_km": 0.680,
        "criterion": "one common additive shift for both fixed-baseline profile planes",
        "construction": "median fixed-baseline local delta-chi2 interpolated at every official 5.99 contour crossing in both panels",
        "official_contour_level": LEVEL,
        "combined_crossings": int(all_fixed_contour_values.size),
        "median_fixed_baseline_delta_chi2_on_official_contour": float(np.median(all_fixed_contour_values)),
        "common_additive_shift_applied_to_fixed_baseline": fixed_shift,
        "per_panel": fixed_per_panel,
        "source_directory": str(FIXED),
        "warning": "The shift aligns the median contour level only; it does not correct shape, kernel, channel, covariance, or coverage differences.",
    }, indent=2) + "\n", encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
