"""output.py: regrouped existing implementations; see docs/ARCHITECTURE.md."""
from __future__ import annotations

from matplotlib.lines import Line2D
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
import matplotlib.pyplot as plt
import numpy as np
import argparse
from hashlib import sha256
import json
import pandas as pd
import yaml
import sys
from sterile_fit.experiments.microboone.public_data import read_full_systematic_covariance, read_full_unconstrained_spectrum
import csv
import matplotlib
import shutil
from datetime import datetime, UTC
import re
from sterile_fit.paths import REPOSITORY_ROOT

_batch_name = None
_batch_directories = set()


def begin_output_batch(name=None):
    """One invocation uses one batch; explicit names group separate commands."""
    global _batch_name
    name = name or datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    _path_label(name)
    _batch_name = name
    _batch_directories.clear()


def _path_label(value):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value) or value.endswith("."):
        raise ValueError("Output labels must be safe single directory names (letters/numbers/_.-)")
    if value.split(".")[0].upper() in {"CON", "PRN", "AUX", "NUL", *[f"{p}{n}" for p in ("COM", "LPT") for n in range(1, 10)]}:
        raise ValueError("Reserved Windows directory name")
    return value


def result_directory(source, model, product):
    """Shared output boundary, also used by studies. Never touches scientific inputs."""
    global _batch_name
    if _batch_name is None:
        begin_output_batch()
    source = {"microboone_bnb_four_channel_only": "microboone_bnb",
              "microboone_bnb_numi_joint_diagnostic": "microboone_bnb_numi_joint"}.get(source, source)
    product = product.replace("appearance-profile", "fig3a").replace("electron-disappearance-profile", "fig3b")
    batch = REPOSITORY_ROOT / "outputs" / _path_label(source) / _path_label(model) / _batch_name
    target = batch / _path_label(product)
    _batch_directories.add(target)
    return target


def finish_output_batch(arguments):
    """Record code/config fingerprints, not a claim of numerical reproducibility."""
    existing = [p for p in _batch_directories if p.is_dir()]
    if not existing:
        return
    files = [REPOSITORY_ROOT / "run.py", *sorted((REPOSITORY_ROOT / "src").rglob("*.py")),
             *sorted((REPOSITORY_ROOT / "configs").rglob("*.yaml"))]
    hashes = {p.relative_to(REPOSITORY_ROOT).as_posix(): sha256(p.read_bytes()).hexdigest() for p in files}
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    for batch in {p.parent for p in existing}:
        write_json(batch / "provenance" / (stamp + ".json"), {
            "batch": batch.name, "recorded_at_utc": stamp, "arguments": list(arguments),
            "code_and_config_sha256": hashes,
            "products": sorted(p.name for p in existing if p.parent == batch),
            "note": "Invocation record; completion and scientific settings remain in product metadata.",
        })


def write_csv(table, path, *, index=False, float_format="%.17g", **kwargs):
    """One visible numerical-result writer; preserve caller column order."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=index, float_format=float_format, **kwargs)


def write_json(path, document):
    """One metadata writer; no numerical recalculation or hidden serialization."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def plot_statistic_calibration(panels, output_path, *, title):
    """Density, right tail and CDF residual; no model/profile calculation here."""
    from scipy.stats import norm
    figure, axes = plt.subplots(3, len(panels), figsize=(12, 12), squeeze=False)
    cls_threshold = None
    if len(panels) == 2 and all(panel.get("candidate") is not None for panel in panels):
        lower = max(float(panel["candidate"]["T"].min()) for panel in panels)
        upper = min(float(panel["candidate"]["T"].max()) for panel in panels)
        threshold_grid = np.linspace(lower, upper, 4000)
        tails = [
            np.interp(threshold_grid, panel["candidate"]["T"], panel["candidate"]["sf"])
            for panel in panels
        ]
        difference = tails[1] - 0.05 * tails[0]
        crossings = np.flatnonzero(difference[:-1] * difference[1:] <= 0.0)
        if crossings.size:
            observed = float(panels[0]["observed_T"])
            candidates = []
            for index in crossings:
                fraction = difference[index] / (difference[index] - difference[index + 1])
                candidates.append(threshold_grid[index] + fraction * (threshold_grid[index + 1] - threshold_grid[index]))
            cls_threshold = min(candidates, key=lambda value: abs(value - observed))
    for column, panel in enumerate(panels):
        values = np.asarray(panel["profiled_T"])
        fixed = None if panel.get("fixed_T") is None else np.asarray(panel["fixed_T"])
        toy_label = panel.get("toy_label", "Toy: reprofiled")
        distribution = norm(loc=panel["mean"], scale=panel["sigma"])
        low = min(values.min(), distribution.ppf(0.001), panel["observed_T"])
        high = max(values.max(), distribution.ppf(0.999), panel["observed_T"])
        if fixed is not None:
            low, high = min(low, fixed.min()), max(high, fixed.max())
        grid = np.linspace(low, high, 800)
        bins = np.linspace(low, high, 36)
        ax = axes[0, column]
        ax.hist(values, bins=bins, density=True, histtype="step", color="tab:green", zorder=3, label=toy_label)
        if fixed is not None:
            ax.hist(fixed, bins=bins, density=True, histtype="step", color="0.45", linestyle="--", zorder=2, label="Same Toy: fixed hypotheses")
        ax.plot(grid, distribution.pdf(grid), color="tab:red", label="Covariance Gaussian (not fitted)")
        candidate = panel.get("candidate")
        if candidate is not None:
            ax.lines[-1].set(alpha=.35, zorder=1, label="Gaussian: visual reference only")
            ax.plot(candidate["T"], candidate["pdf"], color="tab:blue", label="Fixed quadratic CF inversion", zorder=4)
        ax.axvline(panel["observed_T"], color="black", linestyle=":", label=r"$T_{obs}$")
        ax.set(title=panel["label"], xlabel=r"$T=\chi^2_{4\nu}-\chi^2_{3\nu}$", ylabel="Probability density")
        ax.legend(fontsize=8)
        ax = axes[1, column]
        ax.plot(grid, distribution.sf(grid), color="tab:red", label="Gaussian right tail")
        if candidate is not None:
            ax.lines[-1].set(alpha=.35, zorder=1, label="Gaussian: visual reference only")
            ax.plot(candidate["T"], candidate["sf"], color="tab:blue", label="Fixed quadratic right tail", zorder=4)
        tail_samples = [(values, "tab:green", panel.get("tail_label", "Reprofiled empirical tail"))]
        if fixed is not None:
            tail_samples.append((fixed, "0.45", "Fixed empirical tail"))
        for sample, color, label in tail_samples:
            ordered = np.sort(sample)
            tail = (sample.size - np.searchsorted(ordered, grid, side="left")) / sample.size
            ax.step(grid, tail, where="post", color=color, label=label, linestyle="--" if color == "0.45" else "-", zorder=2 if color == "0.45" else 3)
        ax.axvline(panel["observed_T"], color="black", linestyle=":")
        ax.axhline(0.05, color="0.7", linestyle="--")
        ax.set(xlabel=r"Threshold $t$", ylabel=r"$P(T\geq t\mid H)$", ylim=(0, 1))
        ax.legend(fontsize=8)
        ax = axes[2, column]
        epsilon = np.sqrt(np.log(2 / .05) / (2 * values.size))
        ax.axhspan(-epsilon, epsilon, color="0.8", alpha=.4,
                   label="95% DKW band (one specified CDF)")
        empirical_cdf = np.searchsorted(np.sort(values), grid, side="right") / len(values)
        ax.axhline(0, color="tab:green", label="Fixed Toy CDF baseline")
        ax.set(xlabel=r"Threshold $t$", ylabel=r"$F_{model}(t)-F_{Toy}(t)$")
        if candidate is not None:
            candidate_cdf = 1-np.interp(grid, candidate["T"], candidate["sf"])
            ax.plot(grid, candidate_cdf-empirical_cdf, color="tab:blue",
                    label="Quadratic CDF - Fixed Toy CDF", zorder=4)
            ax.plot(grid, distribution.cdf(grid)-empirical_cdf, color="tab:orange",
                    linestyle="--", label="Gaussian CDF - Fixed Toy CDF", zorder=3)
        if cls_threshold is not None:
            ax.axvline(cls_threshold, color="black", linestyle=":", label=r"Quadratic $CL_s(t)=0.05$")
        ax.legend(fontsize=8)
        for row in (0, 1, 2):
            axes[row, column].ticklabel_format(axis="x", style="sci", scilimits=(-3, 3), useOffset=False)
            axes[row, column].locator_params(axis="x", nbins=6)
    figure.suptitle(title)
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
def plot_calibration_boundaries(points, boundaries, output_path, *, amplitude_label=r"$\sin^2(2\theta_{\mu e})$"):
    """Study-only amplitude slices; supplied CLs, no recalculation or fitting."""
    figure, axes = plt.subplots(1, len(boundaries), figsize=(6*len(boundaries), 5), squeeze=False)
    for ax, (_, boundary) in zip(axes[0], boundaries.iterrows()):
        group = points[(points.mass == boundary.mass)&(points.region == "near")].sort_values("amplitude")
        for method, color in (("gaussian", "tab:red"), ("quadratic", "tab:blue"), ("toy", "tab:green")):
            ax.semilogx(group.amplitude, group[f"cls_{method}"], "o-", color=color, label=method)
        ax.axhline(.05, color="black", linestyle="--", label=r"$CL_s=0.05$")
        if np.isfinite(boundary.boundary_gaussian):
            ax.scatter([boundary.boundary_gaussian],[.05],marker="x",color="tab:red",zorder=5,label="Gaussian profiled root")
        if np.isfinite(boundary.toy_bootstrap95_low) and np.isfinite(boundary.toy_bootstrap95_high):
            ax.axvspan(boundary.toy_bootstrap95_low, boundary.toy_bootstrap95_high, color="tab:green", alpha=.15,
                       label=f"Toy bootstrap interval (in-range {boundary.bootstrap_in_range_fraction:.0%})")
        ax.set(title=rf"$\Delta m^2_{{41}}={boundary.mass:.5g}$ eV$^2$", xlabel=amplitude_label, ylabel=r"$CL_s$", ylim=(0, max(.2,group.cls_toy.max()*1.15,group.cls_gaussian.max()*1.1)))
        ax.set_xticks(group.amplitude)
        ax.set_xticklabels([f"{x:.3g}" for x in group.amplitude])
        ax.minorticks_off()
        ax.legend(fontsize=8)
    figure.suptitle("Local boundary slices; lines connect sampled points, not a full contour")
    figure.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def plot_gaussian_toy_grid(samples, summaries, output_path):
    """One column per point; density and right-tail rows for each hypothesis."""
    from scipy.stats import norm
    points = sorted(samples.point_index.unique())
    figure, axes = plt.subplots(4, len(points), figsize=(5 * len(points), 14), squeeze=False)
    for col, point in enumerate(points):
        for i, generator in enumerate(("3nu", "4nu")):
            group = samples[(samples.point_index == point) & (samples.generator == generator)]
            score = summaries[(summaries.point_index == point) & (summaries.generator == generator) & (summaries.variant == "reprofiled")].iloc[0]
            values = group.test_statistic.to_numpy()
            gaussian = norm(score.gaussian_mean, score.gaussian_sigma)
            low = min(values.min(), gaussian.ppf(.0005), score.observed_T)
            high = max(values.max(), gaussian.ppf(.9995), score.observed_T)
            grid = np.linspace(low, high, 1000)
            density, tail = axes[2*i, col], axes[2*i+1, col]
            density.hist(values, bins=np.linspace(low, high, 65), density=True,
                         histtype="step", color="tab:green", label=f"Profiled Toy (N={len(values):,})")
            density.plot(grid, gaussian.pdf(grid), color="tab:red", label="Covariance Gaussian")
            density.set_ylabel(rf"${i+3}\nu$: density")
            tail.plot(grid, gaussian.sf(grid), color="tab:red", label="Gaussian right tail")
            empirical = (len(values)-np.searchsorted(np.sort(values), grid, side="left"))/len(values)
            tail.step(grid, empirical, where="post", color="tab:green", label="Toy right tail")
            tail.set_ylabel(rf"$P(T\geq t\mid {i+3}\nu)$")
            tail.set_ylim(0, 1)
            for ax in (density, tail):
                ax.axvline(score.observed_T, color="0.3", linestyle=":", label=r"$T_{obs}$")
                ax.set_xlabel(r"$T=\chi^2_{4\nu}-\chi^2_{3\nu}$" if ax is density else r"Threshold $t$")
                ax.ticklabel_format(axis="x", style="sci", scilimits=(-3, 3), useOffset=False)
                ax.locator_params(axis="x", nbins=5)
                ax.legend(fontsize=8)
            if i == 0:
                density.set_title(rf"Point {point}: $\Delta m^2_{{41}}={score.delta_m2_41_eV2:.5g}$ eV$^2$" + "\n" + rf"$\sin^2(2\theta_{{\mu e}})={score.sin2_2theta_mue:.5g}$")
    figure.suptitle("BNB + NuMI: covariance Gaussian versus profiled Toy MC", fontsize=16)
    figure.tight_layout(rect=(0, 0, 1, .96))
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _log_cell_edges(values: np.ndarray) -> np.ndarray:
    """Cell edges whose midpoints are geometric on a logarithmic plot."""
    coordinates = np.asarray(values, dtype=float)
    if coordinates.ndim != 1 or coordinates.size < 2:
        raise ValueError("a plotted logarithmic scan axis needs at least two coordinates")
    if np.any(coordinates <= 0.0) or np.any(np.diff(coordinates) <= 0.0):
        raise ValueError("plotted logarithmic scan coordinates must be positive and increasing")
    edges = np.empty(coordinates.size + 1, dtype=float)
    edges[1:-1] = np.sqrt(coordinates[:-1] * coordinates[1:])
    edges[0] = coordinates[0] ** 2 / edges[1]
    edges[-1] = coordinates[-1] ** 2 / edges[-2]
    return edges



# One visual contract for binned BNB, NuMI, and joint spectrum plots.


@dataclass(frozen=True)
class SpectrumCurve:
    label: str
    counts: np.ndarray
    color: str
    linestyle: str = "-"
    linewidth: float = 1.5


@dataclass(frozen=True)
class SpectrumPanel:
    title: str
    energy_edges_GeV: np.ndarray
    observed_counts: np.ndarray
    observed_error_down: np.ndarray
    observed_error_up: np.ndarray
    background_counts: np.ndarray
    signal_plus_background_counts: np.ndarray
    comparison_curves: Sequence[SpectrumCurve] = ()
    prediction_systematic_sigma: np.ndarray | None = None


def render_microboone_spectrum_panels(
    panels: Sequence[SpectrumPanel],
    output_path: Path,
    *,
    title: str,
    layout: str = "channels",
) -> None:
    """Render every beam through the original BNB panel style."""
    if not panels:
        raise ValueError("at least one spectrum panel is required")
    if layout not in {"channels", "figure1"}:
        raise ValueError("unknown spectrum layout")
    figure1 = layout == "figure1"
    columns = 1 if figure1 else 2
    rows = (len(panels) + columns - 1) // columns
    figure, axes = plt.subplots(
        rows, columns, figsize=(9.0, 10.0) if figure1 else (13, 4.5 * rows),
        squeeze=False, sharex=figure1,
    )
    for axis, panel in zip(axes.flat, panels, strict=False):
        edges = np.asarray(panel.energy_edges_GeV, dtype=float)
        observed = np.asarray(panel.observed_counts, dtype=float)
        centres = (edges[:-1] + edges[1:]) / 2.0
        axis.stairs(
            panel.background_counts,
            edges,
            label="Background" if figure1 else "Published background",
            color="tab:blue",
            linestyle="--",
        )
        if panel.prediction_systematic_sigma is not None:
            sigma = np.asarray(panel.prediction_systematic_sigma, dtype=float)
            lower = np.maximum(np.asarray(panel.signal_plus_background_counts) - sigma, 0.0)
            upper = np.asarray(panel.signal_plus_background_counts) + sigma
            axis.fill_between(
                edges,
                np.append(lower, lower[-1]),
                np.append(upper, upper[-1]),
                step="post",
                color="0.55",
                alpha=0.28,
                label="Prediction systematic (diagonal)" if figure1 else "Published prediction systematic (diagonal)",
            )
        axis.stairs(
            panel.signal_plus_background_counts,
            edges,
            label="Signal + Background" if figure1 else "HEPData unconstrained Signal + Background",
            color="black",
        )
        for curve in panel.comparison_curves:
            axis.stairs(
                curve.counts,
                edges,
                label=curve.label,
                color=curve.color,
                linestyle=curve.linestyle,
                linewidth=curve.linewidth,
            )
        axis.errorbar(
            centres,
            observed,
            yerr=[panel.observed_error_down, panel.observed_error_up],
            fmt="o",
            color="tab:red",
            label="Data" if figure1 else "Published data",
        )
        axis.set_title(panel.title)
        if not figure1:
            axis.set_xlabel("Reconstructed neutrino energy [GeV]")
        axis.set_ylabel("Counts per bin")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    for axis in axes.flat[len(panels):]:
        axis.set_visible(False)
    if figure1:
        axes[-1, 0].set_xlabel(r"Reconstructed neutrino energy, $E_\nu^{\mathrm{reco}}$ [GeV]")
    figure.suptitle(title)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_three_plus_one_scan(result_table, output_directory, arguments, analysis, cls_column):
    x_name = {
        "appearance-profile": "fixed_sin2_2theta_mue",
        "electron-disappearance-profile": "fixed_sin2_2theta_ee",
        "s14-profile": "fixed_sin2_theta14",
    }[arguments.mode]
    pivot = result_table.pivot(index="fixed_delta_m2_41_eV2", columns=x_name, values="chi2")
    x_values = pivot.columns.to_numpy(dtype=float)
    y_values = pivot.index.to_numpy(dtype=float)
    x_grid, y_grid = np.meshgrid(x_values, y_values)
    x_edges = _log_cell_edges(x_values)
    y_edges = _log_cell_edges(y_values)
    figure, axis = plt.subplots(figsize=(7.5, 5.8))
    cls_surface = result_table.pivot(
        index="fixed_delta_m2_41_eV2", columns=x_name, values=cls_column
    ).to_numpy(dtype=float)
    colour = axis.pcolormesh(
        x_edges,
        y_edges,
        cls_surface,
        shading="flat",
        cmap="viridis_r",
        vmin=0.0,
        vmax=1.0,
    )
    has_cls_95 = float(cls_surface.min()) <= 0.05 <= float(cls_surface.max())
    if has_cls_95 and len(x_values) >= 8 and len(y_values) >= 8:
        axis.contour(
            x_grid,
            y_grid,
            cls_surface,
            levels=[0.05],
            colors="tab:red",
            linewidths=2.0,
        )
        axis.legend(
            handles=[Line2D([0], [0], color="tab:red", linewidth=2.0, label=(
                r"95% $CL_s$ (fixed-hypothesis Toy MC)"
                if arguments.cls_calibration == "toy"
                else (
                    r"95% $CL_s$ (adaptive quadratic form + Toy MC)"
                    if arguments.cls_calibration == "adaptive-toy"
                    else r"95% $CL_s$ (quadratic-form inversion)"
                )
            ))],
            loc="best",
            fontsize=8,
        )
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel({
        "appearance-profile": r"$\sin^2(2\theta_{\mu e})$",
        "electron-disappearance-profile": r"$\sin^2(2\theta_{ee})$",
        "s14-profile": r"$\sin^2\theta_{14}$",
    }[arguments.mode])
    axis.set_ylabel(r"$\Delta m^2_{41}\;[\mathrm{eV}^2]$")
    analysis_label = (
        "MicroBooNE BNB + NuMI"
        if analysis.analysis_name == "microboone_bnb_numi_joint_diagnostic"
        else analysis.analysis_name.replace("_", " ")
    )
    calibration_title = {
        "toy": "fixed-hypothesis Toy-MC",
        "adaptive-toy": "adaptive quadratic form + fixed-hypothesis Toy-MC",
        "analytic": "profiled quadratic-form inversion",
    }[arguments.cls_calibration]
    axis.set_title(rf"$3+1$ {analysis_label}: {calibration_title} $CL_s$")
    figure.colorbar(colour, ax=axis, label=r"$CL_s$")
    figure.tight_layout()
    figure.savefig(output_directory / "profile.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_one_plus_three_plus_one_scan(table: pd.DataFrame, output: Path) -> None:
    q41 = np.sort(table["delta_m2_41_absolute_eV2"].unique())
    q51 = np.sort(table["delta_m2_51_eV2"].unique())
    surface = (
        table.pivot(
            index="delta_m2_51_eV2",
            columns="delta_m2_41_absolute_eV2",
            values="cls",
        )
        .reindex(index=q51, columns=q41)
        .to_numpy(dtype=float)
    )
    figure, axis = plt.subplots(figsize=(7.2, 5.8), constrained_layout=True)
    image = axis.pcolormesh(
        _extended_plot_log_cell_edges(q41),
        _extended_plot_log_cell_edges(q51),
        surface,
        shading="flat",
        cmap="viridis_r",
        vmin=0.0,
        vmax=1.0,
    )
    if np.nanmin(surface) <= 0.05 <= np.nanmax(surface):
        axis.contour(q41, q51, surface, levels=[0.05], colors=["red"], linewidths=2.0)
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel(r"$|\Delta m^2_{41}|\;[\mathrm{eV}^2]$")
    axis.set_ylabel(r"$\Delta m^2_{51}\;[\mathrm{eV}^2]$")
    axis.set_title("1+3+1 mixing-profile diagnostic")
    colorbar = figure.colorbar(image, ax=axis)
    colorbar.set_label(r"$CL_s$")
    figure.savefig(output, dpi=180)
    plt.close(figure)



def _extended_plot_log_cell_edges(values: np.ndarray) -> np.ndarray:
    coordinates = np.asarray(values, dtype=float)
    edges = np.empty(coordinates.size + 1, dtype=float)
    edges[1:-1] = np.sqrt(coordinates[:-1] * coordinates[1:])
    edges[0] = coordinates[0] ** 2 / edges[1]
    edges[-1] = coordinates[-1] ** 2 / edges[-2]
    return edges



# Draw the released 14-channel spectra, covariance, and visible BNB flux.


from sterile_fit.paths import REPOSITORY_ROOT as INPUTS_PLOT_ROOT
if str(INPUTS_PLOT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(INPUTS_PLOT_ROOT / "src"))



INPUTS_PLOT_SPECTRUM_PATH = (
    INPUTS_PLOT_ROOT
    / "data"
    / "experiments"
    / "microboone"
    / "shared"
    / "raw"
    / "hepdata_microboone_2025"
    / "HEPData-ins3088922-v1-Unconstrained_14_channels.csv"
)
INPUTS_PLOT_COVARIANCE_PATH = INPUTS_PLOT_SPECTRUM_PATH.with_name(
    "HEPData-ins3088922-v1-14_channel_covariance_matrix.csv"
)
INPUTS_PLOT_BNB_FLUX_PATH = (
    INPUTS_PLOT_ROOT / "data" / "experiments" / "microboone" / "bnb" / "inputs" / "bnb_flux.csv"
)

INPUTS_PLOT_CHANNEL_NAMES = (
    r"$\nu_e$ CC FC",
    r"$\nu_e$ CC PC",
    r"$\nu_\mu$ CC FC",
    r"$\nu_\mu$ CC PC",
    r"$\nu_\mu$ CC $\pi^0$ FC",
    r"$\nu_\mu$ CC $\pi^0$ PC",
    r"NC $\pi^0$",
)
INPUTS_PLOT_BEAM_NAMES = ("BNB", "NuMI")
INPUTS_PLOT_RECO_BIN_COUNT = 26


def inputs_plot_plot_fourteen_channel_spectra(output_path: Path) -> None:
    spectra = read_full_unconstrained_spectrum(INPUTS_PLOT_SPECTRUM_PATH)
    data = spectra["observed_counts"]
    error_up = spectra["observed_statistical_error_up"]
    error_down = spectra["observed_statistical_error_down"]
    background = spectra["published_background_counts"]
    total = spectra["published_total_prediction_counts"]

    figure, axes = plt.subplots(7, 2, figsize=(13, 20), sharex=True, constrained_layout=True)
    x = np.arange(INPUTS_PLOT_RECO_BIN_COUNT)
    for beam_index, beam_name in enumerate(INPUTS_PLOT_BEAM_NAMES):
        for channel_index, channel_name in enumerate(INPUTS_PLOT_CHANNEL_NAMES):
            global_channel = beam_index * len(INPUTS_PLOT_CHANNEL_NAMES) + channel_index
            start = global_channel * INPUTS_PLOT_RECO_BIN_COUNT
            stop = start + INPUTS_PLOT_RECO_BIN_COUNT
            axis = axes[channel_index, beam_index]
            axis.step(x, total[start:stop], where="mid", color="tab:blue", label="Signal + Background")
            axis.step(x, background[start:stop], where="mid", color="tab:orange", label="Background")
            axis.errorbar(
                x,
                data[start:stop],
                yerr=np.vstack((error_down[start:stop], error_up[start:stop])),
                fmt="o",
                color="black",
                markersize=2.8,
                linewidth=0.8,
                capsize=1.5,
                label="Data",
            )
            axis.set_title(f"{beam_name}: {channel_name}")
            axis.set_ylabel("Events / bin")
            axis.grid(alpha=0.2)
            if channel_index == len(INPUTS_PLOT_CHANNEL_NAMES) - 1:
                axis.set_xlabel("Reconstructed-energy bin (25 is overflow)")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def inputs_plot_plot_full_covariance(output_path: Path) -> None:
    covariance = read_full_systematic_covariance(INPUTS_PLOT_COVARIANCE_PATH)
    scale = np.max(np.abs(covariance))
    figure, axis = plt.subplots(figsize=(10, 9), constrained_layout=True)
    image = axis.imshow(covariance, origin="lower", cmap="RdBu_r", vmin=-scale, vmax=scale)
    for boundary in range(INPUTS_PLOT_RECO_BIN_COUNT, covariance.shape[0], INPUTS_PLOT_RECO_BIN_COUNT):
        axis.axhline(boundary - 0.5, color="black", linewidth=0.25, alpha=0.5)
        axis.axvline(boundary - 0.5, color="black", linewidth=0.25, alpha=0.5)
    axis.axhline(7 * INPUTS_PLOT_RECO_BIN_COUNT - 0.5, color="black", linewidth=1.2)
    axis.axvline(7 * INPUTS_PLOT_RECO_BIN_COUNT - 0.5, color="black", linewidth=1.2)
    axis.set_xlabel("Global reconstructed-bin index")
    axis.set_ylabel("Global reconstructed-bin index")
    axis.set_title("MicroBooNE released 14-channel systematic covariance")
    figure.colorbar(image, ax=axis, label=r"Covariance [events$^2$]")
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def inputs_plot_plot_bnb_flux(output_path: Path) -> None:
    table = pd.read_csv(INPUTS_PLOT_BNB_FLUX_PATH)
    required = {"true_energy_GeV", "numu_flux", "numubar_flux", "nue_flux", "nuebar_flux"}
    if not required.issubset(table.columns) or len(table) != 60:
        raise ValueError("BNB flux must contain the declared four flavours and 60 true-energy bins")

    energy = table["true_energy_GeV"].to_numpy(dtype=float)
    curves = (
        ("numu_flux", r"$\nu_\mu$"),
        ("numubar_flux", r"$\bar{\nu}_\mu$"),
        ("nue_flux", r"$\nu_e$"),
        ("nuebar_flux", r"$\bar{\nu}_e$"),
    )
    figure, axis = plt.subplots(figsize=(9, 6), constrained_layout=True)
    for column, label in curves:
        values = table[column].to_numpy(dtype=float)
        if not np.all(np.isfinite(values)) or np.any(values < 0.0):
            raise ValueError(f"invalid BNB flux values in {column}")
        axis.step(energy, values, where="mid", linewidth=1.5, label=label)
    axis.set_yscale("log")
    axis.set_xlabel(r"True neutrino energy $E_\nu$ [GeV]")
    axis.set_ylabel("Stored flux value per 0.05 GeV bin")
    axis.set_title("Visible BNB flux inputs")
    axis.grid(which="both", alpha=0.25)
    axis.legend(frameon=False, ncol=2)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_public_inputs() -> None:
    INPUTS_PLOT_OUTPUT_DIRECTORY = result_directory("microboone_public", "inputs", "public_plots")
    INPUTS_PLOT_OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    products = {
        "14-channel spectra": INPUTS_PLOT_OUTPUT_DIRECTORY / "microboone_14_channel_spectra.png",
        "14-channel covariance": INPUTS_PLOT_OUTPUT_DIRECTORY / "microboone_14_channel_covariance.png",
        "BNB flux": INPUTS_PLOT_OUTPUT_DIRECTORY / "bnb_four_flavour_flux.png",
    }
    inputs_plot_plot_fourteen_channel_spectra(products["14-channel spectra"])
    inputs_plot_plot_full_covariance(products["14-channel covariance"])
    inputs_plot_plot_bnb_flux(products["BNB flux"])
    for name, path in products.items():
        print(f"WROTE {name}: {path}")


# Plot only the eight visible MicroBooNE NuMI flux-component CSV files.


matplotlib.use("Agg")


from sterile_fit.paths import REPOSITORY_ROOT as NUMI_PLOT_ROOT
NUMI_PLOT_INPUT_DIRECTORY = NUMI_PLOT_ROOT / "data" / "experiments" / "microboone" / "numi" / "inputs" / "flux_components"
NUMI_PLOT_FLAVOURS = ("numu", "numubar", "nue", "nuebar")
NUMI_PLOT_HORN_MODES = ("fhc", "rhc")
NUMI_PLOT_EXPECTED_COLUMNS = [
    "energy_low_GeV",
    "energy_high_GeV",
    "energy_center_GeV",
    "flux_per_POT_per_cm2_per_100MeV",
    "is_censored",
    "derivation",
]
NUMI_PLOT_DISPLAY_NAMES = {
    "numu": r"$\nu_\mu$",
    "numubar": r"$\bar\nu_\mu$",
    "nue": r"$\nu_e$",
    "nuebar": r"$\bar\nu_e$",
}


def numi_plot_load_one(path: Path) -> pd.DataFrame:
    table = pd.read_csv(path)
    if list(table.columns) != NUMI_PLOT_EXPECTED_COLUMNS or table.shape[0] != 50:
        raise ValueError(f"unexpected NuMI flux schema or bin count: {path}")
    expected_low = np.arange(0.0, 5.0, 0.1)
    expected_high = expected_low + 0.1
    if not np.allclose(table["energy_low_GeV"], expected_low, atol=1e-12):
        raise ValueError(f"energy_low_GeV is not the declared 0.1 GeV grid: {path}")
    if not np.allclose(table["energy_high_GeV"], expected_high, atol=1e-12):
        raise ValueError(f"energy_high_GeV is not the declared 0.1 GeV grid: {path}")
    flux = table["flux_per_POT_per_cm2_per_100MeV"].to_numpy(dtype=float)
    if not np.all(np.isfinite(flux)) or np.any(flux < 0.0):
        raise ValueError(f"flux must be finite and non-negative: {path}")
    if table["is_censored"].astype(bool).any():
        raise ValueError(f"censored PDF values cannot enter the NuMI input plot: {path}")
    return table


def plot_numi_flux_inputs() -> None:
    parser = argparse.ArgumentParser(description="Plot the eight visible NuMI flux inputs.")
    parser.add_argument("--output", type=Path, default=result_directory("microboone_numi", "inputs", "flux_plots") / "flux_components.png")
    arguments = parser.parse_args()
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 4, figsize=(15.5, 7.4), sharex=True)
    summary_rows: list[dict[str, object]] = []

    for row, horn_mode in enumerate(NUMI_PLOT_HORN_MODES):
        for column, flavour in enumerate(NUMI_PLOT_FLAVOURS):
            path = NUMI_PLOT_INPUT_DIRECTORY / f"numi_{horn_mode}_{flavour}_flux.csv"
            table = numi_plot_load_one(path)
            flux = table["flux_per_POT_per_cm2_per_100MeV"].to_numpy(dtype=float)
            edges = np.concatenate(
                [table["energy_low_GeV"].to_numpy(dtype=float), [float(table["energy_high_GeV"].iloc[-1])]]
            )
            values = np.concatenate([flux, [flux[-1]]])
            axis = axes[row, column]
            axis.step(edges, values, where="post", color="#0072B2", linewidth=1.8)
            axis.set_yscale("log")
            axis.set_xlim(0.0, 5.0)
            axis.grid(alpha=0.2, which="both")
            axis.set_title(f"{horn_mode.upper()}  {NUMI_PLOT_DISPLAY_NAMES[flavour]}")
            axis.set_xlabel("Neutrino energy [GeV]")
            if column == 0:
                axis.set_ylabel(r"Flux [$\nu$/POT/cm$^2$/100 MeV]")

            summary_rows.append(
                {
                    "horn_mode": horn_mode.upper(),
                    "flavour": flavour,
                    "bins": len(table),
                    "energy_min_GeV": float(table["energy_low_GeV"].iloc[0]),
                    "energy_max_GeV": float(table["energy_high_GeV"].iloc[-1]),
                    "minimum_flux": float(np.min(flux)),
                    "maximum_flux": float(np.max(flux)),
                    "sum_of_0p1_GeV_bin_flux_values": float(np.sum(flux)),
                    "sha256": sha256(path.read_bytes()).hexdigest().upper(),
                    "source_file": str(path),
                }
            )

    figure.suptitle("MicroBooNE NuMI New Flux inputs reconstructed from PDF pages 4-7")
    figure.tight_layout()
    figure.savefig(arguments.output, dpi=240, bbox_inches="tight")
    plt.close(figure)
    with arguments.output.with_suffix(".csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    write_json(arguments.output.with_suffix(".metadata.json"), {
                "status": "regenerable_spectrum_output",
                "numerical_input_directory": str(NUMI_PLOT_INPUT_DIRECTORY),
                "plotted_components": 8,
                "bins_per_component": 50,
                "plot_is_numerical_input": False,
                "scientific_boundary": "PDF-vector reconstructed flux inputs; not official unpublished arrays",
            })
    print(arguments.output)


# Collect completed scan plots and redraw their CLs=0.05 contours without heatmaps.


from sterile_fit.paths import REPOSITORY_ROOT as COMPARISON_ROOT

COMPARISON_SCANS = (
    (
        "fig3a_analytic",
        "fixed_sin2_2theta_mue",
        "cls_quadratic",
        r"$\sin^2(2\theta_{\mu e})$",
        r"Fig. 3a: analytic $CL_s$",
    ),
    (
        "fig3a_adaptive_toy",
        "fixed_sin2_2theta_mue",
        "cls_adaptive_hybrid",
        r"$\sin^2(2\theta_{\mu e})$",
        r"Fig. 3a: adaptive fixed-hypothesis Toy-MC $CL_s$",
    ),
    (
        "fig3b_analytic",
        "fixed_sin2_2theta_ee",
        "cls_quadratic",
        r"$\sin^2(2\theta_{ee})$",
        r"Fig. 3b: analytic $CL_s$",
    ),
    (
        "fig3b_adaptive_toy",
        "fixed_sin2_2theta_ee",
        "cls_adaptive_hybrid",
        r"$\sin^2(2\theta_{ee})$",
        r"Fig. 3b: adaptive fixed-hypothesis Toy-MC $CL_s$",
    ),
)


def plot_completed_scan_contours() -> None:
    parser = argparse.ArgumentParser()
    for name in ("fig3a-analytic", "fig3b-analytic", "fig3a-toy", "fig3b-toy"):
        parser.add_argument("--" + name, type=Path, required=True, help="Completed result directory")
    parser.add_argument("--source", default="microboone_bnb_numi_joint", help="Output grouping label only")
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=None,
    )
    arguments = parser.parse_args()
    if arguments.output_directory is None:
        arguments.output_directory = result_directory(arguments.source, "three_plus_one", "contour_comparison")
    arguments.output_directory.mkdir(parents=True, exist_ok=True)

    combined_surfaces = {}

    sources = {"fig3a_analytic": arguments.fig3a_analytic, "fig3b_analytic": arguments.fig3b_analytic,
               "fig3a_adaptive_toy": arguments.fig3a_toy, "fig3b_adaptive_toy": arguments.fig3b_toy}
    for short_name, x_column, cls_column, x_label, title in COMPARISON_SCANS:
        source = sources[short_name]
        table = pd.read_csv(source / "result.csv")
        required = {"fixed_delta_m2_41_eV2", x_column, cls_column}
        missing = required.difference(table.columns)
        if missing:
            raise ValueError(f"{source}: missing columns {sorted(missing)}")

        pivot = table.pivot(
            index="fixed_delta_m2_41_eV2",
            columns=x_column,
            values=cls_column,
        )
        x_values = pivot.columns.to_numpy(dtype=float)
        y_values = pivot.index.to_numpy(dtype=float)
        cls_values = pivot.to_numpy(dtype=float)
        if not cls_values.min() <= 0.05 <= cls_values.max():
            raise ValueError(f"{source}: CLs=0.05 is outside the calculated surface")
        combined_surfaces[short_name] = (x_values, y_values, cls_values)

        figure, axis = plt.subplots(figsize=(7.5, 5.8))
        axis.contour(
            x_values,
            y_values,
            cls_values,
            levels=[0.05],
            colors="tab:red",
            linewidths=2.2,
        )
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlim(x_values.min(), x_values.max())
        axis.set_ylim(y_values.min(), y_values.max())
        axis.set_xlabel(x_label)
        axis.set_ylabel(r"$\Delta m^2_{41}\;[\mathrm{eV}^2]$")
        axis.set_title(title)
        axis.grid(False)
        axis.legend(
            handles=[Line2D([0], [0], color="tab:red", linewidth=2.2, label=r"95% $CL_s$")] ,
            loc="best",
        )
        figure.tight_layout()
        figure.savefig(
            arguments.output_directory / f"line_only_{short_name}.png",
            dpi=180,
            bbox_inches="tight",
        )
        plt.close(figure)

        shutil.copy2(
            source / "profile.png",
            arguments.output_directory / f"original_{short_name}.png",
        )

    figure, axis = plt.subplots(figsize=(8.2, 6.2))
    combined_styles = (
        ("fig3a_analytic", "tab:red", "solid"),
        ("fig3a_adaptive_toy", "tab:green", "solid"),
        ("fig3b_analytic", "tab:red", "dashed"),
        ("fig3b_adaptive_toy", "tab:green", "dashed"),
    )
    for short_name, colour, line_style in combined_styles:
        x_values, y_values, cls_values = combined_surfaces[short_name]
        contours = axis.contour(
            x_values,
            y_values,
            cls_values,
            levels=[0.05],
            colors=colour,
            linewidths=2.0,
            linestyles=line_style,
        )
        # Matplotlib may create several disconnected contour paths; apply the
        # requested line style to every component explicitly.
        contours.set_linestyle(line_style)

    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlim(1e-4, 1.0)
    axis.set_ylim(1e-2, 1e2)
    axis.set_xlabel(
        r"$\sin^2(2\theta_{\mu e})$ (Fig. 3a) or $\sin^2(2\theta_{ee})$ (Fig. 3b)"
    )
    axis.set_ylabel(r"$\Delta m^2_{41}\;[\mathrm{eV}^2]$")
    axis.set_title(r"MicroBooNE 95% $CL_s$ exclusion contours")
    axis.grid(False)
    axis.legend(
        handles=[
            Line2D([0], [0], color="tab:red", linestyle="solid", linewidth=2.0,
                   label=r"Fig. 3a analytic (no Toy MC)"),
            Line2D([0], [0], color="tab:green", linestyle="solid", linewidth=2.0,
                   label=r"Fig. 3a adaptive Toy MC"),
            Line2D([0], [0], color="tab:red", linestyle="dashed", linewidth=2.0,
                   label=r"Fig. 3b analytic (no Toy MC)"),
            Line2D([0], [0], color="tab:green", linestyle="dashed", linewidth=2.0,
                   label=r"Fig. 3b adaptive Toy MC"),
        ],
        loc="best",
        fontsize=8,
    )
    figure.tight_layout()
    figure.savefig(
        arguments.output_directory / "combined_four_contours_single_axes.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)

    paired_plots = (
        (
            "fig3a",
            "fig3a_analytic",
            "fig3a_adaptive_toy",
            r"$\sin^2(2\theta_{\mu e})$",
            r"Fig. 3a: 95% $CL_s$ exclusion contours",
        ),
        (
            "fig3b",
            "fig3b_analytic",
            "fig3b_adaptive_toy",
            r"$\sin^2(2\theta_{ee})$",
            r"Fig. 3b: 95% $CL_s$ exclusion contours",
        ),
    )
    for plot_name, analytic_name, toy_name, x_label, title in paired_plots:
        figure, axis = plt.subplots(figsize=(7.5, 5.8))
        for surface_name, colour in (
            (analytic_name, "tab:red"),
            (toy_name, "tab:green"),
        ):
            x_values, y_values, cls_values = combined_surfaces[surface_name]
            axis.contour(
                x_values,
                y_values,
                cls_values,
                levels=[0.05],
                colors=colour,
                linewidths=2.1,
            )
        analytic_x, analytic_y, _ = combined_surfaces[analytic_name]
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlim(analytic_x.min(), analytic_x.max())
        axis.set_ylim(analytic_y.min(), analytic_y.max())
        axis.set_xlabel(x_label)
        axis.set_ylabel(r"$\Delta m^2_{41}\;[\mathrm{eV}^2]$")
        axis.set_title(title)
        axis.grid(False)
        axis.legend(
            handles=[
                Line2D([0], [0], color="tab:red", linewidth=2.1,
                       label=r"Analytic (no Toy MC)"),
                Line2D([0], [0], color="tab:green", linewidth=2.1,
                       label=r"Adaptive fixed-hypothesis Toy MC"),
            ],
            loc="best",
        )
        figure.tight_layout()
        figure.savefig(
            arguments.output_directory / f"overlay_{plot_name}_analytic_vs_toy.png",
            dpi=180,
            bbox_inches="tight",
        )
        plt.close(figure)

    write_json(arguments.output_directory / "comparison_sources.json", {k: str(v.resolve()) for k, v in sources.items()})
    print(arguments.output_directory)



