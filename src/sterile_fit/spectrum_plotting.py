"""One visual contract for binned BNB, NuMI, and joint spectrum plots."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np


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
) -> None:
    """Render every beam through the original BNB panel style."""
    if not panels:
        raise ValueError("at least one spectrum panel is required")
    columns = 2
    rows = (len(panels) + columns - 1) // columns
    figure, axes = plt.subplots(rows, columns, figsize=(13, 4.5 * rows), squeeze=False)
    for axis, panel in zip(axes.flat, panels, strict=False):
        edges = np.asarray(panel.energy_edges_GeV, dtype=float)
        observed = np.asarray(panel.observed_counts, dtype=float)
        centres = (edges[:-1] + edges[1:]) / 2.0
        axis.stairs(
            panel.background_counts,
            edges,
            label="Published background",
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
                label="Published prediction systematic (diagonal)",
            )
        axis.stairs(
            panel.signal_plus_background_counts,
            edges,
            label="HEPData unconstrained Signal + Background",
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
            label="Published data",
        )
        axis.set_title(panel.title)
        axis.set_xlabel("Reconstructed neutrino energy [GeV]")
        axis.set_ylabel("Counts per bin")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    for axis in axes.flat[len(panels):]:
        axis.set_visible(False)
    figure.suptitle(title)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
