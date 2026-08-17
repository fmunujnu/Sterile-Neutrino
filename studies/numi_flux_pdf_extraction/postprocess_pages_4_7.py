from __future__ import annotations

from typing import Any

import numpy as np


TARGET_BIN_WIDTH_GEV = 0.1
TARGET_ENERGY_MAX_GEV = 5.0
EDGE_TOLERANCE_GEV = 2e-5


def target_edges() -> np.ndarray:
    """Return the declared 0-5 GeV grid in exact 0.1 GeV steps."""

    count = int(round(TARGET_ENERGY_MAX_GEV / TARGET_BIN_WIDTH_GEV))
    return np.linspace(0.0, TARGET_ENERGY_MAX_GEV, count + 1)


def expand_native_histogram_to_target_grid(
    native_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Expand displayed variable-width steps onto the 0.1 GeV grid.

    The plot ordinate is explicitly per 100 MeV. A wide displayed step is
    therefore copied to each enclosed 0.1 GeV bin. This is piecewise-constant
    expansion, not interpolation, and adds no unshown within-bin structure.
    """

    ordered = sorted(native_rows, key=lambda row: float(row["energy_low_GeV"]))
    if not ordered:
        raise ValueError("cannot expand an empty native histogram")
    edges = target_edges()
    expanded: list[dict[str, Any]] = []
    for low, high in zip(edges[:-1], edges[1:]):
        center = (low + high) / 2.0
        matches = [
            row
            for row in ordered
            if float(row["energy_low_GeV"]) - EDGE_TOLERANCE_GEV <= center
            < float(row["energy_high_GeV"]) + EDGE_TOLERANCE_GEV
        ]
        if len(matches) != 1:
            raise ValueError(
                f"target bin [{low:.1f}, {high:.1f}] GeV has {len(matches)} native owners"
            )
        source = matches[0]
        expanded.append(
            {
                "energy_low_GeV": float(low),
                "energy_high_GeV": float(high),
                "energy_center_GeV": float(center),
                "flux_per_POT_per_cm2_per_100MeV": float(source["flux_plot_units"]),
                "is_lower_censored": bool(source["is_lower_clipped"]),
                "is_upper_censored": bool(source["is_upper_clipped"]),
                "native_energy_low_GeV": float(source["energy_low_GeV"]),
                "native_energy_high_GeV": float(source["energy_high_GeV"]),
            }
        )
    return expanded


def derive_antineutrino(
    neutrino: list[dict[str, Any]],
    total: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Calculate antineutrino flux as the same-mode total minus neutrino."""

    if len(neutrino) != len(total):
        raise ValueError("neutrino and total arrays have different lengths")
    result: list[dict[str, Any]] = []
    for neutrino_row, total_row in zip(neutrino, total):
        for edge_name in ("energy_low_GeV", "energy_high_GeV"):
            if not np.isclose(neutrino_row[edge_name], total_row[edge_name], atol=1e-12):
                raise ValueError("neutrino and total arrays are not on the same energy grid")
        difference = (
            float(total_row["flux_per_POT_per_cm2_per_100MeV"])
            - float(neutrino_row["flux_per_POT_per_cm2_per_100MeV"])
        )
        result.append(
            {
                "energy_low_GeV": float(neutrino_row["energy_low_GeV"]),
                "energy_high_GeV": float(neutrino_row["energy_high_GeV"]),
                "energy_center_GeV": float(neutrino_row["energy_center_GeV"]),
                "flux_per_POT_per_cm2_per_100MeV": difference,
                "is_lower_censored": bool(
                    neutrino_row["is_lower_censored"] or total_row["is_lower_censored"]
                ),
                "is_upper_censored": bool(
                    neutrino_row["is_upper_censored"] or total_row["is_upper_censored"]
                ),
                "neutrino_component": float(
                    neutrino_row["flux_per_POT_per_cm2_per_100MeV"]
                ),
                "neutrino_plus_antineutrino_total": float(
                    total_row["flux_per_POT_per_cm2_per_100MeV"]
                ),
            }
        )
    return result
