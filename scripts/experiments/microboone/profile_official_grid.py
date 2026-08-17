"""Profile the released 3+1 chi-square grid in the paper's Fig. 3 coordinates."""

from __future__ import annotations

import argparse
import json
import mmap
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_GRID = (
    ROOT
    / "data"
    / "experiments"
    / "microboone"
    / "shared"
    / "microboone_material_gridscan_numi_dm2_t14_t24_dchi2.txt"
)
DEFAULT_OUTPUT = ROOT / "outputs" / "paper_reproduction" / "official_grid_wilks"
WILKS_95_LEVEL = 5.99


def _block_bounds(document: mmap.mmap, mass_index: int, cursor: int) -> tuple[int, int]:
    marker = f"\n{mass_index}  1    1   ".encode("ascii")
    start = document.find(marker, cursor)
    if start < 0:
        raise ValueError(f"official grid is missing mass index {mass_index}")
    start += 1
    if mass_index == 100:
        return start, len(document)
    next_marker = f"\n{mass_index + 1}  1    1   ".encode("ascii")
    stop = document.find(next_marker, start)
    if stop < 0:
        raise ValueError(f"official grid is missing mass index {mass_index + 1}")
    return start, stop + 1


def _read_mass_block(document: mmap.mmap, mass_index: int, cursor: int) -> tuple[np.ndarray, int]:
    start, stop = _block_bounds(document, mass_index, cursor)
    values = np.fromstring(document[start:stop].decode("ascii"), sep=" ")
    expected_rows = 4000 * 100
    if values.size != expected_rows * 7:
        raise ValueError(
            f"mass index {mass_index} has {values.size // 7} rows; expected {expected_rows}"
        )
    table = values.reshape(expected_rows, 7)
    if not np.all(table[:, 0] == mass_index):
        raise ValueError(f"mass index {mass_index} block contains inconsistent indices")
    return table, stop - 1


def _interpolate_rows_on_log_s24(
    chi2: np.ndarray,
    s24_axis: np.ndarray,
    requested_s24: np.ndarray,
) -> np.ndarray:
    clipped = np.clip(requested_s24, s24_axis[0], s24_axis[-1])
    upper = np.searchsorted(s24_axis, clipped, side="right")
    upper = np.clip(upper, 1, len(s24_axis) - 1)
    lower = upper - 1
    log_axis = np.log(s24_axis)
    weight = (np.log(clipped) - log_axis[lower]) / (log_axis[upper] - log_axis[lower])
    rows = np.arange(chi2.shape[0])[:, None]
    return chi2[rows, lower] * (1.0 - weight) + chi2[rows, upper] * weight


def _profile_appearance(
    chi2: np.ndarray,
    s14_axis: np.ndarray,
    s24_axis: np.ndarray,
    appearance_axis: np.ndarray,
) -> np.ndarray:
    electron_amplitude = 4.0 * s14_axis * (1.0 - s14_axis)
    requested = appearance_axis[None, :] / electron_amplitude[:, None]
    physical = requested <= 1.0
    interpolated = _interpolate_rows_on_log_s24(chi2, s24_axis, requested)
    interpolated[~physical] = np.inf
    result = np.min(interpolated, axis=0)
    result[~np.isfinite(result)] = np.nan
    return result


def _interpolate_s14_branch(
    chi2: np.ndarray,
    s14_axis: np.ndarray,
    requested_s14: np.ndarray,
) -> np.ndarray:
    clipped = np.clip(requested_s14, s14_axis[0], s14_axis[-1])
    upper = np.searchsorted(s14_axis, clipped, side="right")
    upper = np.clip(upper, 1, len(s14_axis) - 1)
    lower = upper - 1
    weight = (clipped - s14_axis[lower]) / (s14_axis[upper] - s14_axis[lower])
    return chi2[lower, :] * (1.0 - weight[:, None]) + chi2[upper, :] * weight[:, None]


def _profile_electron_disappearance(
    chi2: np.ndarray,
    s14_axis: np.ndarray,
    electron_axis: np.ndarray,
) -> np.ndarray:
    root = np.sqrt(1.0 - electron_axis)
    small_branch = (1.0 - root) / 2.0
    large_branch = (1.0 + root) / 2.0
    small_values = _interpolate_s14_branch(chi2, s14_axis, small_branch)
    large_values = _interpolate_s14_branch(chi2, s14_axis, large_branch)
    return np.minimum(np.min(small_values, axis=1), np.min(large_values, axis=1))


def _cell_edges(values: np.ndarray) -> np.ndarray:
    coordinates = np.asarray(values, dtype=float)
    edges = np.empty(coordinates.size + 1)
    edges[1:-1] = np.sqrt(coordinates[:-1] * coordinates[1:])
    edges[0] = coordinates[0] ** 2 / edges[1]
    edges[-1] = coordinates[-1] ** 2 / edges[-2]
    return edges


def _plot(
    x_values: np.ndarray,
    y_values: np.ndarray,
    surface: np.ndarray,
    output_path: Path,
    *,
    x_label: str,
    x_limits: tuple[float, float],
    y_limits: tuple[float, float],
    panel: str,
) -> None:
    figure, axis = plt.subplots(figsize=(7.5, 5.8))
    colour = axis.pcolormesh(
        _cell_edges(x_values),
        _cell_edges(y_values),
        surface,
        shading="flat",
        cmap="viridis",
        vmin=0.0,
        vmax=25.0,
    )
    x_grid, y_grid = np.meshgrid(x_values, y_values)
    axis.contour(
        x_grid,
        y_grid,
        surface,
        levels=[WILKS_95_LEVEL],
        colors="tab:red",
        linewidths=2.0,
    )
    axis.legend(
        handles=[
            Line2D(
                [0],
                [0],
                color="tab:red",
                linewidth=2.0,
                label=r"95% CL ($\Delta\chi^2=5.99$)",
            )
        ],
        loc="best",
        fontsize=8,
    )
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlim(*x_limits)
    axis.set_ylim(*y_limits)
    axis.set_xlabel(x_label)
    axis.set_ylabel(r"$\Delta m^2_{41}\;[\mathrm{eV}^2]$")
    axis.set_title(rf"Official MicroBooNE grid, Fig. 3{panel}: profiled $\Delta\chi^2$")
    figure.colorbar(colour, ax=axis, label=r"Official $\Delta\chi^2$")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", type=Path, default=DEFAULT_GRID)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--x-points", type=int, default=61)
    arguments = parser.parse_args()
    if arguments.x_points < 8:
        raise ValueError("--x-points must be at least 8")
    appearance_axis = np.geomspace(1e-4, 1.0, arguments.x_points)
    electron_axis = np.geomspace(1e-2, 1.0, arguments.x_points)
    mass_values: list[float] = []
    appearance_rows: list[np.ndarray] = []
    electron_rows: list[np.ndarray] = []
    s14_axis = None
    s24_axis = None
    with arguments.grid.open("rb") as source:
        document = mmap.mmap(source.fileno(), 0, access=mmap.ACCESS_READ)
        cursor = 0
        for mass_index in range(1, 101):
            table, cursor = _read_mass_block(document, mass_index, cursor)
            mass_values.append(float(table[0, 3]))
            current_s14 = table[::100, 4]
            current_s24 = table[:100, 5]
            if s14_axis is None:
                s14_axis = current_s14.copy()
                s24_axis = current_s24.copy()
            elif not np.array_equal(current_s14, s14_axis) or not np.array_equal(current_s24, s24_axis):
                raise ValueError("mixing axes vary between official mass blocks")
            chi2 = table[:, 6].reshape(4000, 100)
            appearance_rows.append(
                _profile_appearance(chi2, current_s14, current_s24, appearance_axis)
            )
            electron_rows.append(
                _profile_electron_disappearance(chi2, current_s14, electron_axis)
            )
        document.close()
    mass_axis = np.asarray(mass_values)
    appearance_surface = np.asarray(appearance_rows)
    electron_surface = np.asarray(electron_rows)
    arguments.output_directory.mkdir(parents=True, exist_ok=True)

    appearance_table = pd.DataFrame(
        {
            "delta_m2_41_eV2": np.repeat(mass_axis, appearance_axis.size),
            "sin2_2theta_mue": np.tile(appearance_axis, mass_axis.size),
            "official_profile_delta_chi2": appearance_surface.ravel(),
        }
    )
    electron_table = pd.DataFrame(
        {
            "delta_m2_41_eV2": np.repeat(mass_axis, electron_axis.size),
            "sin2_2theta_ee": np.tile(electron_axis, mass_axis.size),
            "official_profile_delta_chi2": electron_surface.ravel(),
        }
    )
    appearance_table.to_csv(arguments.output_directory / "fig3a_official_profile.csv", index=False, float_format="%.17g")
    electron_table.to_csv(arguments.output_directory / "fig3b_official_profile.csv", index=False, float_format="%.17g")

    _plot(
        appearance_axis,
        mass_axis,
        appearance_surface,
        arguments.output_directory / "fig3a_official_wilks.png",
        x_label=r"$\sin^2(2\theta_{\mu e})$",
        x_limits=(1e-4, 1.0),
        y_limits=(1e-2, 1e2),
        panel="a",
    )
    electron_mask = (mass_axis >= 0.1) & (mass_axis <= 14.0)
    _plot(
        electron_axis,
        mass_axis[electron_mask],
        electron_surface[electron_mask],
        arguments.output_directory / "fig3b_official_wilks.png",
        x_label=r"$\sin^2(2\theta_{ee})$",
        x_limits=(1e-2, 1.0),
        y_limits=(1e-1, 14.0),
        panel="b",
    )
    metadata = {
        "source": str(arguments.grid),
        "source_size_bytes": arguments.grid.stat().st_size,
        "source_definition": "delta_chi2 = chi2_gridpoint - chi2_globalmin",
        "source_grid_shape": [100, 4000, 100],
        "criterion": "delta_chi2 = 5.99, identical to the repository Wilks diagnostic",
        "not_the_paper_limit": "the published paper contour uses CLs pseudo-experiments",
        "fig3a_profile": "linear interpolation on log(sin2_theta24) followed by minimization over the complete sin2_theta14 axis",
        "fig3b_profile": "linear interpolation on both exact sin2_theta14 branches followed by minimization over sin2_theta24",
        "endpoint_policy": "physical values inside [0,1] are clipped only to the released cell-centre range at the outermost grid cells",
    }
    (arguments.output_directory / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(arguments.output_directory)


if __name__ == "__main__":
    main()
