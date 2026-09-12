"""Extract and validate public LSND Fig. 16/24 binned approximations.

This study deliberately does not claim to reproduce the collaboration's
four-dimensional event likelihood.  It reads vector coordinates from the
final-paper PDF, reconstructs the plotted beam-excess/background/reference
signal bins, and builds a model-portable appearance reweighting in L/E.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
PDF = ROOT / "data/experiments/lsnd/sources/aguilar_2001_hep-ex_0104049.pdf"


@dataclass(frozen=True)
class FigureSpec:
    page: int
    axis_low: float
    axis_high: float
    event_axis_high: float
    expected_bins: int
    name: str


SPECS = {
    "fig16_energy": FigureSpec(54, 20.0, 60.0, 35.0, 5, "fig16_energy"),
    "fig24_l_over_e": FigureSpec(62, 0.4, 1.5, 20.0, 11, "fig24_l_over_e"),
}


def _path_points(path: str) -> list[tuple[float, float]]:
    tokens = re.findall(r"[ML]|[-+]?(?:\d+(?:\.\d*)?|\.\d+)", path)
    points: list[tuple[float, float]] = []
    index = 0
    while index < len(tokens):
        if tokens[index] in {"M", "L"}:
            points.append((float(tokens[index + 1]), float(tokens[index + 2])))
            index += 3
        else:
            index += 1
    return points


def _style_colour(style: str) -> str | None:
    colours = {
        "stroke:rgb(100%,0%,0%)": "other_background",
        "stroke:_mc_rgb": "unused",
        "stroke:rgb(0%,100%,0%)": "intrinsic_background",
        "stroke:rgb(0%,0%,100%)": "signal_plus_background",
        "stroke:rgb(0%,0%,0%)": "black",
    }
    for key, value in colours.items():
        if key in style:
            return value
    return None


def _render_svg(spec: FigureSpec, scratch: Path) -> Path:
    scratch.mkdir(parents=True, exist_ok=True)
    output = scratch / f"page{spec.page}.svg"
    subprocess.run(
        [
            "pdftocairo", "-f", str(spec.page), "-l", str(spec.page),
            "-svg", str(PDF), str(output),
        ],
        check=True,
    )
    # Poppler versions disagree on whether they append .svg.
    appended = output.with_suffix(".svg.svg")
    return appended if appended.exists() else output


def _extract(spec: FigureSpec, scratch: Path) -> pd.DataFrame:
    svg = _render_svg(spec, scratch)
    root = ET.parse(svg).getroot()
    paths: list[tuple[str, list[tuple[float, float]]]] = []
    for element in root.iter():
        if not element.tag.endswith("path"):
            continue
        style = element.get("style", "")
        if "stroke-width:8.75" not in style:
            continue
        colour = _style_colour(style)
        points = _path_points(element.get("d", ""))
        if colour and len(points) >= 2:
            paths.append((colour, points))

    frame = max(
        (points for colour, points in paths if colour == "black"),
        key=lambda points: (max(x for x, _ in points) - min(x for x, _ in points))
        * (max(y for _, y in points) - min(y for _, y in points)),
    )
    plot_x_low, plot_x_high = min(x for x, _ in frame), max(x for x, _ in frame)
    plot_y_high = max(y for _, y in frame)

    coloured = [(colour, points) for colour, points in paths if colour != "black"]
    red = next(points for colour, points in coloured if colour == "other_background")
    zero_y = red[0][1]

    # Every horizontal coloured segment spans one plotted bin.  Their common
    # x edges define the binning exactly in the PDF's vector coordinate system.
    horizontal: dict[str, list[tuple[float, float, float]]] = {}
    for colour, points in coloured:
        for first, second in zip(points[:-1], points[1:]):
            if abs(first[1] - second[1]) < 1e-5 and abs(first[0] - second[0]) > 1.0:
                horizontal.setdefault(colour, []).append(
                    (min(first[0], second[0]), max(first[0], second[0]), first[1])
                )
    edges = sorted({value for spans in horizontal.values() for a, b, _ in spans for value in (a, b)})
    edges = [value for value in edges if plot_x_low - 1e-3 <= value <= plot_x_high + 1e-3]
    # Legend samples can create unrelated short edges; retain the evenly spaced
    # run spanning the plot frame.
    candidate = np.array(edges, dtype=float)
    runs: list[np.ndarray] = []
    for start in range(len(candidate) - spec.expected_bins):
        run = candidate[start:start + spec.expected_bins + 1]
        if abs(run[0] - plot_x_low) < 2 and abs(run[-1] - plot_x_high) < 2:
            runs.append(run)
    if not runs:
        raise RuntimeError(f"could not identify {spec.expected_bins} bins in {spec.name}")
    x_edges_pdf = min(runs, key=lambda run: np.std(np.diff(run)))

    def event_value(y_pdf: float) -> float:
        return (y_pdf - zero_y) * spec.event_axis_high / (plot_y_high - zero_y)

    def axis_value(x_pdf: float) -> float:
        return spec.axis_low + (x_pdf - plot_x_low) * (
            spec.axis_high - spec.axis_low
        ) / (plot_x_high - plot_x_low)

    rows = []
    black_paths = [points for colour, points in paths if colour == "black" and points is not frame]
    for left, right in zip(x_edges_pdf[:-1], x_edges_pdf[1:]):
        center = (left + right) / 2
        cumulative: dict[str, float] = {}
        for colour in ("other_background", "intrinsic_background", "signal_plus_background"):
            spans = horizontal.get(colour, [])
            matches = [y for a, b, y in spans if a - 1e-3 <= center <= b + 1e-3]
            if not matches:
                raise RuntimeError(f"missing {colour} in {spec.name} bin at {center}")
            cumulative[colour] = event_value(max(matches))

        # The marker splits its horizontal error bar into two pieces.  Select
        # equal-y horizontal black segments that terminate near this bin centre.
        segments = []
        verticals = []
        for points in black_paths:
            for first, second in zip(points[:-1], points[1:]):
                if abs(first[1] - second[1]) < 1e-5:
                    a, b = sorted((first[0], second[0]))
                    if left - 2 <= a <= right + 2 and left - 2 <= b <= right + 2:
                        segments.append((a, b, first[1]))
                if abs(first[0] - second[0]) < 1e-5 and abs(first[0] - center) < 3:
                    verticals.append((min(first[1], second[1]), max(first[1], second[1])))
        pairs = []
        for a1, b1, y1 in segments:
            for a2, b2, y2 in segments:
                if abs(y1 - y2) < 1e-4 and b1 < center < a2:
                    pairs.append((abs(center - (b1 + a2) / 2), y1))
        if not pairs or not verticals:
            raise RuntimeError(f"missing data/error bar in {spec.name} bin at {center}")
        data_y = min(pairs)[1]
        error_low_pdf = min(a for a, _ in verticals)
        error_high_pdf = max(b for _, b in verticals)
        rows.append({
            "bin_low": axis_value(left),
            "bin_high": axis_value(right),
            "bin_center": axis_value(center),
            "beam_excess": event_value(data_y),
            "error_low": event_value(data_y) - event_value(error_low_pdf),
            "error_high": event_value(error_high_pdf) - event_value(data_y),
            "other_background": cumulative["other_background"],
            "intrinsic_background": cumulative["intrinsic_background"] - cumulative["other_background"],
            "reference_signal": cumulative["signal_plus_background"] - cumulative["intrinsic_background"],
            "stack_total": cumulative["signal_plus_background"],
        })
    return pd.DataFrame(rows)


def _mean_x2(low: np.ndarray, high: np.ndarray) -> np.ndarray:
    return (high**3 - low**3) / (3.0 * (high - low))


def _mean_probability(low: np.ndarray, high: np.ndarray, mass: float) -> np.ndarray:
    nodes, weights = np.polynomial.legendre.leggauss(48)
    x = 0.5 * (high - low)[:, None] * nodes[None, :] + 0.5 * (high + low)[:, None]
    return 0.5 * np.sum(weights[None, :] * np.sin(1.27 * mass * x) ** 2, axis=1)


def _build_kernel(table: pd.DataFrame, x_low: np.ndarray, x_high: np.ndarray) -> np.ndarray:
    reference_signal = table["reference_signal"].to_numpy(float)
    relative_kernel = np.maximum(reference_signal, 0.0) / _mean_x2(x_low, x_high)
    # Final paper: 33300 events for 100% transmutation before gamma selection;
    # correlated-gamma efficiency for R_gamma > 10 is 0.39.
    selected_full_transmutation = 33300.0 * 0.39
    return relative_kernel / relative_kernel.sum() * selected_full_transmutation


def _scan(table: pd.DataFrame, x_low: np.ndarray, x_high: np.ndarray) -> pd.DataFrame:
    kernel = _build_kernel(table, x_low, x_high)
    background = table[["other_background", "intrinsic_background"]].sum(axis=1).to_numpy(float)
    data = table["beam_excess"].to_numpy(float)
    sigma = 0.5 * (table["error_low"].to_numpy(float) + table["error_high"].to_numpy(float))
    inverse_variance = 1.0 / sigma**2
    signal_normalization_sigma = float(np.hypot(0.10, 0.07))
    background_normalization_sigma = 2.3 / 16.9
    prior_precision = np.diag([
        1.0 / signal_normalization_sigma**2,
        1.0 / background_normalization_sigma**2,
    ])
    prior_mean = np.ones(2, dtype=float)
    masses = np.geomspace(1e-2, 1e2, 241)
    amplitudes = np.geomspace(1e-4, 1.0, 241)
    rows = []
    for mass in masses:
        unit_probability = _mean_probability(x_low, x_high, mass)
        signal_at_unit_amplitude = kernel * unit_probability
        for amplitude in amplitudes:
            signal = amplitude * signal_at_unit_amplitude
            design = np.column_stack([signal, background])
            normal_matrix = design.T @ (inverse_variance[:, None] * design) + prior_precision
            right_hand_side = design.T @ (inverse_variance * data) + prior_precision @ prior_mean
            profiled = np.linalg.solve(normal_matrix, right_hand_side)
            signal_scale, background_scale = (float(profiled[0]), float(profiled[1]))
            prediction = signal_scale * signal + background_scale * background
            residual_chi2 = float(np.sum((data - prediction) ** 2 * inverse_variance))
            prior_chi2 = (
                ((signal_scale - 1.0) / signal_normalization_sigma) ** 2
                + ((background_scale - 1.0) / background_normalization_sigma) ** 2
            )
            rows.append((mass, amplitude, residual_chi2 + prior_chi2,
                         signal_scale, background_scale))
    result = pd.DataFrame(rows, columns=[
        "delta_m2_41_eV2", "sin2_2theta_mue", "chi2",
        "profiled_signal_normalization", "profiled_background_normalization",
    ])
    result["delta_chi2"] = result["chi2"] - result["chi2"].min()
    return result


def _energy_bin_l_over_e(table: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    # IBD kinematics at these energies give E_nu approximately E_e + 1.806 MeV.
    # The order reverses when converting energy edges to L/E edges.
    energy_low = table["bin_low"].to_numpy(float)
    energy_high = table["bin_high"].to_numpy(float)
    return 30.0 / (energy_high + 1.806), 30.0 / (energy_low + 1.806)


def _plot_extraction(table: pd.DataFrame, output: Path, xlabel: str) -> None:
    width = table["bin_high"] - table["bin_low"]
    fig, ax = plt.subplots(figsize=(8.0, 5.8), constrained_layout=True)
    bottom = np.zeros(len(table))
    for column, label, colour in (
        ("other_background", "Other background", "#e15759"),
        ("intrinsic_background", r"Intrinsic $\bar\nu_e$ background", "#59a14f"),
        ("reference_signal", r"Published low-$\Delta m^2$ signal", "#4e79a7"),
    ):
        values = table[column].to_numpy(float)
        ax.bar(table["bin_center"], values, width=width, bottom=bottom,
               color=colour, alpha=0.42, edgecolor=colour, label=label)
        bottom += values
    ax.errorbar(table["bin_center"], table["beam_excess"],
                yerr=np.vstack([table["error_low"], table["error_high"]]),
                fmt="o", color="black", capsize=3, label="Beam excess")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set(xlabel=xlabel, ylabel="Beam-excess events")
    ax.legend(frameon=False)
    fig.savefig(output, dpi=220)
    plt.close(fig)


def _plot_scan(scan: pd.DataFrame, output: Path) -> None:
    pivot = scan.pivot(index="delta_m2_41_eV2", columns="sin2_2theta_mue", values="delta_chi2")
    x = pivot.columns.to_numpy(float)
    y = pivot.index.to_numpy(float)
    z = pivot.to_numpy(float)
    best = scan.loc[scan["chi2"].idxmin()]
    fig, ax = plt.subplots(figsize=(7.3, 5.8), constrained_layout=True)
    mesh = ax.pcolormesh(x, y, np.minimum(z, 20), shading="auto", cmap="viridis_r")
    ax.contour(x, y, z, levels=[4.605, 9.210], colors=["#d62728", "#ff7f0e"], linewidths=2)
    ax.scatter([best.sin2_2theta_mue], [best.delta_m2_41_eV2], marker="*", s=100, color="white", edgecolor="black")
    ax.scatter([0.003], [1.2], marker="x", s=75, color="cyan", linewidth=2, label="LSND published best fit")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set(xlabel=r"$\sin^2(2\theta_{\mu e})$", ylabel=r"$\Delta m^2_{41}\;[\mathrm{eV}^2]$")
    ax.legend(frameon=False)
    fig.colorbar(mesh, ax=ax, label=r"$\Delta\chi^2$ (public $L/E$ approximation)")
    fig.savefig(output, dpi=220)
    plt.close(fig)


def _plot_scan_comparison(
    energy_scan: pd.DataFrame,
    le_scan: pd.DataFrame,
    output: Path,
    *,
    level: float = 4.605,
    confidence_label: str = "90%",
) -> None:
    fig, ax = plt.subplots(figsize=(7.3, 5.8), constrained_layout=True)
    for scan, colour, linestyle, label in (
        (energy_scan, "#d62728", "-", "Fig. 16: five energy bins"),
        (le_scan, "#1f77b4", "--", r"Fig. 24: eleven $L/E$ bins"),
    ):
        pivot = scan.pivot(index="delta_m2_41_eV2", columns="sin2_2theta_mue", values="delta_chi2")
        ax.contour(
            pivot.columns.to_numpy(float), pivot.index.to_numpy(float), pivot.to_numpy(float),
            levels=[level], colors=[colour], linestyles=[linestyle], linewidths=[2.0],
        )
    ax.scatter([0.003], [1.2], marker="x", s=80, color="black", linewidth=2,
               label="LSND published best fit")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set(xlabel=r"$\sin^2(2\theta_{\mu e})$", ylabel=r"$\Delta m^2_{41}\;[\mathrm{eV}^2]$")
    handles = [
        Line2D([0], [0], color="#d62728", linestyle="-", linewidth=2,
               label=f"Fig. 16: five energy bins ({confidence_label})"),
        Line2D([0], [0], color="#1f77b4", linestyle="--", linewidth=2,
               label=rf"Fig. 24: eleven $L/E$ bins ({confidence_label})"),
        Line2D([0], [0], color="black", marker="x", linestyle="none", markersize=8,
               label="LSND published best fit"),
    ]
    ax.legend(handles=handles, frameon=False)
    fig.savefig(output, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/studies/lsnd_public_spectrum_reweighting/latest")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    scratch = args.output / "extraction_work"
    tables = {name: _extract(spec, scratch) for name, spec in SPECS.items()}
    for name, table in tables.items():
        table.to_csv(args.output / f"{name}.csv", index=False)
    _plot_extraction(tables["fig16_energy"], args.output / "fig16_reconstructed.png", r"$E_e\;[\mathrm{MeV}]$")
    _plot_extraction(tables["fig24_l_over_e"], args.output / "fig24_reconstructed.png", r"$L_\nu/E_\nu\;[\mathrm{m/MeV}]$")
    fig24 = tables["fig24_l_over_e"]
    scan = _scan(
        fig24,
        fig24["bin_low"].to_numpy(float),
        fig24["bin_high"].to_numpy(float),
    )
    scan.to_csv(args.output / "fig24_three_plus_one_scan.csv", index=False)
    _plot_scan(scan, args.output / "fig24_three_plus_one_parameter_space.png")
    fig16_x_low, fig16_x_high = _energy_bin_l_over_e(tables["fig16_energy"])
    energy_scan = _scan(tables["fig16_energy"], fig16_x_low, fig16_x_high)
    energy_scan.to_csv(args.output / "fig16_three_plus_one_scan.csv", index=False)
    _plot_scan(energy_scan, args.output / "fig16_three_plus_one_parameter_space.png")
    _plot_scan_comparison(
        energy_scan, scan, args.output / "fig16_fig24_contour_comparison.png"
    )
    _plot_scan_comparison(
        energy_scan,
        scan,
        args.output / "fig16_fig24_99pct_contour_comparison.png",
        level=9.210,
        confidence_label="99%",
    )
    best = scan.loc[scan["chi2"].idxmin()]
    energy_best = energy_scan.loc[energy_scan["chi2"].idxmin()]
    metadata = {
        "source_pdf": str(PDF.relative_to(ROOT)),
        "source_pages": {name: spec.page for name, spec in SPECS.items()},
        "data_semantics": "beam-on minus beam-off; neutrino backgrounds remain",
        "likelihood": "independent-bin Gaussian using plotted asymmetric errors averaged per bin, with analytic profiling of signal and total neutrino-background normalizations",
        "kernel": "low-delta-m2 reference signal divided by bin-averaged (L/E)^2 and normalized to 33300*0.39 selected full-transmutation events",
        "limitations": [
            "not the LSND four-variable event likelihood",
            "no public bin-to-bin covariance",
            "no public bin-to-bin covariance or background-shape nuisance model",
            "low-delta-m2 reference-template inversion and within-bin kernel shape are approximations",
        ],
        "profiled_normalization_priors": {
            "signal_relative_sigma": float(np.hypot(0.10, 0.07)),
            "signal_evidence": "33300 +/- 3300 full-transmutation events and 7% correlated-gamma efficiency uncertainty",
            "background_relative_sigma": 2.3 / 16.9,
            "background_evidence": "Rgamma > 10 neutrino background 16.9 +/- 2.3 events",
            "correlation_model": "one common scale for the plotted neutrino-background stack; no unpublished bin covariance invented",
        },
        "closure": {
            name: {
                "beam_excess_sum": float(table.beam_excess.sum()),
                "background_sum": float((table.other_background + table.intrinsic_background).sum()),
                "reference_signal_sum": float(table.reference_signal.sum()),
            } for name, table in tables.items()
        },
        "best_fit": {
            "delta_m2_41_eV2": float(best.delta_m2_41_eV2),
            "sin2_2theta_mue": float(best.sin2_2theta_mue),
            "chi2": float(best.chi2),
            "profiled_signal_normalization": float(best.profiled_signal_normalization),
            "profiled_background_normalization": float(best.profiled_background_normalization),
        },
        "fig16_energy_best_fit": {
            "delta_m2_41_eV2": float(energy_best.delta_m2_41_eV2),
            "sin2_2theta_mue": float(energy_best.sin2_2theta_mue),
            "chi2": float(energy_best.chi2),
            "profiled_signal_normalization": float(energy_best.profiled_signal_normalization),
            "profiled_background_normalization": float(energy_best.profiled_background_normalization),
        },
    }
    def nearest_delta(candidate: pd.DataFrame) -> float:
        distance = (
            np.abs(np.log(candidate["delta_m2_41_eV2"].to_numpy(float) / 1.2))
            + np.abs(np.log(candidate["sin2_2theta_mue"].to_numpy(float) / 0.003))
        )
        return float(candidate.iloc[int(np.argmin(distance))].delta_chi2)

    checks = {
        "fig16_background_closes_to_published_16p9": abs(metadata["closure"]["fig16_energy"]["background_sum"] - 16.9) < 0.2,
        "fig16_signal_closes_to_published_32p2": abs(metadata["closure"]["fig16_energy"]["reference_signal_sum"] - 32.2) < 1.0,
        "fig16_beam_excess_closes_to_published_49p1": abs(metadata["closure"]["fig16_energy"]["beam_excess_sum"] - 49.1) < 2.0,
        "published_best_fit_inside_fig16_delta_chi2_2p3": nearest_delta(energy_scan) < 2.3,
        "published_best_fit_inside_fig24_delta_chi2_2p3": nearest_delta(scan) < 2.3,
    }
    metadata["validation"] = {"checks": checks, "passed": all(checks.values())}
    if not metadata["validation"]["passed"]:
        raise RuntimeError(f"LSND public-spectrum validation failed: {checks}")
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
