from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw

STUDY_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(STUDY_ROOT / "src"))

from geometry import AxisTransform, transform_to_dict  # noqa: E402
from pdf_extract import (  # noqa: E402
    calibrate_axes,
    detect_plot_box,
    extract_horizontal_bins,
    load_page,
    select_target_curve,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _load_truth(path: Path) -> np.ndarray:
    first_line = path.read_text(encoding="utf-8").splitlines()[0]
    values = np.fromstring(first_line, sep=" ")
    if values.size != 200 or np.any(values < 0) or not np.all(np.isfinite(values)):
        raise ValueError("MINERvA truth line must contain 200 finite non-negative values")
    return values


def _render_pdf_page(pdf: Path, page_number: int, destination: Path, dpi: int = 180) -> None:
    renderer = shutil.which("pdftocairo")
    if renderer is None:
        raise RuntimeError("pdftocairo is required to create the visual overlay")
    prefix = destination.with_suffix("")
    subprocess.run(
        [
            renderer,
            "-f",
            str(page_number),
            "-l",
            str(page_number),
            "-singlefile",
            "-png",
            "-r",
            str(dpi),
            str(pdf),
            str(prefix),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    rendered = prefix.with_suffix(".png")
    if rendered != destination:
        rendered.replace(destination)


def _page_to_pixel(points: Iterable[tuple[float, float]], page_width: float, page_height: float, image: Image.Image) -> list[tuple[float, float]]:
    scale_x = image.width / page_width
    scale_y = image.height / page_height
    return [(x * scale_x, y * scale_y) for x, y in points]


def _stairs_page_points(
    energy_edges: np.ndarray,
    values: np.ndarray,
    x_transform: AxisTransform,
    y_transform: AxisTransform,
) -> list[tuple[float, float]]:
    if energy_edges.size != values.size + 1:
        raise ValueError("stair plot requires one more edge than value")
    points: list[tuple[float, float]] = []
    for index, value in enumerate(values):
        y = float(y_transform.inverse(value))
        left = float(x_transform.inverse(energy_edges[index]))
        right = float(x_transform.inverse(energy_edges[index + 1]))
        if not points:
            points.append((left, y))
        elif points[-1][1] != y:
            points.append((left, y))
        points.append((right, y))
    return points


def _point_to_polyline_distance(point: np.ndarray, polyline: np.ndarray) -> float:
    starts = polyline[:-1]
    ends = polyline[1:]
    vectors = ends - starts
    lengths_squared = np.sum(vectors * vectors, axis=1)
    valid = lengths_squared > 0
    if not np.any(valid):
        return float(np.min(np.linalg.norm(polyline - point, axis=1)))
    starts = starts[valid]
    vectors = vectors[valid]
    lengths_squared = lengths_squared[valid]
    parameters = np.clip(np.sum((point - starts) * vectors, axis=1) / lengths_squared, 0.0, 1.0)
    projections = starts + parameters[:, None] * vectors
    return float(np.min(np.linalg.norm(projections - point, axis=1)))


def _symmetric_polyline_vertex_distance(first: np.ndarray, second: np.ndarray) -> float:
    """Symmetric vertex-to-segment Hausdorff approximation in PDF points."""

    first_to_second = max(_point_to_polyline_distance(point, second) for point in first)
    second_to_first = max(_point_to_polyline_distance(point, first) for point in second)
    return max(first_to_second, second_to_first)


def _draw_overlay(
    rendered_page: Path,
    destination: Path,
    page_width: float,
    page_height: float,
    path_points: list[tuple[float, float]],
    color: tuple[int, int, int, int],
) -> None:
    image = Image.open(rendered_page).convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    pixel_points = _page_to_pixel(path_points, page_width, page_height, image)
    draw.line(pixel_points, fill=color, width=4, joint="curve")
    Image.alpha_composite(image, overlay).convert("RGB").save(destination)


def _write_csv(path: Path, rows: list[dict[str, float]], truth_scale: float, truth: np.ndarray) -> None:
    fieldnames = [
        "energy_low_GeV",
        "energy_high_GeV",
        "flux_pdf_plot_units",
        "flux_pdf_neutrinos_per_m2_per_POT",
        "flux_official_neutrinos_per_m2_per_POT",
        "absolute_difference",
        "relative_difference",
        "source_page_x_low",
        "source_page_x_high",
        "source_page_y",
        "is_lower_clipped",
        "is_upper_clipped",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for index, row in enumerate(rows):
            recovered = row["flux_plot_units"] / truth_scale
            official = truth[index]
            relative_difference = "" if official == 0 else f"{(recovered - official) / official:.12g}"
            writer.writerow(
                {
                    "energy_low_GeV": f"{row['energy_low_GeV']:.12g}",
                    "energy_high_GeV": f"{row['energy_high_GeV']:.12g}",
                    "flux_pdf_plot_units": f"{row['flux_plot_units']:.12g}",
                    "flux_pdf_neutrinos_per_m2_per_POT": f"{recovered:.12g}",
                    "flux_official_neutrinos_per_m2_per_POT": f"{official:.12g}",
                    "absolute_difference": f"{recovered - official:.12g}",
                    "relative_difference": relative_difference,
                    "source_page_x_low": f"{row['source_page_x_low']:.12g}",
                    "source_page_x_high": f"{row['source_page_x_high']:.12g}",
                    "source_page_y": f"{row['source_page_y']:.12g}",
                    "is_lower_clipped": str(bool(row["is_lower_clipped"])).lower(),
                    "is_upper_clipped": str(bool(row["is_upper_clipped"])).lower(),
                }
            )


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate PDF-vector flux extraction against MINERvA truth")
    parser.add_argument(
        "--config",
        type=Path,
        default=STUDY_ROOT / "configs" / "minerva_numu_fhc.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=STUDY_ROOT.parents[1] / "outputs" / "checks" / "numi_flux_pdf_extraction" / "minerva_numu_fhc",
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    pdf_path = STUDY_ROOT / config["source_pdf"]
    truth_path = STUDY_ROOT / config["truth_file"]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    document, page = load_page(pdf_path, int(config["page_number_1_based"]))
    try:
        curve, candidates = select_target_curve(page, config["curve_selector"])
        box = detect_plot_box(page, curve)
        x_transform, y_transform, object_audit = calibrate_axes(page, box, config["axis"])
        recovered_bins = extract_horizontal_bins(curve, x_transform, y_transform, box=box)

        truth_all = _load_truth(truth_path)
        bin_width = float(config["truth_bin_width_GeV"])
        first_energy = float(config["truth_energy_min_GeV"])
        maximum_energy = float(config["plot_energy_max_GeV"])
        expected_bin_count = int(round((maximum_energy - first_energy) / bin_width))
        if len(recovered_bins) != expected_bin_count:
            raise ValueError(
                f"PDF contains {len(recovered_bins)} horizontal bins but {expected_bin_count} are expected"
            )
        truth = truth_all[:expected_bin_count]
        truth_scale = float(config["truth_to_plot_scale"])
        recovered_release_units = np.asarray(
            [row["flux_plot_units"] / truth_scale for row in recovered_bins], dtype=float
        )
        nonzero_truth = truth != 0
        clipped_bin_count = sum(
            bool(row["is_lower_clipped"] or row["is_upper_clipped"]) for row in recovered_bins
        )
        relative_errors = np.full_like(truth, np.nan, dtype=float)
        relative_errors[nonzero_truth] = np.abs(
            (recovered_release_units[nonzero_truth] - truth[nonzero_truth]) / truth[nonzero_truth]
        )
        integral_relative_error = float(abs(recovered_release_units.sum() - truth.sum()) / truth.sum())

        energy_edges = first_energy + np.arange(expected_bin_count + 1) * bin_width
        truth_page_path = _stairs_page_points(
            energy_edges, truth * truth_scale, x_transform, y_transform
        )
        recovered_page_path = _stairs_page_points(
            np.asarray(
                [recovered_bins[0]["energy_low_GeV"]]
                + [row["energy_high_GeV"] for row in recovered_bins]
            ),
            np.asarray([row["flux_plot_units"] for row in recovered_bins]),
            x_transform,
            y_transform,
        )

        source_points = np.asarray(curve["pts"], dtype=float)
        truth_points = np.asarray(truth_page_path, dtype=float)
        maximum_page_error = _symmetric_polyline_vertex_distance(source_points, truth_points)

        _write_csv(output / "recovered_flux.csv", recovered_bins, truth_scale, truth)
        detected = {
            "selected_curve": {
                "page_object_index": candidates[0]["index"],
                "stroke_rgb": curve.get("stroking_color"),
                "line_width_points": curve.get("linewidth"),
                "point_count": len(curve["pts"]),
                "bounding_box": {
                    "x0": curve["x0"],
                    "x1": curve["x1"],
                    "top": curve["top"],
                    "bottom": curve["bottom"],
                },
            },
            "curve_candidates": candidates,
            "axis_detection": object_audit,
            "inferred_physical_bounds": {
                "x_min": min(float(x_transform.forward(box.x_left)), float(x_transform.forward(box.x_right))),
                "x_max": max(float(x_transform.forward(box.x_left)), float(x_transform.forward(box.x_right))),
                "y_min": min(float(y_transform.forward(box.top)), float(y_transform.forward(box.bottom))),
                "y_max": max(float(y_transform.forward(box.top)), float(y_transform.forward(box.bottom))),
                "x_scale": x_transform.scale,
                "y_scale": y_transform.scale,
            },
        }
        (output / "detected_objects.json").write_text(
            json.dumps(detected, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        rendered = output / "source_page.png"
        _render_pdf_page(pdf_path, int(config["page_number_1_based"]), rendered)
        _draw_overlay(
            rendered,
            output / "overlay_truth_vs_pdf.png",
            float(page.width),
            float(page.height),
            truth_page_path,
            (255, 0, 0, 150),
        )
        _draw_overlay(
            rendered,
            output / "overlay_recovered_vs_pdf.png",
            float(page.width),
            float(page.height),
            recovered_page_path,
            (0, 180, 255, 150),
        )

        thresholds = config["validation"]
        checks = {
            "bin_count_matches": len(recovered_bins) == expected_bin_count,
            "x_axis_classified_linear": x_transform.scale == "linear",
            "y_axis_classified_linear": y_transform.scale == "linear",
            "maximum_relative_value_error_passes": float(np.nanmax(relative_errors))
            <= float(thresholds["maximum_relative_value_error"]),
            "maximum_absolute_page_error_passes": maximum_page_error
            <= float(thresholds["maximum_absolute_page_error_points"]),
            "integral_relative_error_passes": integral_relative_error
            <= float(thresholds["maximum_integral_relative_error"]),
            "no_clipped_or_censored_bins": clipped_bin_count == 0,
        }
        validation = {
            "status": "pass" if all(checks.values()) else "fail",
            "scientific_scope": "MINERvA PDF extraction algorithm validation only",
            "source_pdf": str(pdf_path),
            "source_pdf_sha256": _sha256(pdf_path),
            "truth_file": str(truth_path),
            "truth_file_sha256": _sha256(truth_path),
            "x_transform": transform_to_dict(x_transform),
            "y_transform": transform_to_dict(y_transform),
            "recovered_bin_count": len(recovered_bins),
            "maximum_relative_value_error": float(np.nanmax(relative_errors)),
            "median_relative_value_error": float(np.nanmedian(relative_errors)),
            "zero_truth_bin_count": int(np.count_nonzero(~nonzero_truth)),
            "clipped_or_censored_bin_count": clipped_bin_count,
            "integral_relative_error": integral_relative_error,
            "maximum_truth_path_distance_points": maximum_page_error,
            "source_line_width_points": float(curve["linewidth"]),
            "checks": checks,
        }
        (output / "validation.json").write_text(
            json.dumps(validation, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(json.dumps(validation, indent=2, ensure_ascii=False))
        return 0 if validation["status"] == "pass" else 1
    finally:
        document.close()


if __name__ == "__main__":
    raise SystemExit(main())
