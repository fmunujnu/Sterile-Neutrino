from __future__ import annotations

import csv
import json
from pathlib import Path
import shutil
import subprocess
import sys

from PIL import Image, ImageDraw

STUDY_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = STUDY_ROOT.parents[1]
sys.path.insert(0, str(STUDY_ROOT / "src"))

from geometry import transform_to_dict  # noqa: E402
from pdf_extract import (  # noqa: E402
    calibrate_axes,
    detect_plot_box,
    extract_horizontal_bins,
    load_page,
    select_target_curve,
)


PANELS = [
    (5, "rhc_numu", 0.10, 0.52),
    (5, "rhc_nue", 0.52, 0.95),
    (6, "fhc_numu_plus_numubar", 0.10, 0.52),
    (6, "fhc_nue_plus_nuebar", 0.52, 0.95),
]
NEW_FLUX_BLUE = [0.0039215686, 0.4509803922, 0.6980392157]


def _render_page(pdf: Path, page_number: int, destination: Path) -> None:
    renderer = shutil.which("pdftocairo")
    if renderer is None:
        raise RuntimeError("pdftocairo is required for local overlay rendering")
    prefix = destination.with_suffix("")
    subprocess.run(
        [renderer, "-f", str(page_number), "-l", str(page_number), "-singlefile", "-png", "-r", "180", str(pdf), str(prefix)],
        check=True,
        capture_output=True,
        text=True,
    )


def _write_flux_csv(path: Path, rows: list[dict[str, float]]) -> None:
    fields = ["energy_low_GeV", "energy_high_GeV", "flux_plot_units", "is_lower_clipped", "is_upper_clipped"]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "energy_low_GeV": f"{row['energy_low_GeV']:.12g}",
                    "energy_high_GeV": f"{row['energy_high_GeV']:.12g}",
                    "flux_plot_units": f"{row['flux_plot_units']:.12g}",
                    "is_lower_clipped": str(bool(row["is_lower_clipped"])).lower(),
                    "is_upper_clipped": str(bool(row["is_upper_clipped"])).lower(),
                }
            )


def _read_and_project_csv(path: Path, x_transform, y_transform) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            left = float(x_transform.inverse(float(row["energy_low_GeV"])))
            right = float(x_transform.inverse(float(row["energy_high_GeV"])))
            y = float(y_transform.inverse(float(row["flux_plot_units"])))
            points.extend([(left, y), (right, y)])
    return points


def _draw_page_overlay(source: Path, destination: Path, page_width: float, page_height: float, paths) -> None:
    image = Image.open(source).convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    scale_x = image.width / page_width
    scale_y = image.height / page_height
    for path in paths:
        draw.line([(x * scale_x, y * scale_y) for x, y in path], fill=(220, 0, 180, 170), width=4)
    Image.alpha_composite(image, overlay).convert("RGB").save(destination)


def main() -> int:
    pdf = STUDY_ROOT / "data" / "microboone_note_1129" / "MICROBOONE-NOTE-1129-PUB.pdf"
    output = REPOSITORY_ROOT / "outputs" / "checks" / "numi_flux_pdf_extraction" / "microboone_pages_5_6"
    output.mkdir(parents=True, exist_ok=True)
    page_paths: dict[int, list[list[tuple[float, float]]]] = {5: [], 6: []}
    page_sizes: dict[int, tuple[float, float]] = {}
    results = []

    for page_number, name, roi_left, roi_right in PANELS:
        document, page = load_page(pdf, page_number)
        try:
            curve, candidates = select_target_curve(
                page,
                {
                    "stroke_rgb": NEW_FLUX_BLUE,
                    "minimum_points": 40,
                    "minimum_horizontal_coverage_fraction": 0.45,
                    "ambiguity_score_fraction": 0.05,
                    "roi_fraction": {"left": roi_left, "right": roi_right, "top": 0.25, "bottom": 0.49},
                },
            )
            box = detect_plot_box(page, curve)
            x_transform, y_transform, audit = calibrate_axes(page, box, {"x_scale": "auto", "y_scale": "auto"})
            rows = extract_horizontal_bins(curve, x_transform, y_transform, box)
            csv_path = output / f"{name}_new_flux.csv"
            _write_flux_csv(csv_path, rows)
            page_paths[page_number].append(_read_and_project_csv(csv_path, x_transform, y_transform))
            page_sizes[page_number] = (float(page.width), float(page.height))
            results.append(
                {
                    "name": name,
                    "page": page_number,
                    "status": "pass",
                    "curve_candidates": len(candidates),
                    "pdf_path_points": len(curve["pts"]),
                    "recovered_horizontal_segments": len(rows),
                    "x_axis": transform_to_dict(x_transform),
                    "y_axis": transform_to_dict(y_transform),
                    "x_bounds_GeV": sorted([float(x_transform.forward(box.x_left)), float(x_transform.forward(box.x_right))]),
                    "y_display_bounds": sorted([float(y_transform.forward(box.top)), float(y_transform.forward(box.bottom))]),
                    "lower_clipped_segments": sum(row["is_lower_clipped"] for row in rows),
                    "upper_clipped_segments": sum(row["is_upper_clipped"] for row in rows),
                    "axis_audit": audit,
                    "csv": str(csv_path),
                }
            )
        finally:
            document.close()

    for page_number in (5, 6):
        rendered = output / f"page_{page_number}_source.png"
        _render_page(pdf, page_number, rendered)
        _draw_page_overlay(rendered, output / f"page_{page_number}_array_redraw_overlay.png", *page_sizes[page_number], page_paths[page_number])

    report = {
        "status": "pass",
        "scope": "MicroBooNE NOTE-1129 PDF pages 5-6 vector extraction; no unpublished truth array",
        "api_or_network_required_to_rerun": False,
        "panels": results,
    }
    (output / "extraction_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
