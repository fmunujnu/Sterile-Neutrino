from __future__ import annotations

import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

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
from postprocess_pages_4_7 import (  # noqa: E402
    TARGET_BIN_WIDTH_GEV,
    TARGET_ENERGY_MAX_GEV,
    derive_antineutrino,
    expand_native_histogram_to_target_grid,
    target_edges,
)


NEW_FLUX_BLUE = [0.0039215686, 0.4509803922, 0.6980392157]


@dataclass(frozen=True)
class SourcePanel:
    page: int
    horn_mode: str
    flavor_family: str
    quantity: str
    roi_left: float
    roi_right: float
    roi_top: float
    roi_bottom: float

    @property
    def key(self) -> tuple[str, str, str]:
        return self.horn_mode, self.flavor_family, self.quantity

    @property
    def name(self) -> str:
        return "_".join(self.key)


# Pages 4-5 contain neutrino components. Pages 6-7 contain the corresponding
# neutrino-plus-antineutrino sums. These are the only relations used to derive
# the four antineutrino arrays.
SOURCE_PANELS = [
    SourcePanel(4, "fhc", "numu", "neutrino", 0.10, 0.52, 0.25, 0.49),
    SourcePanel(4, "fhc", "nue", "neutrino", 0.52, 0.95, 0.25, 0.49),
    SourcePanel(5, "rhc", "numu", "neutrino", 0.10, 0.52, 0.25, 0.49),
    SourcePanel(5, "rhc", "nue", "neutrino", 0.52, 0.95, 0.25, 0.49),
    SourcePanel(6, "fhc", "numu", "neutrino_plus_antineutrino", 0.10, 0.52, 0.25, 0.49),
    SourcePanel(6, "fhc", "nue", "neutrino_plus_antineutrino", 0.52, 0.95, 0.25, 0.49),
    # Figure 4 is placed higher on page 7 than Figures 1-3. This ROI keeps the
    # main flux panel and excludes the blue ratio path below it.
    SourcePanel(7, "rhc", "numu", "neutrino_plus_antineutrino", 0.10, 0.52, 0.05, 0.32),
    SourcePanel(7, "rhc", "nue", "neutrino_plus_antineutrino", 0.52, 0.95, 0.05, 0.32),
]


def _render_page(pdf: Path, page_number: int, destination: Path) -> None:
    renderer = shutil.which("pdftocairo")
    if renderer is None:
        raise RuntimeError("pdftocairo is required for local overlay rendering")
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
            "180",
            str(pdf),
            str(prefix),
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def _draw_page_overlay(
    source: Path,
    destination: Path,
    page_width: float,
    page_height: float,
    paths: list[list[tuple[float, float]]],
) -> None:
    image = Image.open(source).convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    scale_x = image.width / page_width
    scale_y = image.height / page_height
    for path in paths:
        draw.line(
            [(x * scale_x, y * scale_y) for x, y in path],
            fill=(220, 0, 180, 180),
            width=4,
        )
    Image.alpha_composite(image, overlay).convert("RGB").save(destination)


def _write_native_coordinate_audit(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "energy_low_GeV",
        "energy_high_GeV",
        "flux_per_POT_per_cm2_per_100MeV",
        "source_page_x_low_points",
        "source_page_x_high_points",
        "source_page_y_points",
        "is_lower_censored",
        "is_upper_censored",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "energy_low_GeV": f"{row['energy_low_GeV']:.17g}",
                    "energy_high_GeV": f"{row['energy_high_GeV']:.17g}",
                    "flux_per_POT_per_cm2_per_100MeV": f"{row['flux_plot_units']:.17g}",
                    "source_page_x_low_points": f"{row['source_page_x_low']:.17g}",
                    "source_page_x_high_points": f"{row['source_page_x_high']:.17g}",
                    "source_page_y_points": f"{row['source_page_y']:.17g}",
                    "is_lower_censored": str(bool(row["is_lower_clipped"])).lower(),
                    "is_upper_censored": str(bool(row["is_upper_clipped"])).lower(),
                }
            )


def _write_final_flux(
    path: Path,
    rows: list[dict[str, Any]],
    derivation: str,
) -> None:
    fields = [
        "energy_low_GeV",
        "energy_high_GeV",
        "energy_center_GeV",
        "flux_per_POT_per_cm2_per_100MeV",
        "is_censored",
        "derivation",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "energy_low_GeV": f"{row['energy_low_GeV']:.1f}",
                    "energy_high_GeV": f"{row['energy_high_GeV']:.1f}",
                    "energy_center_GeV": f"{row['energy_center_GeV']:.2f}",
                    "flux_per_POT_per_cm2_per_100MeV": f"{row['flux_per_POT_per_cm2_per_100MeV']:.17g}",
                    "is_censored": str(
                        bool(row["is_lower_censored"] or row["is_upper_censored"])
                    ).lower(),
                    "derivation": derivation,
                }
            )


def _project_rows(rows: list[dict[str, Any]], x_transform: Any, y_transform: Any) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for row in rows:
        value = float(row["flux_per_POT_per_cm2_per_100MeV"])
        if value <= 0:
            continue
        y = float(y_transform.inverse(value))
        points.extend(
            [
                (float(x_transform.inverse(row["energy_low_GeV"])), y),
                (float(x_transform.inverse(row["energy_high_GeV"])), y),
            ]
        )
    return points


def _extract_source_panel(pdf: Path, panel: SourcePanel) -> dict[str, Any]:
    document, page = load_page(pdf, panel.page)
    try:
        curve, candidates = select_target_curve(
            page,
            {
                "stroke_rgb": NEW_FLUX_BLUE,
                "minimum_points": 40,
                "minimum_horizontal_coverage_fraction": 0.45,
                "ambiguity_score_fraction": 0.05,
                "roi_fraction": {
                    "left": panel.roi_left,
                    "right": panel.roi_right,
                    "top": panel.roi_top,
                    "bottom": panel.roi_bottom,
                },
            },
        )
        box = detect_plot_box(page, curve)
        x_transform, y_transform, axis_audit = calibrate_axes(
            page, box, {"x_scale": "auto", "y_scale": "auto"}
        )
        if x_transform.scale != "linear" or y_transform.scale != "log10":
            raise ValueError(
                f"{panel.name} axis classification is {x_transform.scale}/{y_transform.scale}, expected linear/log10"
            )
        native_rows = extract_horizontal_bins(curve, x_transform, y_transform, box)
        fine_rows = expand_native_histogram_to_target_grid(native_rows)
        return {
            "panel": panel,
            "native_rows": native_rows,
            "fine_rows": fine_rows,
            "x_transform": x_transform,
            "y_transform": y_transform,
            "page_width": float(page.width),
            "page_height": float(page.height),
            "audit": {
                "name": panel.name,
                "page": panel.page,
                "curve_candidates": len(candidates),
                "pdf_path_points": len(curve["pts"]),
                "native_horizontal_segments": len(native_rows),
                "output_0p1_GeV_bins": len(fine_rows),
                "x_axis": transform_to_dict(x_transform),
                "y_axis": transform_to_dict(y_transform),
                "x_bounds_GeV": sorted(
                    [
                        float(x_transform.forward(box.x_left)),
                        float(x_transform.forward(box.x_right)),
                    ]
                ),
                "y_display_bounds": sorted(
                    [
                        float(y_transform.forward(box.top)),
                        float(y_transform.forward(box.bottom)),
                    ]
                ),
                "lower_censored_native_segments": sum(
                    bool(row["is_lower_clipped"]) for row in native_rows
                ),
                "upper_censored_native_segments": sum(
                    bool(row["is_upper_clipped"]) for row in native_rows
                ),
                "axis_audit": axis_audit,
            },
        }
    finally:
        document.close()


def main() -> int:
    pdf = (
        STUDY_ROOT
        / "data"
        / "microboone_note_1129"
        / "MICROBOONE-NOTE-1129-PUB.pdf"
    )
    output = (
        REPOSITORY_ROOT
        / "outputs"
        / "checks"
        / "numi_flux_pdf_extraction"
        / "microboone_pages_4_7"
    )
    final_directory = output / "flux_arrays"
    source_directory = output / "source_coordinate_audit"
    final_directory.mkdir(parents=True, exist_ok=True)
    source_directory.mkdir(parents=True, exist_ok=True)

    extracted: dict[tuple[str, str, str], dict[str, Any]] = {}
    for panel in SOURCE_PANELS:
        result = _extract_source_panel(pdf, panel)
        extracted[panel.key] = result
        _write_native_coordinate_audit(
            source_directory / f"{panel.name}.csv", result["native_rows"]
        )

    final_arrays: dict[tuple[str, str], list[dict[str, Any]]] = {}
    final_metadata: list[dict[str, Any]] = []
    for horn_mode in ("fhc", "rhc"):
        for family, neutrino_name, antineutrino_name in (
            ("numu", "numu", "numubar"),
            ("nue", "nue", "nuebar"),
        ):
            neutrino = extracted[(horn_mode, family, "neutrino")]["fine_rows"]
            total = extracted[
                (horn_mode, family, "neutrino_plus_antineutrino")
            ]["fine_rows"]
            antineutrino = derive_antineutrino(neutrino, total)
            final_arrays[(horn_mode, neutrino_name)] = neutrino
            final_arrays[(horn_mode, antineutrino_name)] = antineutrino

            neutrino_file = final_directory / f"numi_{horn_mode}_{neutrino_name}_flux.csv"
            antineutrino_file = (
                final_directory / f"numi_{horn_mode}_{antineutrino_name}_flux.csv"
            )
            neutrino_derivation = (
                f"direct New Flux vector path from PDF page "
                f"{extracted[(horn_mode, family, 'neutrino')]['panel'].page}"
            )
            antineutrino_derivation = (
                f"same-mode neutrino-plus-antineutrino total from PDF page "
                f"{extracted[(horn_mode, family, 'neutrino_plus_antineutrino')]['panel'].page} "
                f"minus neutrino component from PDF page "
                f"{extracted[(horn_mode, family, 'neutrino')]['panel'].page}"
            )
            _write_final_flux(neutrino_file, neutrino, neutrino_derivation)
            _write_final_flux(antineutrino_file, antineutrino, antineutrino_derivation)
            final_metadata.extend(
                [
                    {
                        "horn_mode": horn_mode.upper(),
                        "flavor": neutrino_name,
                        "derivation": neutrino_derivation,
                        "csv": str(neutrino_file),
                    },
                    {
                        "horn_mode": horn_mode.upper(),
                        "flavor": antineutrino_name,
                        "derivation": antineutrino_derivation,
                        "csv": str(antineutrino_file),
                    },
                ]
            )

    negative_bins = []
    nonfinite_bins = []
    censored_bins = []
    for (horn_mode, flavor), rows in final_arrays.items():
        for index, row in enumerate(rows):
            value = float(row["flux_per_POT_per_cm2_per_100MeV"])
            if not math.isfinite(value):
                nonfinite_bins.append(
                    {
                        "horn_mode": horn_mode,
                        "flavor": flavor,
                        "bin_index": index,
                        "energy_low_GeV": row["energy_low_GeV"],
                        "value": value,
                    }
                )
            if value < 0:
                negative_bins.append(
                    {
                        "horn_mode": horn_mode,
                        "flavor": flavor,
                        "bin_index": index,
                        "energy_low_GeV": row["energy_low_GeV"],
                        "value": value,
                    }
                )
            if bool(row["is_lower_censored"] or row["is_upper_censored"]):
                censored_bins.append(
                    {
                        "horn_mode": horn_mode,
                        "flavor": flavor,
                        "bin_index": index,
                        "energy_low_GeV": row["energy_low_GeV"],
                    }
                )

    page_paths: dict[int, list[list[tuple[float, float]]]] = {
        page: [] for page in range(4, 8)
    }
    page_sizes: dict[int, tuple[float, float]] = {}
    for panel in SOURCE_PANELS:
        result = extracted[panel.key]
        if panel.quantity == "neutrino":
            displayed_rows = final_arrays[(panel.horn_mode, panel.flavor_family)]
        else:
            neutrino_rows = final_arrays[(panel.horn_mode, panel.flavor_family)]
            antineutrino_rows = final_arrays[
                (
                    panel.horn_mode,
                    "numubar" if panel.flavor_family == "numu" else "nuebar",
                )
            ]
            displayed_rows = []
            for neutrino_row, antineutrino_row in zip(
                neutrino_rows, antineutrino_rows
            ):
                displayed_rows.append(
                    {
                        **neutrino_row,
                        "flux_per_POT_per_cm2_per_100MeV": float(
                            neutrino_row["flux_per_POT_per_cm2_per_100MeV"]
                        )
                        + float(
                            antineutrino_row[
                                "flux_per_POT_per_cm2_per_100MeV"
                            ]
                        ),
                    }
                )
        page_paths[panel.page].append(
            _project_rows(
                displayed_rows, result["x_transform"], result["y_transform"]
            )
        )
        page_sizes[panel.page] = (result["page_width"], result["page_height"])

    for page_number in range(4, 8):
        rendered = output / f"page_{page_number}_source.png"
        _render_page(pdf, page_number, rendered)
        _draw_page_overlay(
            rendered,
            output / f"page_{page_number}_final_arrays_overlay.png",
            *page_sizes[page_number],
            page_paths[page_number],
        )

    report = {
        "status": (
            "pass"
            if not negative_bins and not nonfinite_bins and not censored_bins
            else "review_required"
        ),
        "scope": "MicroBooNE NOTE-1129 PDF pages 4-7 New Flux vector extraction",
        "core_extractor_modified": False,
        "api_or_network_required_to_rerun": False,
        "source_pdf_pages": [4, 5, 6, 7],
        "target_grid": {
            "energy_min_GeV": 0.0,
            "energy_max_GeV": TARGET_ENERGY_MAX_GEV,
            "bin_width_GeV": TARGET_BIN_WIDTH_GEV,
            "bin_count": len(target_edges()) - 1,
            "wide_native_step_policy": "piecewise-constant expansion; no interpolation",
        },
        "flux_unit": "neutrinos / POT / cm^2 / 100 MeV",
        "source_panels": [result["audit"] for result in extracted.values()],
        "final_flux_arrays": final_metadata,
        "checks": {
            "final_array_count": len(final_metadata),
            "all_arrays_have_50_bins": all(
                len(rows) == 50 for rows in final_arrays.values()
            ),
            "negative_flux_bins": negative_bins,
            "nonfinite_bins": nonfinite_bins,
            "censored_final_bins": censored_bins,
            "subtraction_definition": "antineutrino = same-mode total - neutrino",
        },
        "scientific_boundary": (
            "Values are recovered from displayed vector paths, not an official released ROOT array. "
            "Subtracted antineutrino values inherit digitization uncertainty from both source curves."
        ),
    }
    report_path = output / "extraction_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    console_summary = {
        "status": report["status"],
        "output_directory": str(output),
        "final_flux_array_count": len(final_metadata),
        "bins_per_array": len(target_edges()) - 1,
        "negative_bins": len(negative_bins),
        "nonfinite_bins": len(nonfinite_bins),
        "censored_bins": len(censored_bins),
        "full_audit_report": str(report_path),
    }
    print(json.dumps(console_summary, indent=2, ensure_ascii=True))
    return 0 if report["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
