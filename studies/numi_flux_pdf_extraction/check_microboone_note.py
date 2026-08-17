from __future__ import annotations

import json
from pathlib import Path
import sys

STUDY_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(STUDY_ROOT / "src"))

from pdf_extract import (  # noqa: E402
    calibrate_axes,
    detect_plot_box,
    extract_horizontal_bins,
    load_page,
    select_target_curve,
)


PANELS = [
    (11, "fhc_numu", 0.12, 0.50, 0.27, 0.47, False),
    (11, "fhc_nue", 0.50, 0.90, 0.27, 0.47, True),
    (11, "fhc_numubar", 0.12, 0.50, 0.55, 0.77, False),
    (11, "fhc_nuebar", 0.50, 0.90, 0.55, 0.77, True),
    (12, "rhc_numu", 0.12, 0.50, 0.14, 0.34, False),
    (12, "rhc_nue", 0.50, 0.90, 0.14, 0.34, True),
    (12, "rhc_numubar", 0.12, 0.50, 0.56, 0.77, False),
    (12, "rhc_nuebar", 0.50, 0.90, 0.56, 0.77, True),
]


def main() -> int:
    pdf = STUDY_ROOT / "data" / "microboone_note_1129" / "MICROBOONE-NOTE-1129-PUB.pdf"
    results = []
    for page_number, name, left, right, top, bottom, expects_censoring in PANELS:
        document, page = load_page(pdf, page_number)
        try:
            curve, candidates = select_target_curve(
                page,
                {
                    "stroke_rgb": [0.0, 0.0, 0.0],
                    "minimum_points": 100,
                    "minimum_horizontal_coverage_fraction": 0.2,
                    "ambiguity_score_fraction": 0.05,
                    "roi_fraction": {
                        "left": left,
                        "right": right,
                        "top": top,
                        "bottom": bottom,
                    },
                },
            )
            box = detect_plot_box(page, curve)
            x_transform, y_transform, audit = calibrate_axes(
                page, box, {"x_scale": "auto", "y_scale": "auto"}
            )
            bins = extract_horizontal_bins(curve, x_transform, y_transform, box)
            clipped = sum(row["is_lower_clipped"] or row["is_upper_clipped"] for row in bins)
            checks = {
                "one_curve_candidate": len(candidates) == 1,
                "curve_has_400_pdf_points": len(curve["pts"]) == 400,
                "recovered_200_horizontal_bins": len(bins) == 200,
                "x_axis_is_linear": x_transform.scale == "linear",
                "y_axis_is_log10": y_transform.scale == "log10",
                "boundary_censoring_matches_expected": (clipped > 0) == expects_censoring,
                "flattened_superscripts_resolved": audit["y_label_interpretation"]
                == "flattened_base10_with_superscript_exponent",
            }
            results.append(
                {
                    "name": name,
                    "page": page_number,
                    "status": "pass" if all(checks.values()) else "fail",
                    "checks": checks,
                    "x_bounds_GeV": sorted(
                        [float(x_transform.forward(box.x_left)), float(x_transform.forward(box.x_right))]
                    ),
                    "y_display_bounds": sorted(
                        [float(y_transform.forward(box.top)), float(y_transform.forward(box.bottom))]
                    ),
                    "y_tick_labels": [tick["label"] for tick in audit["y_ticks"]],
                    "boundary_censored_bin_count": clipped,
                }
            )
        finally:
            document.close()

    output = STUDY_ROOT.parents[1] / "outputs" / "checks" / "numi_flux_pdf_extraction"
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "pass" if all(item["status"] == "pass" for item in results) else "fail",
        "scope": "MicroBooNE NOTE-1129 vector-layout integration check; no numerical truth comparison",
        "panels": results,
    }
    (output / "microboone_note_1129_integration.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
