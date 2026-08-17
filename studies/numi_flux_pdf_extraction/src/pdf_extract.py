from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

import numpy as np
import pdfplumber

from geometry import (
    AxisTransform,
    classify_vertical_clipping,
    compact_power_of_ten_candidate,
    fit_axis,
    parse_numeric_label,
    transform_to_dict,
)


@dataclass(frozen=True)
class PlotBox:
    x_left: float
    x_right: float
    top: float
    bottom: float


def _rgb_distance(first: Any, second: list[float]) -> float:
    if not isinstance(first, (tuple, list)) or len(first) != 3:
        return math.inf
    return float(np.linalg.norm(np.asarray(first, dtype=float) - np.asarray(second, dtype=float)))


def _staircase_fraction(points: list[tuple[float, float]], tolerance: float = 1e-5) -> float:
    if len(points) < 2:
        return 0.0
    axis_aligned = 0
    for first, second in zip(points, points[1:]):
        if abs(first[0] - second[0]) <= tolerance or abs(first[1] - second[1]) <= tolerance:
            axis_aligned += 1
    return axis_aligned / (len(points) - 1)


def select_target_curve(page: Any, selector: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    target_color = selector.get("stroke_rgb")
    minimum_points = int(selector.get("minimum_points", 10))
    roi_fraction = selector.get("roi_fraction")
    if roi_fraction is None:
        roi_left, roi_top, roi_right, roi_bottom = 0.0, 0.0, float(page.width), float(page.height)
    else:
        roi_left = float(roi_fraction["left"]) * float(page.width)
        roi_top = float(roi_fraction["top"]) * float(page.height)
        roi_right = float(roi_fraction["right"]) * float(page.width)
        roi_bottom = float(roi_fraction["bottom"]) * float(page.height)
        if not (0 <= roi_left < roi_right <= page.width and 0 <= roi_top < roi_bottom <= page.height):
            raise ValueError("curve-selector ROI is outside the PDF page")
    roi_width = roi_right - roi_left
    minimum_coverage = float(selector.get("minimum_horizontal_coverage_fraction", 0.0))
    candidates: list[dict[str, Any]] = []
    for index, curve in enumerate(page.curves):
        points = [(float(x), float(y)) for x, y in curve.get("pts", [])]
        if not curve.get("stroke") or curve.get("fill") or len(points) < minimum_points:
            continue
        width = float(curve.get("width", 0.0))
        if width <= 0:
            continue
        center_x = (float(curve["x0"]) + float(curve["x1"])) / 2
        center_y = (float(curve["top"]) + float(curve["bottom"])) / 2
        if not (roi_left <= center_x <= roi_right and roi_top <= center_y <= roi_bottom):
            continue
        horizontal_coverage = width / roi_width
        if horizontal_coverage < minimum_coverage:
            continue
        color_distance = 0.0 if target_color is None else _rgb_distance(curve.get("stroking_color"), target_color)
        if target_color is not None and color_distance > 0.02:
            continue
        staircase_fraction = _staircase_fraction(points)
        score = width * math.log1p(len(points)) * (0.25 + 0.75 * staircase_fraction)
        candidates.append(
            {
                "index": index,
                "score": score,
                "width": width,
                "height": float(curve.get("height", 0.0)),
                "point_count": len(points),
                "staircase_fraction": staircase_fraction,
                "color_distance": color_distance,
                "horizontal_coverage_fraction": horizontal_coverage,
                "curve": curve,
            }
        )
    if not candidates:
        raise ValueError("no vector curve satisfies the selector")
    candidates.sort(key=lambda item: item["score"], reverse=True)
    if len(candidates) > 1:
        relative_gap = (candidates[0]["score"] - candidates[1]["score"]) / candidates[0]["score"]
        required_gap = float(selector.get("ambiguity_score_fraction", 0.05))
        if relative_gap < required_gap:
            raise ValueError(
                "target curve is ambiguous: the two best path scores differ by "
                f"only {relative_gap:.3g}"
            )
    selected = candidates[0]["curve"]
    audit = [{key: value for key, value in item.items() if key != "curve"} for item in candidates]
    return selected, audit


def detect_plot_box(page: Any, curve: dict[str, Any], tolerance: float = 1.5) -> PlotBox:
    points = [(float(x), float(y)) for x, y in curve["pts"]]
    curve_left = min(point[0] for point in points)
    curve_right = max(point[0] for point in points)
    curve_bottom = max(point[1] for point in points)

    # Prefer an explicit pair of left/right vertical frame lines. ROOT can close
    # a histogram path at y=0 outside a logarithmic plot's clip rectangle, so a
    # curve bbox is not a reliable vertical bound on pages 5-6 of NOTE-1129.
    vertical_lines = [
        line
        for line in page.lines
        if abs(float(line.get("width", 0.0))) < 1e-6
        and float(line.get("height", 0.0)) >= 20.0
        and (
            abs(float(line["x0"]) - curve_left) <= tolerance
            or abs(float(line["x0"]) - curve_right) <= tolerance
        )
    ]
    paired_boxes: list[tuple[float, PlotBox]] = []
    for left_axis in vertical_lines:
        if abs(float(left_axis["x0"]) - curve_left) > tolerance:
            continue
        for right_axis in vertical_lines:
            if abs(float(right_axis["x0"]) - curve_right) > tolerance:
                continue
            if (
                abs(float(left_axis["top"]) - float(right_axis["top"])) > tolerance
                or abs(float(left_axis["bottom"]) - float(right_axis["bottom"])) > tolerance
            ):
                continue
            candidate = PlotBox(
                x_left=float(left_axis["x0"]),
                x_right=float(right_axis["x0"]),
                top=float(left_axis["top"]),
                bottom=float(left_axis["bottom"]),
            )
            contained_fraction = sum(
                candidate.top - tolerance <= point[1] <= candidate.bottom + tolerance
                for point in points
            ) / len(points)
            paired_boxes.append((contained_fraction, candidate))
    if paired_boxes:
        paired_boxes.sort(key=lambda item: (item[0], item[1].bottom - item[1].top), reverse=True)
        if paired_boxes[0][0] >= 0.5:
            return paired_boxes[0][1]

    horizontal_axes = [
        line
        for line in page.lines
        if float(line.get("width", 0.0)) > 0.8 * (curve_right - curve_left)
        and abs(float(line.get("height", 0.0))) < 1e-6
        and abs(float(line["x0"]) - curve_left) <= tolerance
        and abs(float(line["x1"]) - curve_right) <= tolerance
        and float(line["top"]) >= curve_bottom - tolerance
    ]
    if not horizontal_axes:
        raise ValueError("could not identify the x axis enclosing the target curve")
    x_axis = min(horizontal_axes, key=lambda line: float(line["top"]) - curve_bottom)
    bottom = float(x_axis["top"])
    x_left = float(x_axis["x0"])
    x_right = float(x_axis["x1"])

    vertical_axes = [
        line
        for line in page.lines
        if abs(float(line.get("width", 0.0))) < 1e-6
        and float(line.get("height", 0.0)) > 0.5 * (bottom - min(point[1] for point in points))
        and abs(float(line["x0"]) - x_left) <= tolerance
        and abs(float(line["bottom"]) - bottom) <= tolerance
    ]
    if not vertical_axes:
        raise ValueError("could not identify the y axis enclosing the target curve")
    y_axis = max(vertical_axes, key=lambda line: float(line["height"]))
    return PlotBox(x_left=x_left, x_right=x_right, top=float(y_axis["top"]), bottom=bottom)


def _numeric_words(page: Any) -> list[dict[str, Any]]:
    words = []
    for word in page.extract_words():
        if word.get("upright") is not True:
            continue
        value = parse_numeric_label(word["text"])
        if value is not None:
            compact_candidate = compact_power_of_ten_candidate(word["text"])
            characters = [
                character
                for character in page.chars
                if float(word["x0"]) - 0.2 <= (float(character["x0"]) + float(character["x1"])) / 2 <= float(word["x1"]) + 0.2
                and float(word["top"]) - 0.2 <= (float(character["top"]) + float(character["bottom"])) / 2 <= float(word["bottom"]) + 0.2
                and character.get("upright") is True
            ]
            characters.sort(key=lambda character: float(character["x0"]))
            flattened_superscript = False
            if compact_candidate is not None and len(characters) >= 3:
                base = characters[:2]
                exponent = characters[2:]
                base_size = float(np.median([character["size"] for character in base]))
                exponent_size = float(np.median([character["size"] for character in exponent]))
                flattened_superscript = (
                    exponent_size < 0.85 * base_size
                    and min(float(character["top"]) for character in exponent)
                    < min(float(character["top"]) for character in base)
                )
            words.append(
                {
                    **word,
                    "numeric_value": value,
                    "compact_power_candidate": compact_candidate,
                    "flattened_superscript_verified": flattened_superscript,
                }
            )
    return words


def _match_x_ticks(page: Any, box: PlotBox) -> list[dict[str, Any]]:
    tick_lines = [
        line
        for line in page.lines
        if abs(float(line.get("width", 0.0))) < 1e-6
        and 1.0 <= float(line.get("height", 0.0)) <= 15.0
        and box.x_left - 1 <= float(line["x0"]) <= box.x_right + 1
        and (abs(float(line["top"]) - box.bottom) <= 1 or abs(float(line["bottom"]) - box.bottom) <= 1)
    ]
    words = [
        word
        for word in _numeric_words(page)
        if box.bottom <= float(word["top"]) <= box.bottom + 18
        and box.x_left - 8 <= (float(word["x0"]) + float(word["x1"])) / 2 <= box.x_right + 8
    ]
    matches = []
    for word in words:
        center = (float(word["x0"]) + float(word["x1"])) / 2
        nearby = [line for line in tick_lines if abs(float(line["x0"]) - center) <= 5]
        maximum_length = max((float(line["height"]) for line in nearby), default=0.0)
        major_ticks = [line for line in nearby if float(line["height"]) >= maximum_length - 0.05]
        nearest = min(major_ticks, key=lambda line: abs(float(line["x0"]) - center), default=None)
        if nearest is not None:
            matches.append(
                {
                    "page_coordinate": float(nearest["x0"]),
                    "value": float(word["numeric_value"]),
                    "compact_power_candidate": word["compact_power_candidate"],
                    "flattened_superscript_verified": word["flattened_superscript_verified"],
                    "label": word["text"],
                }
            )
    unique = {(item["page_coordinate"], item["label"]): item for item in matches}
    return sorted(unique.values(), key=lambda item: item["page_coordinate"])


def _match_y_ticks(page: Any, box: PlotBox) -> list[dict[str, Any]]:
    tick_lines = [
        line
        for line in page.lines
        if abs(float(line.get("height", 0.0))) < 1e-6
        and 1.0 <= float(line.get("width", 0.0)) <= 15.0
        and abs(float(line["x0"]) - box.x_left) <= 1
        and box.top - 1 <= float(line["top"]) <= box.bottom + 1
    ]
    words = [
        word
        for word in _numeric_words(page)
        if box.x_left - 35 <= float(word["x0"]) and float(word["x1"]) <= box.x_left + 1
        and box.top - 6 <= (float(word["top"]) + float(word["bottom"])) / 2 <= box.bottom + 6
    ]
    matches = []
    for word in words:
        center = (float(word["top"]) + float(word["bottom"])) / 2
        nearby = [line for line in tick_lines if abs(float(line["top"]) - center) <= 5]
        maximum_length = max((float(line["width"]) for line in nearby), default=0.0)
        major_ticks = [line for line in nearby if float(line["width"]) >= maximum_length - 0.05]
        nearest = min(major_ticks, key=lambda line: abs(float(line["top"]) - center), default=None)
        if nearest is not None:
            matches.append(
                {
                    "page_coordinate": float(nearest["top"]),
                    "value": float(word["numeric_value"]),
                    "compact_power_candidate": word["compact_power_candidate"],
                    "flattened_superscript_verified": word["flattened_superscript_verified"],
                    "label": word["text"],
                }
            )
    unique = {(item["page_coordinate"], item["label"]): item for item in matches}
    initially_grouped = sorted(unique.values(), key=lambda item: item["page_coordinate"])
    grouped_by_coordinate: dict[float, list[dict[str, Any]]] = {}
    for item in initially_grouped:
        grouped_by_coordinate.setdefault(item["page_coordinate"], []).append(item)

    reconstructed: list[dict[str, Any]] = []
    allowed_characters = set("0123456789−–-+")
    for coordinate, items in grouped_by_coordinate.items():
        characters = [
            character
            for character in page.chars
            if character.get("upright") is True
            and box.x_left - 35 <= float(character["x0"])
            and float(character["x1"]) <= box.x_left + 1
            and abs((float(character["top"]) + float(character["bottom"])) / 2 - coordinate) <= 7
            and character.get("text") in allowed_characters
        ]
        characters.sort(key=lambda character: float(character["x0"]))
        character_text = "".join(character["text"] for character in characters)
        character_value = parse_numeric_label(character_text)
        if character_value is None:
            reconstructed.extend(items)
            continue
        compact_candidate = compact_power_of_ten_candidate(character_text)
        superscript_verified = False
        if compact_candidate is not None and len(characters) >= 3:
            base_size = float(np.median([character["size"] for character in characters[:2]]))
            exponent_size = float(np.median([character["size"] for character in characters[2:]]))
            superscript_verified = exponent_size < 0.85 * base_size
        reconstructed.append(
            {
                "page_coordinate": coordinate,
                "value": float(character_value),
                "compact_power_candidate": compact_candidate,
                "flattened_superscript_verified": superscript_verified,
                "label": character_text,
            }
        )
    return sorted(reconstructed, key=lambda item: item["page_coordinate"])


def _fit_tick_candidates(ticks: list[dict[str, Any]], requested_scale: str) -> tuple[AxisTransform, str]:
    coordinates = [tick["page_coordinate"] for tick in ticks]
    interpretations: list[tuple[str, list[float]]] = [
        ("ordinary_numeric_labels", [tick["value"] for tick in ticks])
    ]
    compact_verified = all(
        tick["compact_power_candidate"] is not None and tick["flattened_superscript_verified"]
        for tick in ticks
    )
    if compact_verified:
        interpretations.append(
            (
                "flattened_base10_with_superscript_exponent",
                [float(tick["compact_power_candidate"]) for tick in ticks],
            )
        )
    successful: list[tuple[AxisTransform, str]] = []
    for name, values in interpretations:
        try:
            successful.append((fit_axis(coordinates, values, requested_scale), name))
        except ValueError:
            continue
    if not successful:
        raise ValueError("no tick-label interpretation produced a valid axis calibration")
    successful.sort(key=lambda item: (item[0].rms_residual, item[0].maximum_residual))
    if len(successful) > 1 and math.isclose(
        successful[0][0].rms_residual,
        successful[1][0].rms_residual,
        rel_tol=0.05,
        abs_tol=1e-8,
    ):
        if compact_verified:
            for fit, name in successful:
                if name == "flattened_base10_with_superscript_exponent":
                    return fit, name
        raise ValueError("ordinary and flattened-superscript tick interpretations are ambiguous")
    return successful[0]


def calibrate_axes(page: Any, box: PlotBox, axis_config: dict[str, str]) -> tuple[AxisTransform, AxisTransform, dict[str, Any]]:
    x_ticks = _match_x_ticks(page, box)
    y_ticks = _match_y_ticks(page, box)
    x_transform, x_label_interpretation = _fit_tick_candidates(
        x_ticks, axis_config.get("x_scale", "auto")
    )
    y_transform, y_label_interpretation = _fit_tick_candidates(
        y_ticks, axis_config.get("y_scale", "auto")
    )
    audit = {
        "plot_box_points": box.__dict__,
        "x_ticks": x_ticks,
        "y_ticks": y_ticks,
        "x_label_interpretation": x_label_interpretation,
        "y_label_interpretation": y_label_interpretation,
        "x_transform": transform_to_dict(x_transform),
        "y_transform": transform_to_dict(y_transform),
    }
    return x_transform, y_transform, audit


def extract_horizontal_bins(
    curve: dict[str, Any],
    x_transform: AxisTransform,
    y_transform: AxisTransform,
    box: PlotBox | None = None,
    tolerance: float = 1e-4,
) -> list[dict[str, float]]:
    points = [(float(x), float(y)) for x, y in curve["pts"]]
    if points[-1][0] < points[0][0]:
        points.reverse()
    bins = []
    for first, second in zip(points, points[1:]):
        if second[0] - first[0] <= tolerance or abs(second[1] - first[1]) > tolerance:
            continue
        midpoint_y = (first[1] + second[1]) / 2
        clipping = (
            classify_vertical_clipping(
                midpoint_y,
                box.top,
                box.bottom,
                max(float(curve.get("linewidth", 0.0)) / 2, 0.05),
            )
            if box is not None
            else {"is_lower_clipped": False, "is_upper_clipped": False}
        )
        bins.append(
            {
                "energy_low_GeV": float(x_transform.forward(first[0])),
                "energy_high_GeV": float(x_transform.forward(second[0])),
                "flux_plot_units": float(y_transform.forward(midpoint_y)),
                "source_page_x_low": first[0],
                "source_page_x_high": second[0],
                "source_page_y": midpoint_y,
                **clipping,
            }
        )
    if not bins:
        raise ValueError("selected path has no horizontal histogram segments")
    return bins


def load_page(pdf_path: Path, page_number_1_based: int) -> tuple[Any, Any]:
    document = pdfplumber.open(pdf_path)
    if not 1 <= page_number_1_based <= len(document.pages):
        document.close()
        raise ValueError("PDF page number is out of range")
    return document, document.pages[page_number_1_based - 1]
