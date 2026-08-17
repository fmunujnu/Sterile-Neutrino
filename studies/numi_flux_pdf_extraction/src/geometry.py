from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class AxisTransform:
    """Map an absolute PDF coordinate to a physical axis value."""

    scale: str
    intercept: float
    slope: float
    rms_residual: float
    maximum_residual: float

    def forward(self, page_coordinate: float | np.ndarray) -> float | np.ndarray:
        transformed = self.intercept + self.slope * np.asarray(page_coordinate)
        values = 10.0**transformed if self.scale == "log10" else transformed
        return float(values) if np.ndim(values) == 0 else values

    def inverse(self, value: float | np.ndarray) -> float | np.ndarray:
        values = np.asarray(value)
        if self.scale == "log10":
            if np.any(values <= 0):
                raise ValueError("log10 axis cannot project non-positive values")
            values = np.log10(values)
        coordinates = (values - self.intercept) / self.slope
        return float(coordinates) if np.ndim(coordinates) == 0 else coordinates


def parse_numeric_label(text: str) -> float | None:
    """Parse ordinary numbers and compact PDF forms such as 10−12."""

    cleaned = (
        text.strip()
        .replace("−", "-")
        .replace("–", "-")
        .replace("×", "x")
        .replace(" ", "")
    )
    try:
        return float(cleaned)
    except ValueError:
        pass
    match = re.fullmatch(r"10(?:\^)?(?:\{)?([+-]?\d+)(?:\})?", cleaned)
    if match:
        return 10.0 ** int(match.group(1))
    return None


def compact_power_of_ten_candidate(text: str) -> float | None:
    """Return the TeX-flattened interpretation of labels such as ``1011``.

    pdfplumber can flatten a base 10 and its superscript exponent into one word.
    This function deliberately returns only an alternative candidate. The axis
    fitter must compare the whole candidate sequence before selecting it, since
    an ordinary decimal such as 1011 must not be reinterpreted in isolation.
    """

    cleaned = text.strip().replace(" ", "")
    match = re.fullmatch(r"10(\d{1,2})", cleaned)
    if not match:
        return None
    return 10.0 ** int(match.group(1))


def fit_axis(
    coordinates: Iterable[float],
    values: Iterable[float],
    requested_scale: str = "auto",
    maximum_normalized_residual: float = 1e-3,
) -> AxisTransform:
    coordinates_array = np.asarray(list(coordinates), dtype=float)
    values_array = np.asarray(list(values), dtype=float)
    if coordinates_array.size < 3 or coordinates_array.size != values_array.size:
        raise ValueError("axis calibration requires at least three coordinate/value pairs")
    if np.unique(coordinates_array).size < 3 or np.unique(values_array).size < 3:
        raise ValueError("axis calibration ticks are not distinct")

    candidates = [requested_scale] if requested_scale != "auto" else ["linear", "log10"]
    fits: list[AxisTransform] = []
    for scale in candidates:
        if scale not in {"linear", "log10"}:
            raise ValueError(f"unsupported axis scale: {scale}")
        if scale == "log10" and np.any(values_array <= 0):
            continue
        target = np.log10(values_array) if scale == "log10" else values_array
        slope, intercept = np.polyfit(coordinates_array, target, 1)
        residuals = target - (intercept + slope * coordinates_array)
        target_span = float(np.ptp(target))
        normalization = target_span if target_span > 0 else 1.0
        fits.append(
            AxisTransform(
                scale=scale,
                intercept=float(intercept),
                slope=float(slope),
                rms_residual=float(np.sqrt(np.mean(residuals**2)) / normalization),
                maximum_residual=float(np.max(np.abs(residuals)) / normalization),
            )
        )
    if not fits:
        raise ValueError("no valid axis-scale hypothesis")
    fits.sort(key=lambda fit: (fit.rms_residual, fit.maximum_residual))
    if len(fits) > 1 and math.isclose(
        fits[0].rms_residual, fits[1].rms_residual, rel_tol=0.05, abs_tol=1e-8
    ):
        raise ValueError("linear/logarithmic axis classification is ambiguous")
    if fits[0].maximum_residual > maximum_normalized_residual:
        raise ValueError(
            "axis calibration residual is too large: "
            f"{fits[0].maximum_residual:.6g} > {maximum_normalized_residual:.6g}"
        )
    return fits[0]


def transform_to_dict(transform: AxisTransform) -> dict[str, float | str]:
    return {
        "scale": transform.scale,
        "intercept": transform.intercept,
        "slope": transform.slope,
        "rms_residual": transform.rms_residual,
        "maximum_residual": transform.maximum_residual,
    }


def classify_vertical_clipping(
    page_y: float,
    plot_top: float,
    plot_bottom: float,
    tolerance_points: float,
) -> dict[str, bool]:
    """Mark a displayed level that lies on a plot boundary as censored."""

    return {
        "is_lower_clipped": page_y >= plot_bottom - tolerance_points,
        "is_upper_clipped": page_y <= plot_top + tolerance_points,
    }
