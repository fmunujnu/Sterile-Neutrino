from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

STUDY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STUDY_ROOT / "src"))

from geometry import (  # noqa: E402
    classify_vertical_clipping,
    compact_power_of_ten_candidate,
    fit_axis,
    parse_numeric_label,
)


def test_linear_axis_is_selected_and_invertible() -> None:
    coordinates = [10.0, 20.0, 30.0, 40.0]
    values = [0.0, 2.0, 4.0, 6.0]
    transform = fit_axis(coordinates, values)
    assert transform.scale == "linear"
    assert np.allclose(transform.forward(coordinates), values)
    assert np.allclose(transform.inverse(values), coordinates)


def test_log_axis_is_selected_and_invertible() -> None:
    coordinates = [100.0, 80.0, 60.0, 40.0]
    values = [1e-6, 1e-5, 1e-4, 1e-3]
    transform = fit_axis(coordinates, values)
    assert transform.scale == "log10"
    assert np.allclose(transform.forward(coordinates), values)
    assert np.allclose(transform.inverse(values), coordinates)


def test_compact_power_of_ten_tick_is_parsed() -> None:
    assert parse_numeric_label("10−12") == 1e-12
    assert parse_numeric_label("10^-9") == 1e-9


def test_flattened_superscript_is_only_an_alternative_candidate() -> None:
    assert parse_numeric_label("1011") == 1011.0
    assert compact_power_of_ten_candidate("1011") == 1e11
    assert compact_power_of_ten_candidate("109") == 1e9
    assert compact_power_of_ten_candidate("10") is None


def test_bad_axis_fit_is_rejected() -> None:
    coordinates = [0.0, 1.0, 2.0, 3.0]
    values = [0.0, 1.0, 100.0, 101.0]
    try:
        fit_axis(coordinates, values)
    except ValueError as error:
        assert "residual" in str(error)
    else:
        raise AssertionError("a badly calibrated axis must be rejected")


def test_plot_boundary_is_reported_as_censored_not_silent_value() -> None:
    assert classify_vertical_clipping(99.9, 20.0, 100.0, 0.2)["is_lower_clipped"]
    assert classify_vertical_clipping(101.0, 20.0, 100.0, 0.2)["is_lower_clipped"]
    assert classify_vertical_clipping(20.1, 20.0, 100.0, 0.2)["is_upper_clipped"]
    assert classify_vertical_clipping(19.0, 20.0, 100.0, 0.2)["is_upper_clipped"]
    assert not any(classify_vertical_clipping(60.0, 20.0, 100.0, 0.2).values())
