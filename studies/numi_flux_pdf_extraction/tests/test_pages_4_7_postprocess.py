from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


STUDY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STUDY_ROOT))

from postprocess_pages_4_7 import (  # noqa: E402
    derive_antineutrino,
    expand_native_histogram_to_target_grid,
)


def _native(low: float, high: float, value: float) -> dict[str, float | bool]:
    return {
        "energy_low_GeV": low,
        "energy_high_GeV": high,
        "flux_plot_units": value,
        "is_lower_clipped": False,
        "is_upper_clipped": False,
    }


def test_wide_native_steps_are_expanded_without_interpolation() -> None:
    native = [_native(0.0, 2.0, 10.0), _native(2.0, 5.0, 4.0)]
    expanded = expand_native_histogram_to_target_grid(native)
    assert len(expanded) == 50
    assert [row["flux_per_POT_per_cm2_per_100MeV"] for row in expanded[:20]] == [10.0] * 20
    assert [row["flux_per_POT_per_cm2_per_100MeV"] for row in expanded[20:]] == [4.0] * 30
    assert np.allclose([row["energy_low_GeV"] for row in expanded], np.arange(0.0, 5.0, 0.1))


def test_antineutrino_is_same_grid_total_minus_neutrino() -> None:
    neutrino = expand_native_histogram_to_target_grid([_native(0.0, 5.0, 3.0)])
    total = expand_native_histogram_to_target_grid([_native(0.0, 5.0, 5.0)])
    antineutrino = derive_antineutrino(neutrino, total)
    assert len(antineutrino) == 50
    assert all(
        np.isclose(row["flux_per_POT_per_cm2_per_100MeV"], 2.0)
        for row in antineutrino
    )


def test_censoring_propagates_through_subtraction() -> None:
    neutrino = expand_native_histogram_to_target_grid([_native(0.0, 5.0, 3.0)])
    clipped_total = _native(0.0, 5.0, 5.0)
    clipped_total["is_lower_clipped"] = True
    total = expand_native_histogram_to_target_grid([clipped_total])
    antineutrino = derive_antineutrino(neutrino, total)
    assert all(row["is_lower_censored"] for row in antineutrino)
