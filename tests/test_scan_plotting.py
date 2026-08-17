import numpy as np
import pandas as pd
import pytest

from scripts.scan import _adaptive_toy_candidate_mask, _log_cell_edges


def test_log_cell_edges_use_geometric_midpoints() -> None:
    values = np.array([0.1, 1.0, 10.0])
    edges = _log_cell_edges(values)
    assert edges == pytest.approx([0.1 / np.sqrt(10.0), np.sqrt(0.1), np.sqrt(10.0), 10.0 * np.sqrt(10.0)])
    assert np.sqrt(edges[:-1] * edges[1:]) == pytest.approx(values)


def test_log_cell_edges_reject_nonpositive_or_unsorted_coordinates() -> None:
    with pytest.raises(ValueError, match="positive and increasing"):
        _log_cell_edges(np.array([0.0, 1.0]))
    with pytest.raises(ValueError, match="positive and increasing"):
        _log_cell_edges(np.array([1.0, 0.1]))


def test_adaptive_toy_mask_selects_wide_band_and_neighbours_per_mass_row() -> None:
    table = pd.DataFrame({
        "fixed_delta_m2_41_eV2": [1.0] * 5 + [2.0] * 5,
        "fixed_sin2_2theta_mue": [1e-4, 1e-3, 1e-2, 1e-1, 1.0] * 2,
        "cls_asymptotic": [1.0, 0.5, 0.2, 0.04, 0.001] + [0.8] * 5,
    })
    selected = _adaptive_toy_candidate_mask(
        table,
        x_name="fixed_sin2_2theta_mue",
        lower_analytic_cls=0.005,
        upper_analytic_cls=0.3,
        neighbour_padding=1,
    )
    assert selected.tolist() == [False, True, True, True, True] + [False] * 5


def test_adaptive_toy_mask_requires_band_to_bracket_threshold() -> None:
    table = pd.DataFrame({
        "fixed_delta_m2_41_eV2": [1.0, 1.0],
        "fixed_sin2_2theta_mue": [0.01, 0.1],
        "cls_asymptotic": [0.2, 0.01],
    })
    with pytest.raises(ValueError, match="must bracket 0.05"):
        _adaptive_toy_candidate_mask(
            table,
            x_name="fixed_sin2_2theta_mue",
            lower_analytic_cls=0.06,
            upper_analytic_cls=0.3,
            neighbour_padding=1,
        )
