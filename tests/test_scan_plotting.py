import numpy as np
import pytest

from scripts.scan import _log_cell_edges


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
