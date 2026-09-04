import numpy as np

from sterile_fit.experiments.microboone.bnb.adapters.archival_reco_60_to_bnb26 import (
    bnb26_reco_aggregation,
    rebin_archival_response_to_bnb26,
)


def test_bnb26_adapter_maps_regular_bins_and_overflow() -> None:
    aggregation = bnb26_reco_aggregation()
    assert aggregation.shape == (26, 60)
    assert aggregation[0, 0] == 1.0
    assert aggregation[0, 1] == 1.0
    assert aggregation[24, 48] == 1.0
    assert aggregation[24, 49] == 1.0
    assert np.all(aggregation[25, :50] == 0.0)
    assert np.all(aggregation[25, 50:] == 1.0)


def test_reco_adapter_preserves_zero_columns_and_nonzero_column_norm() -> None:
    response = np.zeros((60, 60))
    response[3, 2] = 0.7
    response[55, 2] = 0.3
    rebinned = rebin_archival_response_to_bnb26(response)
    assert rebinned.shape == (26, 60)
    assert rebinned[1, 2] == 0.7
    assert rebinned[25, 2] == 0.3
    assert np.all(rebinned[:, 0] == 0.0)
