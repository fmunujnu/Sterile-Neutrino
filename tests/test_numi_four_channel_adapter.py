import numpy as np

from sterile_fit.experiments.microboone.numi.binning import (
    NUMI_FOUR_CHANNELS,
    numi_four_channel_published_indices,
)
from sterile_fit.experiments.microboone.numi.published_inputs import (
    PublishedNumiFourChannelInputs,
    load_numi_four_channel_inputs,
)
import pytest


def test_numi_channels_8_to_11_are_four_complete_26_bin_blocks() -> None:
    indices = numi_four_channel_published_indices()
    assert [channel.released_channel_ordinal for channel in NUMI_FOUR_CHANNELS] == [8, 9, 10, 11]
    assert len(indices) == 104
    assert indices == tuple(range(182, 286))


def test_numi_four_channel_public_inputs_and_covariance_are_aligned() -> None:
    inputs = load_numi_four_channel_inputs()
    assert inputs.observed_counts.shape == (104,)
    assert inputs.published_signal_counts.shape == (104,)
    assert inputs.systematic_covariance.shape == (104, 104)
    assert np.all(inputs.published_signal_counts >= 0.0)
    assert np.allclose(inputs.systematic_covariance, inputs.systematic_covariance.T)


def test_numi_input_contract_rejects_a_non_symmetric_selected_covariance() -> None:
    vectors = np.zeros(104)
    total = np.ones(104)
    covariance = np.eye(104)
    covariance[0, 1] = 0.5
    with pytest.raises(ValueError, match="symmetric"):
        PublishedNumiFourChannelInputs(
            observed_counts=vectors,
            published_background_counts=vectors,
            published_total_prediction_counts=total,
            observed_statistical_error_up=vectors,
            observed_statistical_error_down=vectors,
            systematic_covariance=covariance,
        )
