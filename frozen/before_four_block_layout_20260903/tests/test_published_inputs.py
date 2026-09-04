import numpy as np

from sterile_fit.experiments.microboone.bnb.published_inputs import load_bnb_four_channel_inputs, read_full_systematic_covariance


def test_public_input_selection_is_explicit_and_well_formed() -> None:
    inputs = load_bnb_four_channel_inputs()
    assert inputs.observed_counts.shape == (104,)
    assert inputs.systematic_covariance.shape == (104, 104)
    assert np.allclose(inputs.systematic_covariance, inputs.systematic_covariance.T)
    assert np.all(inputs.published_signal_counts >= 0.0)


def test_reader_preserves_the_full_released_covariance_before_selection() -> None:
    covariance = read_full_systematic_covariance()
    assert covariance.shape == (364, 364)
