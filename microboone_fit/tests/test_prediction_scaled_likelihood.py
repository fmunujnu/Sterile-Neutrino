import numpy as np
import pytest

from sterile_fit.likelihood import PredictionScaledGaussianLikelihood


def test_prediction_scaled_covariance_matches_reference_definition() -> None:
    observed = np.full(104, 9.0)
    reference = np.full(104, 10.0)
    systematic = np.eye(104) * 4.0
    likelihood = PredictionScaledGaussianLikelihood(observed, reference, systematic)
    assert likelihood.covariance_for_prediction(reference) == pytest.approx(systematic + np.diag(reference))


def test_prediction_scaled_covariance_preserves_fractional_systematics() -> None:
    reference = np.full(104, 10.0)
    systematic = np.eye(104) * 4.0
    likelihood = PredictionScaledGaussianLikelihood(reference, reference, systematic)
    doubled = np.full(104, 20.0)
    covariance = likelihood.covariance_for_prediction(doubled)
    # Absolute systematic variances scale by 2^2; Pearson statistics scale by 2.
    assert np.diag(covariance) == pytest.approx(np.full(104, 4.0 * 4.0 + 20.0))


def test_prediction_scaled_likelihood_rejects_zero_reference_bins() -> None:
    with pytest.raises(ValueError, match="strictly positive"):
        PredictionScaledGaussianLikelihood(np.ones(104), np.zeros(104), np.eye(104))
