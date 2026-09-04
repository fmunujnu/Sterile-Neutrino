import numpy as np
import pytest

from sterile_fit.likelihood import PredictionScaledGaussianLikelihood
from sterile_fit.covariance import solve_quadratic_form


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


def test_prediction_scaled_likelihood_supports_joint_arbitrary_bin_count() -> None:
    observed = np.array([9.0, 11.0, 8.0])
    reference = np.array([10.0, 10.0, 10.0])
    systematic = np.array([[2.0, 0.5, 0.1], [0.5, 3.0, 0.2], [0.1, 0.2, 1.5]])
    likelihood = PredictionScaledGaussianLikelihood(observed, reference, systematic)
    assert likelihood.covariance_for_prediction(reference) == pytest.approx(
        systematic + np.diag(reference)
    )


def test_optimized_triangular_quadratic_form_matches_direct_inverse_definition() -> None:
    covariance = np.array([[4.0, 0.7, 0.2], [0.7, 3.0, 0.4], [0.2, 0.4, 2.0]])
    residual = np.array([1.5, -0.4, 0.8])
    expected = float(residual @ np.linalg.inv(covariance) @ residual)
    assert solve_quadratic_form(residual, covariance) == pytest.approx(expected, rel=1e-13)
