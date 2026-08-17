import numpy as np
import pytest

from sterile_fit.statistics import GaussianHypothesis, asymptotic_cls


def test_common_covariance_has_known_test_statistic_moments() -> None:
    null = GaussianHypothesis(np.array([0.0]), np.array([[1.0]]))
    tested = GaussianHypothesis(np.array([2.0]), np.array([[1.0]]))
    result = asymptotic_cls(0.0, [(null, tested)])

    # For common covariance, D=(mu4-mu3)^T C^-1 (mu4-mu3)=4 and
    # chi2_4-chi2_3 is N(+D,4D) under 3nu and N(-D,4D) under 4nu.
    assert result.mean_under_3nu == pytest.approx(4.0)
    assert result.mean_under_4nu == pytest.approx(-4.0)
    assert result.standard_deviation_under_3nu == pytest.approx(4.0)
    assert result.standard_deviation_under_4nu == pytest.approx(4.0)
    assert result.p_value_4nu == pytest.approx(0.15865525393145707)
    assert result.p_value_3nu == pytest.approx(0.8413447460685429)
    assert result.cls == pytest.approx(0.18857341734506025)


def test_asymptotic_cls_rejects_non_positive_definite_covariance() -> None:
    with pytest.raises(np.linalg.LinAlgError):
        GaussianHypothesis(np.array([0.0]), np.array([[0.0]]))
