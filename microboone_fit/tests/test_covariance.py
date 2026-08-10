import numpy as np
import pytest

from sterile_fit.covariance import (
    combined_neyman_pearson_variance,
    load_declared_total_covariance,
    pearson_statistical_variance,
    solve_quadratic_form,
)


def test_declared_total_covariance_requires_fixed_reference_contract(tmp_path) -> None:
    path = tmp_path / "covariance.npz"
    np.savez(
        path,
        covariance=np.eye(104),
        statistical_treatment="declared fixed Gaussian covariance",
        parameter_dependence="parameter_dependent",
        reference_prediction_sha256="0" * 64,
        provenance="test",
    )
    with pytest.raises(ValueError, match="parameter-dependent covariance"):
        load_declared_total_covariance(path)


def test_quadratic_form_refuses_non_positive_definite_covariance() -> None:
    with pytest.raises(ValueError, match="not positive definite"):
        solve_quadratic_form(np.array([1.0, 0.0]), np.array([[1.0, 1.0], [1.0, 1.0]]))


def test_cnp_variance_matches_declared_formula() -> None:
    assert combined_neyman_pearson_variance(np.array([6.0]), np.array([3.0])) == pytest.approx([3.6])


def test_pearson_variance_is_the_reference_prediction() -> None:
    assert pearson_statistical_variance(np.ones(104) * 2.5) == pytest.approx(np.ones(104) * 2.5)
