import json

import numpy as np
import pytest

from sterile_fit.covariance import (
    combined_neyman_pearson_variance,
    load_declared_total_covariance,
    pearson_statistical_variance,
    solve_quadratic_form,
)


def test_declared_total_covariance_requires_fixed_reference_contract(tmp_path) -> None:
    path = tmp_path / "covariance.csv"
    np.savetxt(path, np.eye(104), delimiter=",")
    path.with_suffix(".metadata.json").write_text(json.dumps({
        "shape": [104, 104],
        "row_order": "nue_cc_fc,nue_cc_pc,numu_cc_fc,numu_cc_pc; 26 reconstructed bins each",
        "column_order": "same as row_order",
        "statistical_treatment": "declared fixed Gaussian covariance",
        "parameter_dependence": "parameter_dependent",
        "reference_prediction_sha256": "0" * 64,
        "provenance": "test",
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="parameter-dependent covariance"):
        load_declared_total_covariance(path)


def test_declared_total_covariance_rejects_binary_archive(tmp_path) -> None:
    with pytest.raises(ValueError, match="visible .csv"):
        load_declared_total_covariance(tmp_path / "covariance.npz")


def test_declared_total_covariance_rejects_wrong_declared_order(tmp_path) -> None:
    path = tmp_path / "covariance.csv"
    np.savetxt(path, np.eye(104), delimiter=",")
    path.with_suffix(".metadata.json").write_text(json.dumps({
        "shape": [104, 104],
        "row_order": "wrong order",
        "column_order": "same as row_order",
        "statistical_treatment": "declared fixed Gaussian covariance",
        "parameter_dependence": "fixed_at_reference",
        "reference_prediction_sha256": "0" * 64,
        "provenance": "test",
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="row_order"):
        load_declared_total_covariance(path)


def test_quadratic_form_refuses_non_positive_definite_covariance() -> None:
    with pytest.raises(ValueError, match="not positive definite"):
        solve_quadratic_form(np.array([1.0, 0.0]), np.array([[1.0, 1.0], [1.0, 1.0]]))


def test_cnp_variance_matches_declared_formula() -> None:
    assert combined_neyman_pearson_variance(np.array([6.0]), np.array([3.0])) == pytest.approx([3.6])


def test_pearson_variance_is_the_reference_prediction() -> None:
    assert pearson_statistical_variance(np.ones(104) * 2.5) == pytest.approx(np.ones(104) * 2.5)
