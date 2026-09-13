import numpy as np
import pytest
from types import SimpleNamespace

from sterile_fit.core.calibration import GaussianHypothesis, fixed_hypothesis_chi2, prepare_fixed_hypothesis_chi2, prepare_fixed_test_statistic, toy_cls
from sterile_fit.core.three_plus_one import ThreePlusOneParameters
from sterile_fit.experiments.microboone.adapter import (
    _objective_for_toy,
    make_toy_profile_hypothesis_cache,
)


def test_toy_cls_is_seed_reproducible_and_evaluates_every_toy() -> None:
    null = GaussianHypothesis(np.array([0.0]), np.array([[1.0]]))
    tested = GaussianHypothesis(np.array([1.0]), np.array([[1.0]]))
    calls = 0

    def statistic(dataset: tuple[np.ndarray, ...]) -> float:
        nonlocal calls
        calls += 1
        return fixed_hypothesis_chi2(dataset, (tested,)) - fixed_hypothesis_chi2(
            dataset, (null,)
        )

    first = toy_cls(0.0, (null,), (tested,), statistic, number_of_toys=200, seed=7)
    assert calls == 400
    second = toy_cls(
        0.0,
        (null,),
        (tested,),
        lambda dataset: fixed_hypothesis_chi2(dataset, (tested,))
        - fixed_hypothesis_chi2(dataset, (null,)),
        number_of_toys=200,
        seed=7,
        workers=2,
        batch_size=17,
    )
    assert first.p_value_3nu == second.p_value_3nu
    assert first.p_value_4nu == second.p_value_4nu
    assert first.cls == second.cls
    assert first.test_statistics_under_3nu == pytest.approx(
        second.test_statistics_under_3nu
    )


def test_fixed_test_statistic_freezes_both_hypotheses() -> None:
    null = GaussianHypothesis(np.array([0.0]), np.array([[1.0]]))
    tested = GaussianHypothesis(np.array([2.0]), np.array([[4.0]]))
    statistic = prepare_fixed_test_statistic((null,), (tested,))
    dataset = (np.array([1.0]),)
    assert statistic(dataset) == pytest.approx(0.25 - 1.0)


def test_fixed_test_statistic_batch_matches_scalar_quadratic_forms() -> None:
    null = GaussianHypothesis(
        np.array([0.0, 1.0]), np.array([[2.0, 0.3], [0.3, 1.5]])
    )
    tested = GaussianHypothesis(
        np.array([0.4, 0.7]), np.array([[1.7, 0.2], [0.2, 1.2]])
    )
    statistic = prepare_fixed_test_statistic((null,), (tested,))
    draws = (np.array([[0.2, 0.8], [1.1, -0.3], [-0.5, 2.0]]),)
    scalar = np.array([statistic((row,)) for row in draws[0]])
    np.testing.assert_allclose(statistic.evaluate_batch(draws), scalar, rtol=2e-14, atol=2e-14)


def test_batched_fixed_toy_keeps_seeded_results_across_batch_sizes() -> None:
    null = GaussianHypothesis(np.array([0.0, 0.5]), np.array([[1.0, 0.2], [0.2, 2.0]]))
    tested = GaussianHypothesis(np.array([0.3, 0.8]), np.array([[1.2, 0.1], [0.1, 1.6]]))
    statistic = prepare_fixed_test_statistic((null,), (tested,))
    first = toy_cls(0.0, (null,), (tested,), statistic, number_of_toys=53, seed=31, batch_size=53)
    second = toy_cls(0.0, (null,), (tested,), statistic, number_of_toys=53, seed=31, batch_size=7)
    np.testing.assert_allclose(first.test_statistics_under_3nu, second.test_statistics_under_3nu, rtol=2e-14, atol=2e-14)
    np.testing.assert_allclose(first.test_statistics_under_4nu, second.test_statistics_under_4nu, rtol=2e-14, atol=2e-14)
    assert first.cls == second.cls


def test_toy_batch_size_does_not_change_multiple_component_random_streams() -> None:
    null = (
        GaussianHypothesis(np.array([0.0]), np.array([[1.0]])),
        GaussianHypothesis(np.array([2.0]), np.array([[4.0]])),
    )
    tested = (
        GaussianHypothesis(np.array([1.0]), np.array([[1.0]])),
        GaussianHypothesis(np.array([3.0]), np.array([[4.0]])),
    )

    def statistic(dataset: tuple[np.ndarray, ...]) -> float:
        return float(dataset[0][0] + dataset[1][0])

    unbatched = toy_cls(
        0.0, null, tested, statistic, number_of_toys=41, seed=19, batch_size=41
    )
    batched = toy_cls(
        0.0, null, tested, statistic, number_of_toys=41, seed=19, batch_size=7
    )
    assert unbatched.test_statistics_under_3nu == pytest.approx(
        batched.test_statistics_under_3nu
    )
    assert unbatched.test_statistics_under_4nu == pytest.approx(
        batched.test_statistics_under_4nu
    )


def test_toy_cls_uses_empirical_right_tail_with_plus_one_correction() -> None:
    null = GaussianHypothesis(np.array([0.0]), np.array([[1.0]]))
    tested = GaussianHypothesis(np.array([1.0]), np.array([[1.0]]))
    result = toy_cls(
        1.0e9,
        (null,),
        (tested,),
        lambda dataset: float(dataset[0][0]),
        number_of_toys=9,
        seed=11,
    )
    assert result.right_tail_count_under_3nu == 0
    assert result.right_tail_count_under_4nu == 0
    assert result.p_value_3nu == pytest.approx(0.1)
    assert result.p_value_4nu == pytest.approx(0.1)
    assert result.cls == pytest.approx(1.0)


def test_fixed_hypothesis_chi2_sums_registered_contributions() -> None:
    hypotheses = (
        GaussianHypothesis(np.array([1.0]), np.array([[4.0]])),
        GaussianHypothesis(np.array([2.0]), np.array([[1.0]])),
    )
    value = fixed_hypothesis_chi2((np.array([3.0]), np.array([5.0])), hypotheses)
    assert value == pytest.approx(10.0)
    prepared = prepare_fixed_hypothesis_chi2(hypotheses)
    assert prepared((np.array([3.0]), np.array([5.0]))) == pytest.approx(value)


def test_profile_toy_cache_is_exact_and_reuses_only_parameter_work() -> None:
    prediction_calls = 0

    def predict(parameters):
        nonlocal prediction_calls
        prediction_calls += 1
        return np.array([1.0 + parameters.sin2_theta14, 2.0 + parameters.sin2_theta24])

    experiment = SimpleNamespace(
        predict_counts=predict,
        covariance_for_prediction=lambda prediction: np.array(
            [[2.0 + prediction[0], 0.2], [0.2, 3.0 + prediction[1]]]
        ),
    )
    analysis = SimpleNamespace(experiments=(experiment,))
    parameters = ThreePlusOneParameters(1.2, 0.1, 0.2)
    first_data = (np.array([0.8, 2.4]),)
    second_data = (np.array([1.5, 1.7]),)
    expected = (
        _objective_for_toy(analysis, first_data)(parameters),
        _objective_for_toy(analysis, second_data)(parameters),
    )
    prediction_calls = 0
    cache = make_toy_profile_hypothesis_cache(analysis, maxsize=8)
    actual = (
        _objective_for_toy(analysis, first_data, prepared_hypothesis=cache)(parameters),
        _objective_for_toy(analysis, second_data, prepared_hypothesis=cache)(parameters),
    )
    assert actual == expected
    assert prediction_calls == 1
