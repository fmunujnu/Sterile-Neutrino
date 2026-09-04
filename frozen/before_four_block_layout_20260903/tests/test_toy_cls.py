import numpy as np
import pytest

from sterile_fit.statistics import (
    GaussianHypothesis,
    fixed_hypothesis_chi2,
    prepare_fixed_hypothesis_chi2,
    toy_cls,
)


def test_toy_cls_is_seed_reproducible_and_profiles_every_toy() -> None:
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
