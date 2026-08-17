import math

import pytest

from sterile_fit.parameters import ThreePlusOneParameters


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), float("-inf")])
def test_parameters_reject_non_finite_values(invalid: float) -> None:
    with pytest.raises(ValueError):
        ThreePlusOneParameters(invalid, 0.1, 0.1)
    with pytest.raises(ValueError):
        ThreePlusOneParameters(1.0, invalid, 0.1)


def test_parameter_meanings_are_not_ambiguous() -> None:
    parameters = ThreePlusOneParameters(1.2, 0.25, 0.04)
    assert parameters.delta_m2_41_eV2 == 1.2
    assert parameters.sin_theta14 == 0.5
    assert parameters.theta24_rad == pytest.approx(math.asin(0.2))


def test_invalid_parameter_values_fail_early() -> None:
    with pytest.raises(ValueError, match="strictly positive"):
        ThreePlusOneParameters(0.0, 0.1, 0.1)
    with pytest.raises(ValueError, match="sin2_theta14"):
        ThreePlusOneParameters(1.0, 1.01, 0.1)
