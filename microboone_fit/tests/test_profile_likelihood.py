import pytest

from sterile_fit.fitting import (
    profile_appearance_amplitude_grid,
    profile_s14_s24_at_fixed_sin2_2theta_mue,
    profile_grid,
    profile_three_plus_one,
)
from sterile_fit.parameters import ThreePlusOneParameters


def _objective(parameters: ThreePlusOneParameters) -> float:
    return (
        (parameters.delta_m2_41_eV2 - 1.2) ** 2
        + (parameters.sin2_theta14 - 0.04) ** 2
        + (parameters.sin2_theta24 - 0.02) ** 2
    )


def test_profile_minimizes_only_non_fixed_parameters() -> None:
    result = profile_three_plus_one(
        _objective,
        {"delta_m2_41_eV2": 1.2, "sin2_theta14": 0.04},
        seed=1,
    )
    assert result.best_fit.parameters.delta_m2_41_eV2 == 1.2
    assert result.best_fit.parameters.sin2_theta14 == 0.04
    assert result.best_fit.parameters.sin2_theta24 == pytest.approx(0.02, abs=1e-5)
    assert result.best_fit.chi2 < 1e-10


def test_profile_grid_keeps_each_scan_coordinate_fixed() -> None:
    results = profile_grid(_objective, {"delta_m2_41_eV2": [0.8, 1.2]}, seed=2)
    assert [item.best_fit.parameters.delta_m2_41_eV2 for item in results] == [0.8, 1.2]


def test_fixed_appearance_amplitude_profiles_only_the_physical_curve() -> None:
    target_amplitude = 0.04
    result = profile_s14_s24_at_fixed_sin2_2theta_mue(
        _objective,
        delta_m2_41_eV2=1.2,
        sin2_2theta_mue=target_amplitude,
        seed=3,
    )
    point = result.best_fit.parameters
    assert point.sin2_2theta_mue_exact == pytest.approx(target_amplitude, abs=1e-12)
    assert 0.0 <= point.sin2_theta14 <= 1.0
    assert 0.0 <= point.sin2_theta24 <= 1.0


def test_appearance_amplitude_grid_retains_both_scan_coordinates() -> None:
    results = profile_appearance_amplitude_grid(_objective, [0.8, 1.2], [0.01], seed=4)
    assert [item.delta_m2_41_eV2 for item in results] == [0.8, 1.2]
    assert all(item.sin2_2theta_mue == 0.01 for item in results)
