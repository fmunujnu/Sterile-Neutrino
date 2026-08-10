import numpy as np
import pytest

from sterile_fit.models.three_plus_one import ThreePlusOneVacuumModel
from sterile_fit.parameters import ThreePlusOneParameters


def test_probability_is_conserved_for_each_initial_flavour() -> None:
    model = ThreePlusOneVacuumModel(ThreePlusOneParameters(1.2, 0.041666666666666664, 0.018))
    energy = np.array([0.2, 0.7, 1.4])
    for initial in range(4):
        total = sum(model.probability(initial, final, energy, 0.4685) for final in range(4))
        assert np.allclose(total, 1.0, atol=1e-12)


def test_exact_appearance_amplitude_uses_mixing_matrix_elements() -> None:
    parameters = ThreePlusOneParameters(1.2, 0.25, 0.04)
    assert parameters.sin2_2theta_mue_exact == 4.0 * 0.25 * (1.0 - 0.25) * 0.04


def test_probabilities_match_published_short_baseline_formulas() -> None:
    parameters = ThreePlusOneParameters(1.2, 0.04, 0.02)
    model = ThreePlusOneVacuumModel(parameters)
    energy = np.array([0.2, 0.7, 1.4])
    baseline_km = 0.4685
    oscillatory = np.sin(1.267 * parameters.delta_m2_41_eV2 * baseline_km / energy) ** 2
    ue4_squared = parameters.sin2_theta14
    umu4_squared = (1.0 - parameters.sin2_theta14) * parameters.sin2_theta24
    expected_mue = 4.0 * ue4_squared * umu4_squared * oscillatory
    expected_ee = 1.0 - 4.0 * ue4_squared * (1.0 - ue4_squared) * oscillatory
    expected_mumu = 1.0 - 4.0 * umu4_squared * (1.0 - umu4_squared) * oscillatory
    assert model.probability(1, 0, energy, baseline_km) == pytest.approx(expected_mue)
    assert model.probability(0, 0, energy, baseline_km) == pytest.approx(expected_ee)
    assert model.probability(1, 1, energy, baseline_km) == pytest.approx(expected_mumu)
