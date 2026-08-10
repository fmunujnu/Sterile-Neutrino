import numpy as np

from sterile_fit.models.three_plus_one import ThreePlusOneVacuumModel
from sterile_fit.parameters import ThreePlusOneParameters


def test_probability_is_conserved_for_each_initial_flavour() -> None:
    model = ThreePlusOneVacuumModel(ThreePlusOneParameters(1.2, 0.041666666666666664, 0.018))
    energy = np.array([0.2, 0.7, 1.4])
    for initial in range(4):
        total = sum(model.probability(initial, final, energy, 0.541) for final in range(4))
        assert np.allclose(total, 1.0, atol=1e-12)


def test_exact_appearance_amplitude_uses_mixing_matrix_elements() -> None:
    parameters = ThreePlusOneParameters(1.2, 0.25, 0.04)
    assert parameters.sin2_2theta_mue_exact == 4.0 * 0.25 * (1.0 - 0.25) * 0.04
