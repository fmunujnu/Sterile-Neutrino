import numpy as np
import pytest

from sterile_fit.core.three_plus_one import ThreePlusOneVacuumModel
from sterile_fit.core.one_plus_three_plus_one import OnePlusThreePlusOneVacuumModel
from sterile_fit.core.one_plus_three_plus_one import OnePlusThreePlusOneParameters
from sterile_fit.core.three_plus_one import ThreePlusOneParameters


def test_single_isolated_state_limit_exactly_reproduces_three_plus_one() -> None:
    three_plus_one_parameters = ThreePlusOneParameters(1.2, 0.04, 0.02)
    one_plus_three_plus_one_parameters = OnePlusThreePlusOneParameters(
        1.2,
        3.0,
        three_plus_one_parameters.sin2_theta14,
        (1.0 - three_plus_one_parameters.sin2_theta14)
        * three_plus_one_parameters.sin2_theta24,
        0.0,
        0.0,
        0.0,
    )
    reference = ThreePlusOneVacuumModel(three_plus_one_parameters)
    candidate = OnePlusThreePlusOneVacuumModel(one_plus_three_plus_one_parameters)
    energy = np.array([0.2, 0.7, 1.4])
    for initial, final in ((0, 0), (0, 1), (1, 0), (1, 1)):
        for antineutrino in (False, True):
            assert candidate.probability(
                initial, final, energy, 0.4685, antineutrino=antineutrino
            ) == pytest.approx(
                reference.probability(
                    initial, final, energy, 0.4685, antineutrino=antineutrino
                ),
                abs=2e-15,
            )


def test_cp_conjugation_flips_the_short_baseline_phase() -> None:
    positive_phase = OnePlusThreePlusOneVacuumModel(
        OnePlusThreePlusOneParameters(0.8, 1.7, 0.02, 0.03, 0.04, 0.02, 0.7)
    )
    negative_phase = OnePlusThreePlusOneVacuumModel(
        OnePlusThreePlusOneParameters(0.8, 1.7, 0.02, 0.03, 0.04, 0.02, -0.7)
    )
    energy = np.array([0.3, 0.8, 1.5])
    assert positive_phase.probability(
        1, 0, energy, 0.4685, antineutrino=True
    ) == pytest.approx(
        negative_phase.probability(1, 0, energy, 0.4685), abs=2e-15
    )


def test_effective_model_rejects_unimplemented_tau_flavour() -> None:
    model = OnePlusThreePlusOneVacuumModel(
        OnePlusThreePlusOneParameters.three_neutrino_null()
    )
    with pytest.raises(ValueError, match="only e=0 and mu=1"):
        model.probability(2, 0, np.array([1.0]), 0.4685)
