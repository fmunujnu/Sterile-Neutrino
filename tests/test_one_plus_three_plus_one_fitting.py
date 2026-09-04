import pytest

from sterile_fit.core.profile_one_plus_three_plus_one import profile_one_plus_three_plus_one
from sterile_fit.core.one_plus_three_plus_one import OnePlusThreePlusOneParameters


def test_profile_keeps_named_true_parameters_fixed() -> None:
    def objective(parameters: OnePlusThreePlusOneParameters) -> float:
        return (parameters.abs_Ue5_squared - 0.03) ** 2

    result = profile_one_plus_three_plus_one(
        objective,
        {
            "delta_m2_41_absolute_eV2": 1.2,
            "delta_m2_51_eV2": 2.0,
            "abs_Ue4_squared": 0.01,
            "abs_Umu4_squared": 0.02,
            "abs_Umu5_squared": 0.02,
            "cp_phase_mue_rad": 3.0,
        },
        seed=7,
        maxiter=100,
        popsize=8,
    )
    point = result.best_fit.parameters
    assert point.delta_m2_41_absolute_eV2 == 1.2
    assert point.delta_m2_51_eV2 == 2.0
    assert point.abs_Ue4_squared == 0.01
    assert point.abs_Ue5_squared == pytest.approx(0.03, abs=1e-7)
    assert point.unitary_embedding_margin >= -1e-12


def test_profile_rejects_ambiguous_or_nonphysical_fixed_names() -> None:
    with pytest.raises(ValueError, match=r"unknown 1\+3\+1"):
        profile_one_plus_three_plus_one(
            lambda parameters: 0.0,
            {"theta14": 0.1},
        )
