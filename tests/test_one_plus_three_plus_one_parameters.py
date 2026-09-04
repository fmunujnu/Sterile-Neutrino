from math import pi

import pytest

from sterile_fit.core.one_plus_three_plus_one import OnePlusThreePlusOneParameters


def test_mass_ordering_and_derived_splitting_are_explicit() -> None:
    parameters = OnePlusThreePlusOneParameters(
        1.2, 2.3, 0.01, 0.02, 0.03, 0.04, 0.5
    )
    assert parameters.delta_m2_41_eV2 == -1.2
    assert parameters.delta_m2_51_eV2 == 2.3
    assert parameters.delta_m2_54_eV2 == pytest.approx(3.5)


def test_unembeddable_heavy_row_fragments_are_rejected() -> None:
    with pytest.raises(ValueError, match="cannot be embedded"):
        OnePlusThreePlusOneParameters(1.0, 2.0, 0.5, 0.5, 0.5, 0.5, 0.0)


def test_opposite_heavy_row_phases_can_saturate_unitary_embedding() -> None:
    parameters = OnePlusThreePlusOneParameters(
        1.0, 2.0, 0.5, 0.5, 0.5, 0.5, pi
    )
    assert parameters.unitary_embedding_margin == pytest.approx(0.0, abs=1e-15)


def test_three_neutrino_null_has_no_sterile_amplitudes() -> None:
    parameters = OnePlusThreePlusOneParameters.three_neutrino_null()
    assert parameters.electron_heavy_row_norm == 0.0
    assert parameters.muon_heavy_row_norm == 0.0
    assert parameters.sin2_2theta_mue_state4 == 0.0
    assert parameters.sin2_2theta_mue_state5 == 0.0
