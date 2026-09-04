"""Observed-data profile checks for the two published scan coordinates."""
import numpy as np
import pytest

from sterile_fit.core.profile_three_plus_one import (
    profile_s14_s24_at_fixed_sin2_2theta_ee,
    profile_s14_s24_at_fixed_sin2_2theta_mue,
)


def profile_observation(objective, delta_m2, amplitude, mode):
    if mode == "appearance-profile":
        return profile_s14_s24_at_fixed_sin2_2theta_mue(
            objective,
            delta_m2_41_eV2=delta_m2,
            sin2_2theta_mue=amplitude,
        ).best_fit
    if mode == "electron-disappearance-profile":
        return profile_s14_s24_at_fixed_sin2_2theta_ee(
            objective,
            delta_m2_41_eV2=delta_m2,
            sin2_2theta_ee=amplitude,
        ).best_fit
    raise ValueError(f"Unsupported profile mode: {mode}")


@pytest.mark.parametrize("mode,amplitude", [
    ("appearance-profile", .08),
    ("electron-disappearance-profile", .36),
])
def test_observation_profile_respects_scanned_coordinate(mode, amplitude):
    objective = lambda p: (p.sin2_theta14-.8)**2 + (p.sin2_theta24-.3)**2
    observed = profile_observation(objective, 1.2, amplitude, mode)
    parameters = observed.parameters
    actual = (
        parameters.sin2_2theta_mue_exact
        if mode == "appearance-profile"
        else parameters.sin2_2theta_ee_exact
    )
    assert actual == pytest.approx(amplitude, abs=1e-12)
    assert parameters.delta_m2_41_eV2 == 1.2


@pytest.mark.parametrize("preferred_s14", [.1, .9])
def test_ee_keeps_both_physical_branches(preferred_s14):
    visited = []

    def objective(parameters):
        visited.append(parameters.sin2_theta14)
        return (parameters.sin2_theta14-preferred_s14)**2 + (parameters.sin2_theta24-.37)**2

    fit = profile_observation(objective, 2., .36, "electron-disappearance-profile")
    np.testing.assert_allclose(sorted(set(visited)), [.1, .9], atol=1e-12)
    assert fit.parameters.sin2_theta14 == pytest.approx(preferred_s14)
    assert fit.parameters.sin2_theta24 == pytest.approx(.37, abs=1e-7)


def test_unknown_mode_rejected():
    with pytest.raises(ValueError, match="Unsupported"):
        profile_observation(lambda p: 0., 1., .1, "unprofiled")
