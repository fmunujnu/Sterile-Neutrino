from __future__ import annotations

import numpy as np

from sterile_fit.experiments.microboone.numi import FLUX_FHC_EXPOSURE_FRACTION as FHC_EXPOSURE_FRACTION, FLUX_RHC_EXPOSURE_FRACTION as RHC_EXPOSURE_FRACTION, FLUX_SIN2_2THETA_MUE as SIN2_2THETA_MUE, FLUX_SIN2_THETA24_VALUES as SIN2_THETA24_VALUES, flux_oscillated_mode_flux as _oscillated_mode_flux, flux_parameters as _parameters


def test_paper_parameter_conversion_preserves_exact_appearance_amplitude() -> None:
    for sin2_theta24 in SIN2_THETA24_VALUES:
        parameters = _parameters(sin2_theta24)
        assert np.isclose(parameters.sin2_2theta_mue_exact, SIN2_2THETA_MUE, rtol=1e-12, atol=1e-15)


def test_horn_mode_exposure_fractions_are_normalized() -> None:
    assert FHC_EXPOSURE_FRACTION == 0.308
    assert RHC_EXPOSURE_FRACTION == 0.692
    assert np.isclose(FHC_EXPOSURE_FRACTION + RHC_EXPOSURE_FRACTION, 1.0)


def test_weighting_uses_all_four_source_flavours() -> None:
    energy = np.array([0.25, 0.75, 1.25])
    source = {
        "nue": np.array([1.0, 2.0, 3.0]),
        "numu": np.array([10.0, 20.0, 30.0]),
        "nuebar": np.array([4.0, 5.0, 6.0]),
        "numubar": np.array([40.0, 50.0, 60.0]),
    }
    output, probabilities = _oscillated_mode_flux(source, energy, _parameters(0.018))
    assert np.allclose(
        output["nue"],
        source["nue"] * probabilities["nue_to_nue"]
        + source["numu"] * probabilities["numu_to_nue"],
    )
    assert np.allclose(
        output["numubar"],
        source["numubar"] * probabilities["numubar_to_numubar"]
        + source["nuebar"] * probabilities["nuebar_to_numubar"],
    )
