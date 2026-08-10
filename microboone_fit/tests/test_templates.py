import numpy as np
import pytest

from sterile_fit.parameters import ThreePlusOneParameters
from sterile_fit.prediction import BnbFourChannelPredictor
from sterile_fit.templates import BnbFourChannelOscillationTemplates


def _templates() -> BnbFourChannelOscillationTemplates:
    response = np.zeros((104, 3))
    response[:, 1] = 1.0
    return BnbFourChannelOscillationTemplates(
        true_energy_GeV=np.array([0.2, 0.7, 1.2]),
        fixed_published_background_counts=np.full(104, 2.0),
        beam_nue_to_nue_cc_response_counts=response,
        beam_numu_to_nue_cc_response_counts=response * 0.1,
        beam_nue_to_numu_cc_response_counts=response * 0.2,
        beam_numu_to_numu_cc_response_counts=response * 3.0,
        beam_nuebar_to_nuebar_cc_response_counts=response * 0.05,
        beam_numubar_to_nuebar_cc_response_counts=response * 0.02,
        beam_nuebar_to_numubar_cc_response_counts=response * 0.03,
        beam_numubar_to_numubar_cc_response_counts=response * 0.5,
    )


def test_predictor_requires_template_to_close_on_reference() -> None:
    predictor = BnbFourChannelPredictor(_templates(), baseline_km=0.4685)
    reference = ThreePlusOneParameters(1.2, 0.0, 0.0)
    published = predictor.predict_total_counts(reference)
    predictor.validate_published_reference(reference, published)
    with pytest.raises(ValueError, match="does not reproduce"):
        predictor.validate_published_reference(reference, published + 1.0)


def test_template_rejects_negative_detector_folded_counts() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        BnbFourChannelOscillationTemplates(
            true_energy_GeV=np.array([0.2]),
            fixed_published_background_counts=np.zeros(104),
            beam_nue_to_nue_cc_response_counts=-np.ones((104, 1)),
            beam_numu_to_nue_cc_response_counts=np.zeros((104, 1)),
            beam_nue_to_numu_cc_response_counts=np.zeros((104, 1)),
            beam_numu_to_numu_cc_response_counts=np.zeros((104, 1)),
            beam_nuebar_to_nuebar_cc_response_counts=np.zeros((104, 1)),
            beam_numubar_to_nuebar_cc_response_counts=np.zeros((104, 1)),
            beam_nuebar_to_numubar_cc_response_counts=np.zeros((104, 1)),
            beam_numubar_to_numubar_cc_response_counts=np.zeros((104, 1)),
        )


def test_template_rejects_invalid_fixed_published_background_counts() -> None:
    with pytest.raises(ValueError, match="fixed_published_background_counts must be finite and non-negative"):
        BnbFourChannelOscillationTemplates(
            true_energy_GeV=np.array([0.2]),
            fixed_published_background_counts=np.full(104, -1.0),
            beam_nue_to_nue_cc_response_counts=np.zeros((104, 1)),
            beam_numu_to_nue_cc_response_counts=np.zeros((104, 1)),
            beam_nue_to_numu_cc_response_counts=np.zeros((104, 1)),
            beam_numu_to_numu_cc_response_counts=np.zeros((104, 1)),
            beam_nuebar_to_nuebar_cc_response_counts=np.zeros((104, 1)),
            beam_numubar_to_nuebar_cc_response_counts=np.zeros((104, 1)),
            beam_nuebar_to_numubar_cc_response_counts=np.zeros((104, 1)),
            beam_numubar_to_numubar_cc_response_counts=np.zeros((104, 1)),
        )


def test_visible_template_directory_round_trip(tmp_path) -> None:
    original = _templates()
    original.to_directory(tmp_path, metadata={"provenance": "test"})
    loaded = BnbFourChannelOscillationTemplates.from_directory(tmp_path)
    assert np.allclose(loaded.true_energy_GeV, original.true_energy_GeV, rtol=0.0, atol=1e-15)
    assert np.allclose(
        loaded.fixed_published_background_counts,
        original.fixed_published_background_counts,
        rtol=0.0,
        atol=1e-15,
    )
    assert np.allclose(
        loaded.beam_numu_to_nue_cc_response_counts,
        original.beam_numu_to_nue_cc_response_counts,
        rtol=0.0,
        atol=1e-15,
    )


def test_visible_template_directory_rejects_missing_csv(tmp_path) -> None:
    (tmp_path / "metadata.json").write_text(
        '{"format":"bnb_four_channel_text_templates_v1"}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="missing visible files"):
        BnbFourChannelOscillationTemplates.from_directory(tmp_path)
