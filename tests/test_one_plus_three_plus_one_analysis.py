import numpy as np

from sterile_fit.experiments.microboone.adapter import build_three_plus_one_analysis
from sterile_fit.experiments.microboone.adapter import load_analysis_selection
from sterile_fit.experiments.microboone.adapter import build_one_plus_three_plus_one_analysis
from sterile_fit.core.one_plus_three_plus_one import OnePlusThreePlusOneParameters
from sterile_fit.core.three_plus_one import ThreePlusOneParameters
from sterile_fit.paths import REPOSITORY_ROOT


def test_bnb_adapter_reuses_existing_null_prediction_and_covariance() -> None:
    selection = load_analysis_selection(
        REPOSITORY_ROOT / "configs" / "analyses" / "microboone_bnb.yaml",
        repository_root=REPOSITORY_ROOT,
    )
    reference = build_three_plus_one_analysis(selection, repository_root=REPOSITORY_ROOT)
    candidate = build_one_plus_three_plus_one_analysis(
        selection, repository_root=REPOSITORY_ROOT
    )
    reference_experiment = reference.experiments[0]
    candidate_experiment = candidate.experiments[0]
    reference_prediction = reference_experiment.predict_counts(
        ThreePlusOneParameters(1.0, 0.0, 0.0)
    )
    candidate_prediction = candidate_experiment.predict_counts(
        OnePlusThreePlusOneParameters.three_neutrino_null()
    )
    assert np.array_equal(candidate_prediction, reference_prediction)
    assert np.array_equal(
        candidate_experiment.covariance_for_prediction(candidate_prediction),
        reference_experiment.covariance_for_prediction(reference_prediction),
    )
    assert candidate_experiment.metadata["three_plus_one_core_modified"] is False


def test_bnb_adapter_reproduces_nonzero_three_plus_one_boundary() -> None:
    selection = load_analysis_selection(
        REPOSITORY_ROOT / "configs" / "analyses" / "microboone_bnb.yaml",
        repository_root=REPOSITORY_ROOT,
    )
    reference = build_three_plus_one_analysis(selection, repository_root=REPOSITORY_ROOT)
    candidate = build_one_plus_three_plus_one_analysis(
        selection, repository_root=REPOSITORY_ROOT
    )
    reference_parameters = ThreePlusOneParameters(1.2, 0.04, 0.02)
    candidate_parameters = OnePlusThreePlusOneParameters(
        1.2,
        3.0,
        reference_parameters.sin2_theta14,
        (1.0 - reference_parameters.sin2_theta14)
        * reference_parameters.sin2_theta24,
        0.0,
        0.0,
        0.0,
    )
    np.testing.assert_allclose(
        candidate.experiments[0].predict_counts(candidate_parameters),
        reference.experiments[0].predict_counts(reference_parameters),
        rtol=2e-14,
        atol=2e-14,
    )


def test_joint_adapter_preserves_208_bin_contract() -> None:
    selection = load_analysis_selection(
        REPOSITORY_ROOT / "configs" / "analyses" / "microboone_bnb_numi.yaml",
        repository_root=REPOSITORY_ROOT,
    )
    analysis = build_one_plus_three_plus_one_analysis(
        selection, repository_root=REPOSITORY_ROOT
    )
    experiment = analysis.experiments[0]
    prediction = experiment.predict_counts(
        OnePlusThreePlusOneParameters.three_neutrino_null()
    )
    assert prediction.shape == (208,)
    assert experiment.covariance_for_prediction(prediction).shape == (208, 208)
    assert experiment.metadata["covariance_includes_bnb_numi_cross_blocks"] is True
