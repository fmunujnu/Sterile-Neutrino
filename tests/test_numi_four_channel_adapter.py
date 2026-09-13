import numpy as np
from pathlib import Path

from sterile_fit.core.three_plus_one import ThreePlusOneParameters, ThreePlusOneVacuumModel
from sterile_fit.experiments.microboone.numi import (
    NumiEnergyBaselineDistribution, build_diagnostic_numi_workflow,
    build_energy_baseline_numi_workflow,
)
from sterile_fit.experiments.microboone.public_data import NUMI_FOUR_CHANNELS, numi_four_channel_published_indices
from sterile_fit.experiments.microboone.public_data import PublishedNumiFourChannelInputs, load_numi_four_channel_inputs
import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_numi_channels_8_to_11_are_four_complete_26_bin_blocks() -> None:
    indices = numi_four_channel_published_indices()
    assert [channel.released_channel_ordinal for channel in NUMI_FOUR_CHANNELS] == [8, 9, 10, 11]
    assert len(indices) == 104
    assert indices == tuple(range(182, 286))


def test_numi_four_channel_public_inputs_and_covariance_are_aligned() -> None:
    inputs = load_numi_four_channel_inputs()
    assert inputs.observed_counts.shape == (104,)
    assert inputs.published_signal_counts.shape == (104,)
    assert inputs.systematic_covariance.shape == (104, 104)
    assert np.all(inputs.published_signal_counts >= 0.0)
    assert np.allclose(inputs.systematic_covariance, inputs.systematic_covariance.T)


def test_numi_input_contract_rejects_a_non_symmetric_selected_covariance() -> None:
    vectors = np.zeros(104)
    total = np.ones(104)
    covariance = np.eye(104)
    covariance[0, 1] = 0.5
    with pytest.raises(ValueError, match="symmetric"):
        PublishedNumiFourChannelInputs(
            observed_counts=vectors,
            published_background_counts=vectors,
            published_total_prediction_counts=total,
            observed_statistical_error_up=vectors,
            observed_statistical_error_down=vectors,
            systematic_covariance=covariance,
        )


def test_numi_energy_baseline_average_preserves_null_closure_and_changes_oscillation() -> None:
    kernel = ROOT / "data/experiments/microboone/numi/reweighting"
    psi = ROOT / "data/experiments/microboone/numi/derived/public_dk2nu_energy_baseline/psi_exposure_weighted_four_flavours.csv"
    null = ThreePlusOneParameters(delta_m2_41_eV2=1.2, sin2_theta14=0.0, sin2_theta24=0.0)
    fixed = build_diagnostic_numi_workflow(kernel, null, 0.680)
    distributed = build_energy_baseline_numi_workflow(kernel, null, psi)
    assert np.allclose(
        fixed.predictor.predict_total_counts(null),
        distributed.predictor.predict_total_counts(null),
        rtol=1e-12,
        atol=1e-12,
    )
    oscillated = ThreePlusOneParameters(delta_m2_41_eV2=2.0, sin2_theta14=0.03, sin2_theta24=0.08)
    assert not np.allclose(
        fixed.predictor.predict_total_counts(oscillated),
        distributed.predictor.predict_total_counts(oscillated),
    )


def test_cached_three_plus_one_baseline_average_matches_direct_core_evaluation() -> None:
    psi = ROOT / "data/experiments/microboone/numi/derived/public_dk2nu_energy_baseline/psi_exposure_weighted_four_flavours.csv"
    distribution = NumiEnergyBaselineDistribution.from_csv(psi)
    parameters = ThreePlusOneParameters(delta_m2_41_eV2=1.7, sin2_theta14=0.04, sin2_theta24=0.09)
    model = ThreePlusOneVacuumModel(parameters)
    energy = np.array([0.125, 0.575, 1.225, 2.975])
    cached = distribution.three_plus_one_probabilities(parameters, energy)
    assert np.allclose(
        cached["numu_to_nue"],
        distribution.average_probability(model, 1, 0, energy),
        rtol=2e-14,
        atol=2e-14,
    )
    assert np.allclose(
        cached["nuebar_to_nuebar"],
        distribution.average_probability(model, 0, 0, energy, antineutrino=True),
        rtol=2e-14,
        atol=2e-14,
    )


def test_registered_numi_configuration_uses_energy_baseline_input() -> None:
    configuration = yaml.safe_load(
        (ROOT / "configs/experiments/microboone/numi/analysis.yaml").read_text(encoding="utf-8")
    )
    assert configuration["availability"] == "active_joint_approximation"
    assert configuration["include_in_joint_analysis"] is True
    assert configuration["include_as_standalone_experiment"] is False
    path = ROOT / configuration["energy_baseline_distribution"]["path"]
    assert path.is_file()
