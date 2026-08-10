"""Construction of the strict, template-backed BNB four-channel workflow."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .covariance import DeclaredTotalCovariance, load_declared_total_covariance, prediction_sha256
from .likelihood import PredictionScaledGaussianLikelihood
from .parameters import ThreePlusOneParameters
from .prediction import BnbFourChannelPredictor
from .published_inputs import PublishedBnbFourChannelInputs, load_bnb_four_channel_inputs
from .templates import BnbFourChannelOscillationTemplates


@dataclass(frozen=True, slots=True)
class StrictBnbWorkflow:
    """A BNB workflow with declared templates and declared total covariance."""

    inputs: PublishedBnbFourChannelInputs
    predictor: BnbFourChannelPredictor
    likelihood: PredictionScaledGaussianLikelihood
    statistical_treatment: str
    covariance_parameter_dependence: str


def build_strict_bnb_workflow(
    template_path: Path,
    total_covariance_path: Path,
    reference_parameters: ThreePlusOneParameters,
    baseline_km: float,
) -> StrictBnbWorkflow:
    """Build a profile-ready workflow and refuse unvalidated physical inputs."""
    inputs = load_bnb_four_channel_inputs()
    templates = BnbFourChannelOscillationTemplates.from_directory(template_path)
    predictor = BnbFourChannelPredictor(templates, baseline_km=baseline_km)
    predictor.validate_published_reference(reference_parameters, inputs.published_total_prediction_counts)
    total_covariance: DeclaredTotalCovariance = load_declared_total_covariance(total_covariance_path)
    if total_covariance.reference_prediction_sha256 != prediction_sha256(inputs.published_total_prediction_counts):
        raise ValueError("total covariance was not constructed at the published reference prediction used by this workflow")
    return StrictBnbWorkflow(
        inputs=inputs,
        predictor=predictor,
        likelihood=PredictionScaledGaussianLikelihood(
            inputs.observed_counts,
            inputs.published_total_prediction_counts,
            inputs.systematic_covariance,
        ),
        statistical_treatment=total_covariance.statistical_treatment,
        covariance_parameter_dependence="prediction_scaled_fractional_systematics_with_current_Pearson_statistics",
    )
