"""Disabled-by-default NuMI workflow used by explicit diagnostic scripts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ....likelihood import PredictionScaledGaussianLikelihood
from ....parameters import ThreePlusOneParameters
from .event_prediction import NumiFourChannelEmpiricalKernel
from .prediction import NumiFourChannelPredictor
from .published_inputs import PublishedNumiFourChannelInputs, load_numi_four_channel_inputs


@dataclass(frozen=True, slots=True)
class DiagnosticNumiWorkflow:
    inputs: PublishedNumiFourChannelInputs
    predictor: NumiFourChannelPredictor
    likelihood: PredictionScaledGaussianLikelihood


def build_diagnostic_numi_workflow(
    kernel_directory: Path,
    reference_parameters: ThreePlusOneParameters,
    baseline_km: float,
) -> DiagnosticNumiWorkflow:
    """Build the explicit approximation without registering a production likelihood."""
    inputs = load_numi_four_channel_inputs()
    kernel = NumiFourChannelEmpiricalKernel.from_directory(kernel_directory)
    predictor = NumiFourChannelPredictor(kernel, baseline_km)
    predictor.validate_reference(reference_parameters, inputs.published_total_prediction_counts)
    return DiagnosticNumiWorkflow(
        inputs=inputs,
        predictor=predictor,
        likelihood=PredictionScaledGaussianLikelihood(
            inputs.observed_counts,
            inputs.published_total_prediction_counts,
            inputs.systematic_covariance,
        ),
    )
