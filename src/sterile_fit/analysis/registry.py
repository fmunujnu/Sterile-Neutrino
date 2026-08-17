"""Factories connecting experiment IDs to their private workflows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping

import numpy as np
from numpy.typing import NDArray
import yaml

from ..experiments.microboone.bnb.workflow import StrictBnbWorkflow, build_strict_bnb_workflow
from ..experiments.microboone.joint_bnb_numi import build_joint_microboone_bnb_numi_workflow
from ..experiments.microboone.numi.workflow import build_diagnostic_numi_workflow
from ..parameters import ThreePlusOneParameters
from .combination import ChiSquareContribution, CombinedChiSquare
from .selection import AnalysisSelection


MICROBOONE_BNB_FOUR_CHANNEL = "microboone.bnb.four_channel"
MICROBOONE_BNB_NUMI_JOINT_FOUR_CHANNEL = "microboone.bnb_numi.joint_four_channel"


@dataclass(frozen=True, slots=True)
class BuiltExperiment:
    """One selected workflow plus scan metadata."""

    experiment_id: str
    status: str
    correlation_group: str
    configuration: Path
    evaluate: Callable[[ThreePlusOneParameters], float]
    predict_counts: Callable[[ThreePlusOneParameters], NDArray[np.float64]]
    observed_counts: NDArray[np.float64]
    covariance_for_prediction: Callable[[NDArray[np.float64]], NDArray[np.float64]]
    metadata: Mapping[str, object]

    def chi2(self, parameters: ThreePlusOneParameters) -> float:
        return float(self.evaluate(parameters))


@dataclass(frozen=True, slots=True)
class BuiltAnalysis:
    analysis_name: str
    experiments: tuple[BuiltExperiment, ...]
    objective: CombinedChiSquare[ThreePlusOneParameters]


def _repository_path(repository_root: Path, value: str, *, label: str) -> Path:
    path = (repository_root / value).resolve()
    if repository_root.resolve() not in path.parents:
        raise ValueError(f"{label} must stay inside the repository")
    return path


def _build_microboone_bnb(
    configuration: Path,
    repository_root: Path,
    *,
    overrides: Mapping[str, Path] | None,
) -> tuple[StrictBnbWorkflow, ThreePlusOneParameters, float, Path, Path]:
    document = yaml.safe_load(configuration.read_text(encoding="utf-8"))
    reference = ThreePlusOneParameters(
        **{name: float(value) for name, value in document["reference_parameters"].items()}
    )
    baseline_km = float(document["baseline_km"])
    paths = document["analysis_inputs"]
    override_paths = dict(overrides or {})
    kernel = override_paths.get("kernel") or _repository_path(repository_root, paths["kernel"], label="kernel")
    covariance = override_paths.get("covariance") or _repository_path(
        repository_root, paths["covariance"], label="covariance"
    )
    return (
        build_strict_bnb_workflow(kernel, covariance, reference, baseline_km),
        reference,
        baseline_km,
        kernel,
        covariance,
    )


def build_three_plus_one_analysis(
    selection: AnalysisSelection,
    *,
    repository_root: Path,
    bnb_overrides: Mapping[str, Path] | None = None,
) -> BuiltAnalysis:
    """Build exactly the selected experiment likelihoods for a 3+1 scan."""
    built: list[BuiltExperiment] = []
    for item in selection.included:
        if item.experiment_id == MICROBOONE_BNB_FOUR_CHANNEL:
            workflow, reference, baseline_km, kernel, covariance = _build_microboone_bnb(
                item.configuration, repository_root, overrides=bnb_overrides
            )
            built.append(BuiltExperiment(
                experiment_id=item.experiment_id,
                status=item.status,
                correlation_group=item.correlation_group,
                configuration=item.configuration,
                evaluate=lambda parameters, active_workflow=workflow: active_workflow.likelihood.chi2(
                    active_workflow.predictor.predict_total_counts(parameters)
                ),
                predict_counts=workflow.predictor.predict_total_counts,
                observed_counts=workflow.likelihood.observed_counts,
                covariance_for_prediction=workflow.likelihood.covariance_for_prediction,
                metadata={
                    "kernel": str(kernel),
                    "covariance": str(covariance),
                    "statistical_treatment": workflow.statistical_treatment,
                    "covariance_parameter_dependence": workflow.covariance_parameter_dependence,
                    "reference_parameters": {
                        "delta_m2_41_eV2": reference.delta_m2_41_eV2,
                        "sin2_theta14": reference.sin2_theta14,
                        "sin2_theta24": reference.sin2_theta24,
                    },
                    "baseline_km": baseline_km,
                },
            ))
            continue
        if item.experiment_id == MICROBOONE_BNB_NUMI_JOINT_FOUR_CHANNEL:
            joint_document = yaml.safe_load(item.configuration.read_text(encoding="utf-8"))
            component_configs = joint_document["component_configs"]
            bnb_configuration = _repository_path(
                repository_root, component_configs["bnb"], label="joint BNB configuration"
            )
            numi_configuration = _repository_path(
                repository_root, component_configs["numi"], label="joint NuMI configuration"
            )
            bnb_workflow, reference, _, bnb_kernel, bnb_covariance = _build_microboone_bnb(
                bnb_configuration, repository_root, overrides=bnb_overrides
            )
            numi_document = yaml.safe_load(numi_configuration.read_text(encoding="utf-8"))
            numi_reference = ThreePlusOneParameters(
                **{name: float(value) for name, value in numi_document["reference_parameters"].items()}
            )
            if numi_reference != reference:
                raise ValueError("joint BNB and NuMI reference parameters must match")
            numi_kernel = _repository_path(
                repository_root,
                numi_document["diagnostic_four_channel_events"]["kernel_directory"],
                label="NuMI kernel",
            )
            numi_workflow = build_diagnostic_numi_workflow(
                numi_kernel, numi_reference, float(numi_document["baseline_km"])
            )
            released_covariance = _repository_path(
                repository_root, joint_document["released_covariance"], label="joint covariance"
            )
            joint_workflow = build_joint_microboone_bnb_numi_workflow(
                bnb_workflow, numi_workflow, released_covariance
            )
            built.append(BuiltExperiment(
                experiment_id=item.experiment_id,
                status=item.status,
                correlation_group=item.correlation_group,
                configuration=item.configuration,
                evaluate=joint_workflow.chi2,
                predict_counts=joint_workflow.predict_total_counts,
                observed_counts=joint_workflow.likelihood.observed_counts,
                covariance_for_prediction=joint_workflow.likelihood.covariance_for_prediction,
                metadata={
                    "bin_count": 208,
                    "covariance": str(released_covariance),
                    "covariance_includes_bnb_numi_cross_blocks": True,
                    "bnb_kernel": str(bnb_kernel),
                    "bnb_reference_covariance": str(bnb_covariance),
                    "numi_kernel": str(numi_kernel),
                    "statistical_treatment": "current-prediction Pearson diagonal",
                    "covariance_parameter_dependence": "prediction-scaled full 208x208 fractional systematics",
                },
            ))
            continue
        raise ValueError(f"no registered 3+1 workflow for {item.experiment_id!r}")
    contributions = tuple(
        ChiSquareContribution(item.experiment_id, item.correlation_group, item.chi2)
        for item in built
    )
    return BuiltAnalysis(selection.analysis_name, tuple(built), CombinedChiSquare(contributions))
