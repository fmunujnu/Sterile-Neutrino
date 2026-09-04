"""Attach 1+3+1 probabilities to the unchanged experiment likelihood inputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping

import numpy as np
from numpy.typing import NDArray
import yaml

from ..analysis.combination import ChiSquareContribution, CombinedChiSquare
from ..analysis.registry import (
    MICROBOONE_BNB_FOUR_CHANNEL,
    MICROBOONE_BNB_NUMI_JOINT_FOUR_CHANNEL,
    build_three_plus_one_analysis,
)
from ..analysis.selection import AnalysisSelection
from ..covariance import solve_quadratic_form
from ..experiments.microboone.bnb.templates import BnbFourChannelOscillationTemplates
from ..experiments.microboone.numi.event_prediction import NumiFourChannelEmpiricalKernel
from .model import OnePlusThreePlusOneVacuumModel
from .parameters import OnePlusThreePlusOneParameters


FloatVector = NDArray[np.float64]
FloatMatrix = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class BuiltOnePlusThreePlusOneExperiment:
    """One existing experiment likelihood with a 1+3+1 prediction callable."""

    experiment_id: str
    status: str
    correlation_group: str
    configuration: Path
    predict_counts: Callable[[OnePlusThreePlusOneParameters], FloatVector]
    observed_counts: FloatVector
    covariance_for_prediction: Callable[[FloatVector], FloatMatrix]
    metadata: Mapping[str, object]

    def chi2(self, parameters: OnePlusThreePlusOneParameters) -> float:
        prediction = self.predict_counts(parameters)
        covariance = self.covariance_for_prediction(prediction)
        return float(
            solve_quadratic_form(
                np.asarray(self.observed_counts, dtype=float) - prediction,
                covariance,
            )
        )


@dataclass(frozen=True, slots=True)
class BuiltOnePlusThreePlusOneAnalysis:
    analysis_name: str
    experiments: tuple[BuiltOnePlusThreePlusOneExperiment, ...]
    objective: CombinedChiSquare[OnePlusThreePlusOneParameters]


def _repository_path(repository_root: Path, value: str, *, label: str) -> Path:
    path = (repository_root / value).resolve()
    root = repository_root.resolve()
    if path != root and root not in path.parents:
        raise ValueError(f"{label} must stay inside the repository")
    return path


def _joint_baselines(configuration: Path, repository_root: Path) -> tuple[float, float]:
    document = yaml.safe_load(configuration.read_text(encoding="utf-8"))
    component_configs = document["component_configs"]
    bnb_configuration = _repository_path(
        repository_root, component_configs["bnb"], label="joint BNB configuration"
    )
    numi_configuration = _repository_path(
        repository_root, component_configs["numi"], label="joint NuMI configuration"
    )
    bnb_document = yaml.safe_load(bnb_configuration.read_text(encoding="utf-8"))
    numi_document = yaml.safe_load(numi_configuration.read_text(encoding="utf-8"))
    return float(bnb_document["baseline_km"]), float(numi_document["baseline_km"])


def build_one_plus_three_plus_one_analysis(
    selection: AnalysisSelection,
    *,
    repository_root: Path,
    bnb_overrides: Mapping[str, Path] | None = None,
) -> BuiltOnePlusThreePlusOneAnalysis:
    """Reuse the registered 3+1 data/covariance workflows without changing them.

    The existing builder remains the single validation path for public inputs,
    covariance construction and experiment selection.  Only its model-specific
    prediction callable is replaced locally by the parallel 1+3+1 model.  This
    keeps BNB, NuMI and their released cross-covariance treatment byte-for-byte
    identical to the currently selected 3+1 analysis.
    """
    base_analysis = build_three_plus_one_analysis(
        selection,
        repository_root=repository_root,
        bnb_overrides=bnb_overrides,
    )
    built: list[BuiltOnePlusThreePlusOneExperiment] = []

    for base in base_analysis.experiments:
        if base.experiment_id == MICROBOONE_BNB_FOUR_CHANNEL:
            templates = BnbFourChannelOscillationTemplates.from_directory(
                Path(str(base.metadata["kernel"]))
            )
            baseline_km = float(base.metadata["baseline_km"])

            def predict_bnb(
                parameters: OnePlusThreePlusOneParameters,
                *,
                active_templates: BnbFourChannelOscillationTemplates = templates,
                active_baseline_km: float = baseline_km,
            ) -> FloatVector:
                return active_templates.predict_total_counts(
                    OnePlusThreePlusOneVacuumModel(parameters), active_baseline_km
                )

            predict_counts = predict_bnb
            additional_metadata: dict[str, object] = {"baseline_km": baseline_km}
        elif base.experiment_id == MICROBOONE_BNB_NUMI_JOINT_FOUR_CHANNEL:
            bnb_templates = BnbFourChannelOscillationTemplates.from_directory(
                Path(str(base.metadata["bnb_kernel"]))
            )
            numi_kernel = NumiFourChannelEmpiricalKernel.from_directory(
                Path(str(base.metadata["numi_kernel"]))
            )
            bnb_baseline_km, numi_baseline_km = _joint_baselines(
                base.configuration, repository_root
            )

            def predict_joint(
                parameters: OnePlusThreePlusOneParameters,
                *,
                active_bnb_templates: BnbFourChannelOscillationTemplates = bnb_templates,
                active_numi_kernel: NumiFourChannelEmpiricalKernel = numi_kernel,
                active_bnb_baseline_km: float = bnb_baseline_km,
                active_numi_baseline_km: float = numi_baseline_km,
            ) -> FloatVector:
                model = OnePlusThreePlusOneVacuumModel(parameters)
                return np.concatenate((
                    active_bnb_templates.predict_total_counts(
                        model, active_bnb_baseline_km
                    ),
                    active_numi_kernel.predict_total_counts(
                        model, active_numi_baseline_km
                    ),
                ))

            predict_counts = predict_joint
            additional_metadata = {
                "bnb_baseline_km": bnb_baseline_km,
                "numi_baseline_km": numi_baseline_km,
            }
        else:
            raise ValueError(
                f"no parallel 1+3+1 prediction adapter for {base.experiment_id!r}"
            )

        built.append(BuiltOnePlusThreePlusOneExperiment(
            experiment_id=base.experiment_id,
            status=base.status,
            correlation_group=base.correlation_group,
            configuration=base.configuration,
            predict_counts=predict_counts,
            observed_counts=base.observed_counts,
            covariance_for_prediction=base.covariance_for_prediction,
            metadata={
                **dict(base.metadata),
                **additional_metadata,
                "physics_model": "1+3+1 effective e/mu short-baseline CC",
                "three_plus_one_core_modified": False,
            },
        ))

    contributions = tuple(
        ChiSquareContribution(item.experiment_id, item.correlation_group, item.chi2)
        for item in built
    )
    return BuiltOnePlusThreePlusOneAnalysis(
        f"{selection.analysis_name}__one_plus_three_plus_one",
        tuple(built),
        CombinedChiSquare(contributions),
    )
