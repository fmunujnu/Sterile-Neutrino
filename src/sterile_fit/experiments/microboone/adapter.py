"""Select experiments and attach model predictions to unchanged likelihoods."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import yaml
from typing import Callable, Generic, Mapping, TypeVar
from typing import Callable, Mapping
import numpy as np
from numpy.typing import NDArray
from sterile_fit.experiments.microboone.bnb import StrictBnbWorkflow, build_strict_bnb_workflow
from sterile_fit.experiments.microboone.joint import build_joint_microboone_bnb_numi_workflow
from sterile_fit.experiments.microboone.numi import build_energy_baseline_numi_workflow
from sterile_fit.core.three_plus_one import ThreePlusOneParameters
from sterile_fit.core.likelihood import solve_quadratic_form
from sterile_fit.experiments.microboone.bnb import BnbFourChannelOscillationTemplates
from sterile_fit.experiments.microboone.numi import NumiFourChannelEmpiricalKernel
from sterile_fit.core.one_plus_three_plus_one import OnePlusThreePlusOneVacuumModel
from sterile_fit.core.one_plus_three_plus_one import OnePlusThreePlusOneParameters
from sterile_fit.core.calibration import GaussianHypothesis


def check_selected_inputs(selection_name="all"):
    """Cheap input/closure checks, not a full scan or proof of physical accuracy."""
    from sterile_fit.paths import REPOSITORY_ROOT
    from sterile_fit.experiments.microboone.bnb import check_bnb

    if selection_name in {"bnb", "all"}:
        check_bnb()
    if selection_name in {"joint", "all"}:
        selection = load_analysis_selection(
            REPOSITORY_ROOT / "configs/analyses/microboone_bnb_numi.yaml",
            repository_root=REPOSITORY_ROOT,
        )
        analysis = build_three_plus_one_analysis(selection, repository_root=REPOSITORY_ROOT)
        parameters = ThreePlusOneParameters(1.2, 0.0, 0.0)
        experiment = analysis.experiments[0]
        prediction = experiment.predict_counts(parameters)
        if prediction.shape != (208,) or not np.isfinite(analysis.objective.chi2(parameters)):
            raise AssertionError("Joint prediction or covariance check failed")
        print("PASS declared joint BNB+NuMI inputs and full cross-covariance (approximate detector response)")
        extended = build_one_plus_three_plus_one_analysis(selection, repository_root=REPOSITORY_ROOT)
        extended_null = OnePlusThreePlusOneParameters.three_neutrino_null()
        if not np.allclose(extended.experiments[0].predict_counts(extended_null), prediction, rtol=1e-12, atol=1e-10):
            raise AssertionError("1+3+1 and 3+1 null predictions disagree")
        print("PASS 1+3+1 joint null closure (not a full model/profile validation)")
# Visible configuration interface for choosing accepted analysis inputs.


ALLOWED_STATUSES = {"validated_surrogate", "approximate", "inputs_unavailable"}


@dataclass(frozen=True, slots=True)
class ExperimentSelection:
    """One experiment/beam likelihood and whether the scan includes it."""

    experiment_id: str
    include: bool
    status: str
    correlation_group: str
    configuration: Path
    note: str


@dataclass(frozen=True, slots=True)
class AnalysisSelection:
    """Named set of experiment likelihoods used by a scan."""

    analysis_name: str
    experiments: tuple[ExperimentSelection, ...]

    @property
    def included(self) -> tuple[ExperimentSelection, ...]:
        return tuple(item for item in self.experiments if item.include)


def load_analysis_selection(path: Path, *, repository_root: Path) -> AnalysisSelection:
    """Load a selection and reject unavailable included analyses."""
    path = Path(path)
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or not isinstance(document.get("analysis_name"), str):
        raise ValueError("analysis selection must declare analysis_name")
    rows = document.get("experiments")
    if not isinstance(rows, list) or not rows:
        raise ValueError("analysis selection must declare a non-empty experiments list")
    experiments: list[ExperimentSelection] = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("each experiment selection must be a mapping")
        identifier = row.get("experiment_id")
        include = row.get("include")
        status = row.get("status")
        correlation_group = row.get("correlation_group")
        configuration = row.get("configuration")
        if not isinstance(identifier, str) or not identifier:
            raise ValueError("each experiment selection needs a non-empty experiment_id")
        if not isinstance(include, bool):
            raise ValueError(f"{identifier}: include must be true or false")
        if status not in ALLOWED_STATUSES:
            raise ValueError(f"{identifier}: unknown status {status!r}")
        if not isinstance(correlation_group, str) or not correlation_group:
            raise ValueError(f"{identifier}: correlation_group must be a non-empty string")
        if not isinstance(configuration, str) or not configuration:
            raise ValueError(f"{identifier}: configuration must be a repository-relative path")
        configuration_path = (repository_root / configuration).resolve()
        if repository_root.resolve() not in configuration_path.parents:
            raise ValueError(f"{identifier}: configuration must stay inside the repository")
        if not configuration_path.is_file():
            raise ValueError(f"{identifier}: configuration file does not exist: {configuration_path}")
        if include and status == "inputs_unavailable":
            raise ValueError(f"{identifier}: cannot include an analysis whose inputs are unavailable")
        experiments.append(ExperimentSelection(
            experiment_id=identifier,
            include=include,
            status=status,
            correlation_group=correlation_group,
            configuration=configuration_path,
            note=str(row.get("note", "")),
        ))
    identifiers = [item.experiment_id for item in experiments]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("experiment_id values must be unique")
    selection = AnalysisSelection(document["analysis_name"], tuple(experiments))
    if not selection.included:
        raise ValueError("analysis selection must include at least one available experiment")
    return selection


# Model-agnostic combination of explicitly selected experiment objectives.


ParametersT = TypeVar("ParametersT")


@dataclass(frozen=True, slots=True)
class ChiSquareContribution(Generic[ParametersT]):
    """One experiment's named chi-square contribution."""

    experiment_id: str
    correlation_group: str
    evaluate: Callable[[ParametersT], float]


@dataclass(frozen=True, slots=True)
class CombinedChiSquare(Generic[ParametersT]):
    """Sum selected objectives without knowing experiment-specific details."""

    contributions: tuple[ChiSquareContribution[ParametersT], ...]

    def __post_init__(self) -> None:
        identifiers = [item.experiment_id for item in self.contributions]
        if not identifiers:
            raise ValueError("at least one experiment contribution must be selected")
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("experiment contribution identifiers must be unique")
        correlation_groups = [item.correlation_group for item in self.contributions]
        if len(correlation_groups) != len(set(correlation_groups)):
            raise ValueError(
                "correlated datasets cannot be summed as separate chi-square contributions; "
                "register one joint workflow with the full block covariance"
            )

    def breakdown(self, parameters: ParametersT) -> Mapping[str, float]:
        """Return auditable per-experiment chi-square values."""
        return {item.experiment_id: float(item.evaluate(parameters)) for item in self.contributions}

    def chi2(self, parameters: ParametersT) -> float:
        """Return the sum used by a joint fit or profile scan."""
        return float(sum(self.breakdown(parameters).values()))

    __call__ = chi2


# Factories connecting experiment IDs to their private workflows.


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

    def negative_two_log_likelihood(self, parameters: ThreePlusOneParameters) -> float:
        """Shared adapter name; identical to the existing Gaussian chi-square."""
        return self.chi2(parameters)


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
            energy_baseline_distribution = _repository_path(
                repository_root,
                numi_document["energy_baseline_distribution"]["path"],
                label="NuMI energy-baseline distribution",
            )
            numi_workflow = build_energy_baseline_numi_workflow(
                numi_kernel, numi_reference, energy_baseline_distribution
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
                    "numi_energy_baseline_distribution": str(energy_baseline_distribution),
                    "numi_baseline_treatment": "energy- and flavour-dependent public-dk2nu conditional baseline average",
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


# Attach 1+3+1 probabilities to the unchanged experiment likelihood inputs.


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

    def negative_two_log_likelihood(
        self, parameters: OnePlusThreePlusOneParameters
    ) -> float:
        """Shared adapter name; identical to the existing Gaussian chi-square."""
        return self.chi2(parameters)


@dataclass(frozen=True, slots=True)
class BuiltOnePlusThreePlusOneAnalysis:
    analysis_name: str
    experiments: tuple[BuiltOnePlusThreePlusOneExperiment, ...]
    objective: CombinedChiSquare[OnePlusThreePlusOneParameters]


def _extended_repository_path(repository_root: Path, value: str, *, label: str) -> Path:
    path = (repository_root / value).resolve()
    root = repository_root.resolve()
    if path != root and root not in path.parents:
        raise ValueError(f"{label} must stay inside the repository")
    return path


def _joint_baselines(configuration: Path, repository_root: Path) -> tuple[float, float]:
    document = yaml.safe_load(configuration.read_text(encoding="utf-8"))
    component_configs = document["component_configs"]
    bnb_configuration = _extended_repository_path(
        repository_root, component_configs["bnb"], label="joint BNB configuration"
    )
    numi_configuration = _extended_repository_path(
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



# Hypothesis/observation adapters. Original arithmetic order is retained.

def _hypothesis_pairs(analysis, null_parameters, tested_parameters):
    pairs = []
    for experiment in analysis.experiments:
        null_prediction = experiment.predict_counts(null_parameters)
        tested_prediction = experiment.predict_counts(tested_parameters)
        pairs.append((
            GaussianHypothesis(
                null_prediction,
                experiment.covariance_for_prediction(null_prediction),
            ),
            GaussianHypothesis(
                tested_prediction,
                experiment.covariance_for_prediction(tested_prediction),
            ),
        ))
    return pairs


def _objective_for_toy(analysis, toy_dataset):
    """Build the same prediction-scaled chi2 with pseudo-data replacing data."""
    if len(toy_dataset) != len(analysis.experiments):
        raise ValueError("toy dataset does not match the selected analysis")

    def objective(parameters: ThreePlusOneParameters) -> float:
        total = 0.0
        for experiment, observation in zip(
            analysis.experiments, toy_dataset, strict=True
        ):
            prediction = experiment.predict_counts(parameters)
            covariance = experiment.covariance_for_prediction(prediction)
            total += solve_quadratic_form(observation - prediction, covariance)
        return float(total)

    return objective


def extended_hypothesis_pairs(analysis, null_parameters, tested_parameters):
    pairs = []
    for experiment in analysis.experiments:
        null_prediction = experiment.predict_counts(null_parameters)
        tested_prediction = experiment.predict_counts(tested_parameters)
        pairs.append((
            GaussianHypothesis(
                null_prediction,
                experiment.covariance_for_prediction(null_prediction),
            ),
            GaussianHypothesis(
                tested_prediction,
                experiment.covariance_for_prediction(tested_prediction),
            ),
        ))
    return tuple(pairs)


def extended_objective_for_toy(analysis, toy_dataset):
    if len(toy_dataset) != len(analysis.experiments):
        raise ValueError("toy dataset does not match the selected analysis")

    def objective(parameters: OnePlusThreePlusOneParameters) -> float:
        total = 0.0
        for experiment, observation in zip(
            analysis.experiments, toy_dataset, strict=True
        ):
            prediction = experiment.predict_counts(parameters)
            covariance = experiment.covariance_for_prediction(prediction)
            total += solve_quadratic_form(observation - prediction, covariance)
        return float(total)

    return objective

