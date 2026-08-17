"""Visible configuration interface for choosing accepted analysis inputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


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
