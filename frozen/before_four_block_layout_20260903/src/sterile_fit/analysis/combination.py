"""Model-agnostic combination of explicitly selected experiment objectives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Generic, Mapping, TypeVar


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
