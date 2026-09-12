"""Shared contracts between experiment-specific adapters and common analysis code.

Experiments may use different public data products internally.  They cross the
experiment boundary through likelihood and, when available, binned-prediction
interfaces; scanning, calibration, persistence, and plotting remain shared.
"""
from __future__ import annotations

from pathlib import Path
from typing import Generic, Mapping, Protocol, TypeVar, runtime_checkable

import numpy as np
from numpy.typing import NDArray


ParametersT = TypeVar("ParametersT")
FloatVector = NDArray[np.float64]
FloatMatrix = NDArray[np.float64]


@runtime_checkable
class ExperimentLikelihoodAdapter(Protocol, Generic[ParametersT]):
    """Minimum interface required to combine independent experiments."""

    experiment_id: str
    configuration: Path
    metadata: Mapping[str, object]

    def negative_two_log_likelihood(self, parameters: ParametersT) -> float:
        """Return this experiment's contribution for the supplied model point."""


@runtime_checkable
class BinnedExperimentAdapter(ExperimentLikelihoodAdapter[ParametersT], Protocol):
    """Additional interface for covariance-based, binned experiments."""

    observed_counts: FloatVector

    def predict_counts(self, parameters: ParametersT) -> FloatVector:
        """Return predicted reconstructed-bin counts."""

    def covariance_for_prediction(self, prediction: FloatVector) -> FloatMatrix:
        """Return the declared covariance for exactly that prediction."""

