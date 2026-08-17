"""Parameter-dependent prediction for the diagnostic NuMI four-channel adapter."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ....models.three_plus_one import ThreePlusOneVacuumModel
from ....parameters import ThreePlusOneParameters
from .event_prediction import NumiFourChannelEmpiricalKernel


@dataclass(frozen=True, slots=True)
class NumiFourChannelPredictor:
    kernel: NumiFourChannelEmpiricalKernel
    baseline_km: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.baseline_km) or self.baseline_km <= 0.0:
            raise ValueError("NuMI baseline_km must be finite and positive")

    def predict_total_counts(self, parameters: ThreePlusOneParameters) -> NDArray[np.float64]:
        model = ThreePlusOneVacuumModel(parameters)
        return self.kernel.predict_total_counts(model, self.baseline_km)

    def validate_reference(
        self,
        parameters: ThreePlusOneParameters,
        published_total_prediction_counts: NDArray[np.float64],
    ) -> None:
        predicted = self.predict_total_counts(parameters)
        published = np.asarray(published_total_prediction_counts, dtype=float)
        if published.shape != (104,):
            raise ValueError("NuMI published reference must have shape (104,)")
        if not np.allclose(predicted, published, rtol=1e-10, atol=1e-10):
            largest = int(np.argmax(np.abs(predicted - published)))
            raise ValueError(f"NuMI reference closure failed at local bin {largest}")
