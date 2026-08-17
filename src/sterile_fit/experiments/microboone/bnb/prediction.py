"""Strict validation of a BNB template predictor against a published anchor."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ....models.three_plus_one import ThreePlusOneVacuumModel
from ....parameters import ThreePlusOneParameters
from .templates import BnbFourChannelOscillationTemplates


@dataclass(frozen=True, slots=True)
class BnbFourChannelPredictor:
    """A parameter-dependent predictor built solely from declared templates."""

    templates: BnbFourChannelOscillationTemplates
    baseline_km: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.baseline_km) or self.baseline_km <= 0.0:
            raise ValueError("baseline_km must be strictly positive")

    def predict_total_counts(self, parameters: ThreePlusOneParameters) -> NDArray[np.float64]:
        return self.templates.predict_total_counts(ThreePlusOneVacuumModel(parameters), self.baseline_km)

    def validate_published_reference(
        self,
        reference_parameters: ThreePlusOneParameters,
        published_reference_total_counts: NDArray[np.float64],
        *,
        relative_tolerance: float = 1e-6,
        absolute_tolerance: float = 1e-8,
    ) -> None:
        """Require the supplied physical templates to reproduce the chosen anchor.

        No per-bin correction is applied. A mismatch is evidence that the
        template definition, reference parameters, or bin order is wrong.
        """
        published = np.asarray(published_reference_total_counts, dtype=float)
        prediction = self.predict_total_counts(reference_parameters)
        if published.shape != (104,):
            raise ValueError("published reference total must have shape (104,)")
        if not np.allclose(prediction, published, rtol=relative_tolerance, atol=absolute_tolerance):
            largest = int(np.argmax(np.abs(prediction - published)))
            raise ValueError(
                "template does not reproduce the published reference; "
                f"largest mismatch at bin {largest}: predicted={prediction[largest]:.8g}, "
                f"published={published[largest]:.8g}"
            )
