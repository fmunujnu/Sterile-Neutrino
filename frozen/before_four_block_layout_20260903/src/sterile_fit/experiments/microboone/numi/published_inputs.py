"""NuMI-only adapter for the public 14-channel spectra and covariance."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from ..bnb.published_inputs import (
    DEFAULT_COVARIANCE_PATH,
    DEFAULT_SPECTRUM_PATH,
    read_full_systematic_covariance,
    read_full_unconstrained_spectrum,
)
from .binning import numi_four_channel_published_indices


@dataclass(frozen=True, slots=True)
class PublishedNumiFourChannelInputs:
    """Published values for NuMI channels 8--11, flattened to 104 bins."""

    observed_counts: NDArray[np.float64]
    published_background_counts: NDArray[np.float64]
    published_total_prediction_counts: NDArray[np.float64]
    observed_statistical_error_up: NDArray[np.float64]
    observed_statistical_error_down: NDArray[np.float64]
    systematic_covariance: NDArray[np.float64]

    def __post_init__(self) -> None:
        vector_names = (
            "observed_counts",
            "published_background_counts",
            "published_total_prediction_counts",
            "observed_statistical_error_up",
            "observed_statistical_error_down",
        )
        for name in vector_names:
            values = np.asarray(getattr(self, name), dtype=float)
            if values.shape != (104,) or not np.all(np.isfinite(values)):
                raise ValueError(f"{name} must be a finite 104-bin vector")
        if self.systematic_covariance.shape != (104, 104):
            raise ValueError("systematic_covariance must be the 104x104 four-channel block matrix")
        if not np.all(np.isfinite(self.systematic_covariance)):
            raise ValueError("systematic_covariance contains non-finite entries")
        if not np.allclose(self.systematic_covariance, self.systematic_covariance.T, rtol=1e-10, atol=1e-12):
            raise ValueError("systematic_covariance must be symmetric in the selected channel order")
        if np.any(self.published_total_prediction_counts < self.published_background_counts):
            raise ValueError("published total prediction is below published background")

    @property
    def published_signal_counts(self) -> NDArray[np.float64]:
        """HEPData Signal+Background minus its Background block."""
        return self.published_total_prediction_counts - self.published_background_counts


def load_numi_four_channel_inputs(
    spectrum_path: Path = DEFAULT_SPECTRUM_PATH,
    covariance_path: Path = DEFAULT_COVARIANCE_PATH,
) -> PublishedNumiFourChannelInputs:
    """Select channel blocks 8--11 only after validating all 364 bins."""
    full_spectrum = read_full_unconstrained_spectrum(spectrum_path)
    full_covariance = read_full_systematic_covariance(covariance_path)
    indices = np.asarray(numi_four_channel_published_indices(), dtype=int)
    selected = {name: values[indices] for name, values in full_spectrum.items()}
    return PublishedNumiFourChannelInputs(
        **selected,
        systematic_covariance=full_covariance[np.ix_(indices, indices)],
    )
