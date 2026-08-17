"""Interfaces shared by present and future oscillation models."""

from __future__ import annotations

from typing import Protocol

import numpy as np
from numpy.typing import NDArray


class VacuumOscillationModel(Protocol):
    """A model that returns P(initial flavour -> final flavour)."""

    def probability(
        self,
        initial_flavour: int,
        final_flavour: int,
        energy_GeV: NDArray[np.float64],
        baseline_km: float,
        *,
        antineutrino: bool = False,
    ) -> NDArray[np.float64]: ...
