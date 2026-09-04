"""Readers and validators for the public MicroBooNE BNB inputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from ....paths import REPOSITORY_ROOT
from .binning import BNB_FOUR_CHANNELS, bnb_four_channel_indices


MICROBOONE_DATA_ROOT = REPOSITORY_ROOT / "data" / "experiments" / "microboone"
DEFAULT_SPECTRUM_PATH = MICROBOONE_DATA_ROOT / "shared" / "raw" / "hepdata_microboone_2025" / "HEPData-ins3088922-v1-Unconstrained_14_channels.csv"
DEFAULT_COVARIANCE_PATH = MICROBOONE_DATA_ROOT / "shared" / "raw" / "hepdata_microboone_2025" / "HEPData-ins3088922-v1-14_channel_covariance_matrix.csv"


@dataclass(frozen=True, slots=True)
class PublishedBnbFourChannelInputs:
    """Published BNB data and nominal prediction for exactly 104 bins."""

    observed_counts: NDArray[np.float64]
    published_background_counts: NDArray[np.float64]
    published_total_prediction_counts: NDArray[np.float64]
    observed_statistical_error_up: NDArray[np.float64]
    observed_statistical_error_down: NDArray[np.float64]
    systematic_covariance: NDArray[np.float64]

    def __post_init__(self) -> None:
        expected = len(bnb_four_channel_indices())
        arrays = (
            self.observed_counts,
            self.published_background_counts,
            self.published_total_prediction_counts,
            self.observed_statistical_error_up,
            self.observed_statistical_error_down,
        )
        if any(np.asarray(array).shape != (expected,) for array in arrays):
            raise ValueError(f"all BNB four-channel vectors must have shape ({expected},)")
        if self.systematic_covariance.shape != (expected, expected):
            raise ValueError(f"systematic covariance must have shape ({expected}, {expected})")
        if np.any(self.published_total_prediction_counts < self.published_background_counts):
            raise ValueError("published total prediction must be at least the published background")

    @property
    def published_signal_counts(self) -> NDArray[np.float64]:
        return self.published_total_prediction_counts - self.published_background_counts


def _parse_spectrum_blocks(path: Path) -> dict[str, list[list[float]]]:
    blocks: dict[str, list[list[float]]] = {"data": [], "background": [], "total": []}
    current_block: str | None = None
    headers = {
        "Data [counts per bin]": "data",
        "Background [counts per bin]": "background",
        "Signal + Background [counts per bin]": "total",
    }

    with path.open(encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            fields = [field.strip() for field in line.split(",")]
            if len(fields) >= 4 and fields[3] in headers:
                current_block = headers[fields[3]]
                continue
            if current_block is None:
                continue
            try:
                value = float(fields[3])
            except (IndexError, ValueError):
                continue
            # Data has asymmetric statistical errors; other blocks only have the value column.
            row = [value]
            if current_block == "data":
                try:
                    row.extend([float(fields[4]), abs(float(fields[5]))])
                except (IndexError, ValueError):
                    raise ValueError(f"data row lacks usable statistical errors: {line!r}") from None
            blocks[current_block].append(row)
    return blocks


def read_full_unconstrained_spectrum(path: Path = DEFAULT_SPECTRUM_PATH) -> dict[str, NDArray[np.float64]]:
    """Read all 364 released bins before any BNB-only selection."""
    blocks = _parse_spectrum_blocks(path)
    expected = 14 * 26
    if any(len(blocks[name]) != expected for name in blocks):
        sizes = {name: len(rows) for name, rows in blocks.items()}
        raise ValueError(f"expected 364 values in each published spectrum block, got {sizes}")
    data = np.asarray(blocks["data"], dtype=float)
    return {
        "observed_counts": data[:, 0],
        "observed_statistical_error_up": data[:, 1],
        "observed_statistical_error_down": data[:, 2],
        "published_background_counts": np.asarray(blocks["background"], dtype=float)[:, 0],
        "published_total_prediction_counts": np.asarray(blocks["total"], dtype=float)[:, 0],
    }


def read_full_systematic_covariance(path: Path = DEFAULT_COVARIANCE_PATH) -> NDArray[np.float64]:
    """Read the released 364x364 covariance without silently truncating it."""
    table = pd.read_csv(
        path,
        comment="#",
        names=["column", "row", "covariance"],
        header=None,
    )
    for column in table.columns:
        table[column] = pd.to_numeric(table[column], errors="coerce")
    table = table.dropna()
    if not np.all(np.isfinite(table.to_numpy(dtype=float))):
        raise ValueError("published covariance contains non-finite values")
    if not np.all(np.equal(table[["column", "row"]].to_numpy(dtype=float) % 1.0, 0.0)):
        raise ValueError("published covariance indices must be integers")
    minimum_index = int(min(table["column"].min(), table["row"].min()))
    maximum_index = int(max(table["column"].max(), table["row"].max()))
    if (minimum_index, maximum_index) != (0, 363):
        raise ValueError(f"expected zero-based 364-bin covariance, got index range {minimum_index}..{maximum_index}")
    if len(table) != 364 * 364 or table.duplicated(["row", "column"]).any():
        raise ValueError("published covariance must contain every 364x364 coordinate exactly once")
    covariance = np.zeros((364, 364), dtype=float)
    covariance[table["row"].astype(int), table["column"].astype(int)] = table["covariance"].to_numpy(dtype=float)
    if not np.allclose(covariance, covariance.T, rtol=1e-10, atol=1e-12):
        raise ValueError("published covariance is not symmetric; inspect its bin ordering before fitting")
    return covariance


def load_bnb_four_channel_inputs(
    spectrum_path: Path = DEFAULT_SPECTRUM_PATH,
    covariance_path: Path = DEFAULT_COVARIANCE_PATH,
) -> PublishedBnbFourChannelInputs:
    """Select the declared BNB first-four-channel block from validated 364-bin data."""
    full_spectrum = read_full_unconstrained_spectrum(spectrum_path)
    full_covariance = read_full_systematic_covariance(covariance_path)
    indices = np.asarray(bnb_four_channel_indices(), dtype=int)
    selected = {name: values[indices] for name, values in full_spectrum.items()}
    return PublishedBnbFourChannelInputs(
        **selected,
        systematic_covariance=full_covariance[np.ix_(indices, indices)],
    )
