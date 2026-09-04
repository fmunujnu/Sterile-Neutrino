"""Explicit NuMI channel-block selection from the 14-channel release."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class NumiChannel:
    identifier: str
    released_channel_ordinal: int
    first_published_bin: int
    bin_count: int = 26

    @property
    def stop_published_bin(self) -> int:
        return self.first_published_bin + self.bin_count


# Human-readable channels 8--11 in the release (one-based), equivalently
# zero-based channel blocks 7--10.  Each block contains 26 reconstructed bins.
NUMI_FOUR_CHANNELS: tuple[NumiChannel, ...] = (
    NumiChannel("nue_cc_fc", 8, 7 * 26),
    NumiChannel("nue_cc_pc", 9, 8 * 26),
    NumiChannel("numu_cc_fc", 10, 9 * 26),
    NumiChannel("numu_cc_pc", 11, 10 * 26),
)


def numi_four_channel_published_indices() -> tuple[int, ...]:
    """Return the declared 104 global-bin indices in released-table order."""
    indices: list[int] = []
    for channel in NUMI_FOUR_CHANNELS:
        indices.extend(range(channel.first_published_bin, channel.stop_published_bin))
    return tuple(indices)
