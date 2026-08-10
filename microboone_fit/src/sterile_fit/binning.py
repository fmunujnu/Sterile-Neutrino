"""Declared bin layouts; no analysis code may rely on unexplained slices."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Channel:
    identifier: str
    first_global_bin: int
    bin_count: int = 26

    @property
    def stop_global_bin(self) -> int:
        return self.first_global_bin + self.bin_count


BNB_FOUR_CHANNELS: tuple[Channel, ...] = (
    Channel("nue_cc_fc", 0),
    Channel("nue_cc_pc", 26),
    Channel("numu_cc_fc", 52),
    Channel("numu_cc_pc", 78),
)


def bnb_four_channel_indices() -> tuple[int, ...]:
    """Indices 0..103, derived from declared four 26-bin BNB channels."""
    indices: list[int] = []
    for channel in BNB_FOUR_CHANNELS:
        indices.extend(range(channel.first_global_bin, channel.stop_global_bin))
    return tuple(indices)
