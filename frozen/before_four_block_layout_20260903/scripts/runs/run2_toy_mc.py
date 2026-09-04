"""Run the Toy-MC-calibrated CLs scan through the canonical scan engine."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
import runpy
import sys


ROOT = Path(__file__).resolve().parents[2]
SCAN = ROOT / "scripts" / "scan.py"


def _has_option(arguments: list[str], name: str) -> bool:
    return any(item == name or item.startswith(f"{name}=") for item in arguments)


def main() -> None:
    forwarded = list(sys.argv[1:])
    if _has_option(forwarded, "--cls-calibration"):
        raise SystemExit("run2 fixes --cls-calibration=toy; remove that option")
    if not _has_option(forwarded, "--number-of-toys"):
        forwarded.extend(("--number-of-toys", "100"))
    if not _has_option(forwarded, "--output-directory"):
        destination = (
            ROOT
            / "outputs"
            / "run2_toy_mc"
            / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        )
        forwarded.extend(("--output-directory", str(destination)))
    sys.argv = [str(SCAN), "--cls-calibration", "toy", *forwarded]
    runpy.run_path(str(SCAN), run_name="__main__")


if __name__ == "__main__":
    main()
