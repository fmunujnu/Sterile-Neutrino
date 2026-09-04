"""Run the fast non-Toy CLs scan through the single canonical scan engine."""

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
        raise SystemExit("run1 fixes --cls-calibration=analytic; remove that option")
    if not _has_option(forwarded, "--output-directory"):
        destination = (
            ROOT
            / "outputs"
            / "run1_non_toy"
            / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        )
        forwarded.extend(("--output-directory", str(destination)))
    sys.argv = [str(SCAN), "--cls-calibration", "analytic", *forwarded]
    runpy.run_path(str(SCAN), run_name="__main__")


if __name__ == "__main__":
    main()
