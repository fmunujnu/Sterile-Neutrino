"""Run the parallel 1+3+1 scan with empirical Toy-MC CLs calibration."""

from __future__ import annotations

from pathlib import Path
import runpy
import sys


SCAN = Path(__file__).with_name("scan.py")


def main() -> None:
    if any(item == "--cls-calibration" or item.startswith("--cls-calibration=") for item in sys.argv[1:]):
        raise SystemExit("run2 fixes --cls-calibration=toy; remove that option")
    forwarded = list(sys.argv[1:])
    if not any(item == "--number-of-toys" or item.startswith("--number-of-toys=") for item in forwarded):
        forwarded.extend(("--number-of-toys", "100"))
    sys.argv = [str(SCAN), "--cls-calibration", "toy", *forwarded]
    runpy.run_path(str(SCAN), run_name="__main__")


if __name__ == "__main__":
    main()
