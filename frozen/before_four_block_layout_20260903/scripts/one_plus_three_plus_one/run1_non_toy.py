"""Run the parallel 1+3+1 scan with analytic CLs calibration."""

from __future__ import annotations

from pathlib import Path
import runpy
import sys


SCAN = Path(__file__).with_name("scan.py")


def main() -> None:
    if any(item == "--cls-calibration" or item.startswith("--cls-calibration=") for item in sys.argv[1:]):
        raise SystemExit("run1 fixes --cls-calibration=analytic; remove that option")
    sys.argv = [str(SCAN), "--cls-calibration", "analytic", *sys.argv[1:]]
    runpy.run_path(str(SCAN), run_name="__main__")


if __name__ == "__main__":
    main()
