"""Generate the joint spectrum plus paper-coordinate Fig. 3a/3b diagnostics."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
JOINT_CONFIG = ROOT / "configs" / "analyses" / "microboone_bnb_numi.yaml"


def _run(arguments: list[str]) -> None:
    subprocess.run([sys.executable, *arguments], cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate the BNB+NuMI spectrum and both exact profile-coordinate scans."
    )
    parser.add_argument("--grid-points", type=int, default=61)
    parser.add_argument("--output-directory", type=Path)
    arguments = parser.parse_args()
    destination = arguments.output_directory or (
        ROOT
        / "outputs"
        / "paper_reproduction"
        / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ_microboone_bnb_numi")
    )
    destination.mkdir(parents=True, exist_ok=False)

    # This paper-specific two-panel plot is additive.  It does not replace or
    # modify the existing BNB four-channel and joint-spectrum generators.
    _run([
        "scripts/experiments/microboone/plot_figure1_nue_cc_fc.py",
        "--output",
        str(destination / "bnb_numi_spectra.png"),
    ])
    _run([
        "scripts/scan.py",
        "--analysis-config",
        str(JOINT_CONFIG),
        "--mode",
        "appearance-profile",
        "--grid-points",
        str(arguments.grid_points),
        "--sin2-2theta-mue-min", "1e-4",
        "--sin2-2theta-mue-max", "1",
        "--delta-m2-min-eV2", "1e-2",
        "--delta-m2-max-eV2", "1e2",
        "--output-directory",
        str(destination / "fig3a_sin2_2theta_mue"),
    ])
    _run([
        "scripts/scan.py",
        "--analysis-config",
        str(JOINT_CONFIG),
        "--mode",
        "electron-disappearance-profile",
        "--grid-points",
        str(arguments.grid_points),
        "--sin2-2theta-ee-min", "1e-2",
        "--sin2-2theta-ee-max", "1",
        "--delta-m2-min-eV2", "1e-1",
        "--delta-m2-max-eV2", "14",
        "--output-directory",
        str(destination / "fig3b_sin2_2theta_ee"),
    ])
    print(destination)


if __name__ == "__main__":
    main()
