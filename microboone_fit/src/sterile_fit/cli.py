"""Small inspection commands; fitting commands are added only after validation."""

from __future__ import annotations

import argparse

import numpy as np

from .models.three_plus_one import ThreePlusOneVacuumModel
from .parameters import ThreePlusOneParameters
from .published_inputs import load_bnb_four_channel_inputs


def _inspect_published() -> None:
    inputs = load_bnb_four_channel_inputs()
    print("scope: BNB first four channels (104 bins)")
    print(f"observed counts: {inputs.observed_counts.sum():.6g}")
    print(f"published total prediction: {inputs.published_total_prediction_counts.sum():.6g}")
    print(f"published background: {inputs.published_background_counts.sum():.6g}")
    print(f"covariance shape: {inputs.systematic_covariance.shape}")


def _validate_three_plus_one() -> None:
    parameters = ThreePlusOneParameters(
        delta_m2_41_eV2=1.2,
        sin2_theta14=0.041666666666666664,
        sin2_theta24=0.018,
    )
    model = ThreePlusOneVacuumModel(parameters)
    energies = np.array([0.2, 0.7, 1.4], dtype=float)
    total_probability = sum(
        model.probability(0, final_flavour, energies, baseline_km=0.541)
        for final_flavour in range(4)
    )
    if not np.allclose(total_probability, 1.0, atol=1e-12):
        raise AssertionError("3+1 probability does not conserve total flavour probability")
    print("3+1 parameter conversion and probability-conservation check passed")
    print(f"exact sin2(2theta_mue): {parameters.sin2_2theta_mue_exact:.8g}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    subcommands.add_parser("inspect-published", help="validate and summarize the public BNB inputs")
    subcommands.add_parser("validate-3plus1", help="run parameter and probability consistency checks")
    arguments = parser.parse_args()
    if arguments.command == "inspect-published":
        _inspect_published()
    elif arguments.command == "validate-3plus1":
        _validate_three_plus_one()


if __name__ == "__main__":
    main()
