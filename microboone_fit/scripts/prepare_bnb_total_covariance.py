"""Build a fixed-reference BNB total covariance from the released inputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from sterile_fit.covariance import cnp_total_covariance_at_reference, pearson_total_covariance_at_reference
from sterile_fit.published_inputs import load_bnb_four_channel_inputs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add declared data statistics to the released BNB four-channel systematic covariance."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/derived/bnb_four_channel_total_covariance.npz"),
    )
    parser.add_argument(
        "--statistical-method",
        choices=("pearson", "cnp"),
        default="pearson",
        help="Use pearson for the 2025 paper; cnp is retained only for the conflicting HEPData table-header convention.",
    )
    arguments = parser.parse_args()
    inputs = load_bnb_four_channel_inputs()
    provenance = (
        "HEPData 10.17182/hepdata.166435.v1 14-channel covariance, first four BNB channels; "
        "data statistics constructed at the released Signal + Background reference"
    )
    if arguments.statistical_method == "pearson":
        total = pearson_total_covariance_at_reference(
            inputs.systematic_covariance,
            inputs.published_total_prediction_counts,
            provenance=provenance + "; Pearson selected to match the 2025 paper Methods",
        )
    else:
        total = cnp_total_covariance_at_reference(
            inputs.systematic_covariance,
            inputs.observed_counts,
            inputs.published_total_prediction_counts,
            provenance=provenance + "; CNP selected explicitly from the HEPData table header, not as paper default",
        )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        arguments.output,
        covariance=total.covariance,
        statistical_treatment=total.statistical_treatment,
        parameter_dependence=total.parameter_dependence,
        reference_prediction_sha256=total.reference_prediction_sha256,
        provenance=total.provenance,
    )
    print(arguments.output)


if __name__ == "__main__":
    main()
