"""Run a template-validated BNB four-channel 3+1 prefit."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from sterile_fit.fitting import prefit_three_plus_one
from sterile_fit.parameters import ThreePlusOneParameters
from sterile_fit.workflows import build_strict_bnb_workflow


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-delta-m2-eV2", type=float, required=True)
    parser.add_argument("--reference-sin2-theta14", type=float, required=True)
    parser.add_argument("--reference-sin2-theta24", type=float, required=True)
    parser.add_argument("--templates", type=Path, required=True)
    parser.add_argument("--total-covariance", type=Path, required=True)
    arguments = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    reference = ThreePlusOneParameters(
        arguments.reference_delta_m2_eV2,
        arguments.reference_sin2_theta14,
        arguments.reference_sin2_theta24,
    )
    workflow = build_strict_bnb_workflow(arguments.templates, arguments.total_covariance, reference)
    result = prefit_three_plus_one(lambda parameters: workflow.likelihood.chi2(workflow.predictor.predict_total_counts(parameters)))
    run_dir = root / "runs" / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ_prefit_3plus1")
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "result.json").write_text(json.dumps({
        "scope": "MicroBooNE BNB first four channels template-validated 3+1 profile",
        "statistical_treatment": workflow.statistical_treatment,
        "covariance_parameter_dependence": workflow.covariance_parameter_dependence,
        "reference_parameters": reference.__dict__ if hasattr(reference, "__dict__") else {
            "delta_m2_41_eV2": reference.delta_m2_41_eV2,
            "sin2_theta14": reference.sin2_theta14,
            "sin2_theta24": reference.sin2_theta24,
        },
        "best_fit": {
            "delta_m2_41_eV2": result.parameters.delta_m2_41_eV2,
            "sin2_theta14": result.parameters.sin2_theta14,
            "sin2_theta24": result.parameters.sin2_theta24,
            "chi2": result.chi2,
        },
    }, indent=2), encoding="utf-8")
    print(run_dir)


if __name__ == "__main__":
    main()
