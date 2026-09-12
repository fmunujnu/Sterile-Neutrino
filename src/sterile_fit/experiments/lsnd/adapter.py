"""LSND final-publication facts and 3+1 probability-convention entry."""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from sterile_fit.experiments.lsnd.final_2001 import PUBLISHED, core_mapping_rows, load_published_summary
from sterile_fit.output import result_directory, write_csv, write_json
from sterile_fit.experiments.lsnd.public_rate_approximation import (
    LSNDDARRateInputs,
    rate_negative_two_log_likelihood,
    three_plus_one_average_probability,
)


def run_lsnd() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--kind", choices=("official", "core-mapping", "rate-scan"),
        default="official",
    )
    parser.add_argument("--output-directory", type=Path)
    args = parser.parse_args()
    summary = load_published_summary()
    output = args.output_directory or result_directory("lsnd_final_2001", "two_flavour_appearance", args.kind)
    output.mkdir(parents=True, exist_ok=False)
    if args.kind == "official":
        table = pd.read_csv(PUBLISHED)
    elif args.kind == "core-mapping":
        table = pd.DataFrame(core_mapping_rows(summary))
    else:
        inputs = LSNDDARRateInputs()
        masses = np.geomspace(1e-2, 1e2, 241)
        amplitudes = np.geomspace(1e-4, 1.0, 241)
        rows = []
        for mass in masses:
            # Appearance probability is exactly linear in this 3+1 amplitude.
            unit_average = three_plus_one_average_probability(mass, 1.0, inputs)
            predictions = amplitudes * unit_average
            statistics = [
                rate_negative_two_log_likelihood(value, inputs)
                for value in predictions
            ]
            rows.extend(
                (mass, amplitude, prediction, statistic)
                for amplitude, prediction, statistic
                in zip(amplitudes, predictions, statistics, strict=True)
            )
        table = pd.DataFrame(rows, columns=[
            "delta_m2_41_eV2", "sin2_2theta_mue",
            "predicted_average_probability", "negative_2_log_likelihood",
        ])
    write_csv(table, output / "result.csv", index=False)
    metadata = {
        "release": "LSND final oscillation paper, Aguilar et al., PRD 64 112007 (2001)",
        "arxiv": "https://arxiv.org/abs/hep-ex/0104049",
        "mode": args.kind,
        "published_best_fit": {"delta_m2_eV2": summary.best_fit_delta_m2_eV2,
                                "sin2_2theta_mue": summary.best_fit_sin2_2theta_mue},
        "three_plus_one_mapping": "sin2_2theta_mue = 4|Ue4|^2|Umu4|^2; representative sin2(theta14)=1/2",
        "official_surface_or_contour": "not machine-readablely released; not digitised",
        "local_likelihood_surface": "not computed: missing event table, component PDFs, and Gaussian background inputs",
        "line_overlay_and_surface_metrics": "not generated: no comparable numerical surfaces",
        "coverage_pilot": "not run: final paper's generated-data inputs and likelihood are not public",
        "statistical_method_evidence": "Sec. IX.B--F, pp.23--25: four-variable event likelihood with Gaussian background variation; final regions use constant slices; full FC described but not followed.",
    }
    if args.kind == "rate-scan":
        from sterile_fit.output import render_lsnd_rate_parameter_space
        inputs = LSNDDARRateInputs()
        published_prediction = three_plus_one_average_probability(
            summary.best_fit_delta_m2_eV2,
            summary.best_fit_sin2_2theta_mue,
            inputs,
        )
        metadata.update({
            "local_likelihood_surface": "computed public-input DAR rate-only approximation",
            "result_role": "model-portable approximation; not the LSND four-variable likelihood",
            "measurement": {
                "average_probability": inputs.observed_average_probability,
                "statistical_sigma": inputs.statistical_sigma,
                "systematic_sigma": inputs.systematic_sigma,
            },
            "public_kernel": {
                "flux": "muon-DAR anti-nu_mu Michel spectrum",
                "cross_section": "leading IBD positron energy times momentum",
                "baseline": "uniform 8.3 m detector axis about 30 m, weighted by 1/L^2",
                "selection_efficiency": "constant; published average cancels in probability ratio",
                "missing": "source extent, transverse geometry, energy-dependent efficiency and reconstruction migration",
            },
            "constant_slice_thresholds": {
                "90_percent": 4.605,
                "99_percent": 9.210,
                "warning": "paper-matching two-parameter slices, not calibrated coverage for this one-rate approximation",
            },
            "published_best_fit_in_rate_approximation": {
                "predicted_average_probability": published_prediction,
                "delta_negative_2_log_likelihood": rate_negative_two_log_likelihood(
                    published_prediction, inputs
                ),
            },
        })
        render_lsnd_rate_parameter_space(table, output / "parameter_space.png")
        render_lsnd_rate_parameter_space(
            table, output / "parameter_space_lines.png", heatmap=False
        )
    write_json(output / "metadata.json", metadata)
    print(output)
