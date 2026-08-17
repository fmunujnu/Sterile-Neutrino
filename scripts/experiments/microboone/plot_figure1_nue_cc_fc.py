"""Plot only the BNB and NuMI electron-neutrino CC fully-contained samples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from sterile_fit.experiments.microboone.bnb.workflow import build_strict_bnb_workflow
from sterile_fit.experiments.microboone.joint_bnb_numi import build_joint_microboone_bnb_numi_workflow
from sterile_fit.experiments.microboone.numi.workflow import build_diagnostic_numi_workflow
from sterile_fit.parameters import ThreePlusOneParameters


ROOT = Path(__file__).resolve().parents[3]
BNB_CONFIG = ROOT / "configs" / "experiments" / "microboone" / "bnb" / "analysis.yaml"
NUMI_CONFIG = ROOT / "configs" / "experiments" / "microboone" / "numi" / "analysis.yaml"
DELTA_M2_41_EV2 = 1.2
SIN2_2THETA_MUE = 0.003
SIN2_THETA24_VALUES = (0.018, 0.0045)


def _reference_parameters(document: dict[str, object]) -> ThreePlusOneParameters:
    values = document["reference_parameters"]
    if not isinstance(values, dict):
        raise ValueError("reference_parameters must be a mapping")
    return ThreePlusOneParameters(**{name: float(value) for name, value in values.items()})


def _paper_parameters(sin2_theta24: float) -> ThreePlusOneParameters:
    sin2_2theta14 = SIN2_2THETA_MUE / sin2_theta24
    sin2_theta14 = (1.0 - np.sqrt(1.0 - sin2_2theta14)) / 2.0
    point = ThreePlusOneParameters(DELTA_M2_41_EV2, float(sin2_theta14), sin2_theta24)
    if not np.isclose(point.sin2_2theta_mue_exact, SIN2_2THETA_MUE, rtol=1e-12, atol=1e-15):
        raise RuntimeError("paper parameter conversion failed")
    return point


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot the two nue CC FC paper-comparison panels.")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "paper_reproduction" / "microboone_figure1_nue_cc_fc.png",
    )
    arguments = parser.parse_args()

    bnb_document = yaml.safe_load(BNB_CONFIG.read_text(encoding="utf-8"))
    numi_document = yaml.safe_load(NUMI_CONFIG.read_text(encoding="utf-8"))
    nominal_parameters = _reference_parameters(bnb_document)
    if _reference_parameters(numi_document) != nominal_parameters:
        raise ValueError("BNB and NuMI nominal parameters do not match")
    bnb = build_strict_bnb_workflow(
        ROOT / bnb_document["analysis_inputs"]["kernel"],
        ROOT / bnb_document["analysis_inputs"]["covariance"],
        nominal_parameters,
        float(bnb_document["baseline_km"]),
    )
    numi = build_diagnostic_numi_workflow(
        ROOT / numi_document["diagnostic_four_channel_events"]["kernel_directory"],
        nominal_parameters,
        float(numi_document["baseline_km"]),
    )
    joint = build_joint_microboone_bnb_numi_workflow(bnb, numi)
    paper_parameters = {value: _paper_parameters(value) for value in SIN2_THETA24_VALUES}
    nominal = joint.predict_total_counts(nominal_parameters)
    paper_predictions = {value: joint.predict_total_counts(point) for value, point in paper_parameters.items()}

    edges = np.concatenate((np.arange(0.0, 2.6, 0.1), [3.0]))
    centres = (edges[:-1] + edges[1:]) / 2.0
    figure, axes = plt.subplots(2, 1, figsize=(9.0, 10.0), sharex=True)
    rows: list[dict[str, object]] = []
    panels = (("BNB", 0, bnb.inputs), ("NuMI", 104, numi.inputs))
    for axis, (beam_name, joint_offset, inputs) in zip(axes, panels, strict=True):
        local = slice(0, 26)
        joint_bins = slice(joint_offset, joint_offset + 26)
        total = inputs.published_total_prediction_counts[local]
        sigma = np.sqrt(np.diag(inputs.systematic_covariance)[local])
        lower = np.maximum(total - sigma, 0.0)
        upper = total + sigma
        axis.stairs(inputs.published_background_counts[local], edges, color="tab:blue", linestyle="--", label="Background")
        axis.fill_between(
            edges, np.append(lower, lower[-1]), np.append(upper, upper[-1]),
            step="post", color="0.55", alpha=0.28, label="Prediction systematic (diagonal)",
        )
        axis.stairs(total, edges, color="black", label="Signal + Background")
        axis.stairs(nominal[joint_bins], edges, color="tab:purple", linestyle=":", linewidth=2.0, label=r"$3\nu$ nominal")
        axis.stairs(paper_predictions[0.018][joint_bins], edges, color="#00A6C8", linewidth=2.0, label=r"$\sin^2\theta_{24}=0.018$")
        axis.stairs(paper_predictions[0.0045][joint_bins], edges, color="#C24E00", linewidth=2.0, label=r"$\sin^2\theta_{24}=0.0045$")
        axis.errorbar(
            centres, inputs.observed_counts[local],
            yerr=[inputs.observed_statistical_error_down[local], inputs.observed_statistical_error_up[local]],
            fmt="o", color="tab:red", label="Data",
        )
        axis.set_title(rf"{beam_name} $\nu_e$ CC FC")
        axis.set_ylabel("Counts per bin")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
        for bin_index in range(26):
            rows.append({
                "beam": beam_name,
                "reco_bin": bin_index,
                "energy_low_GeV": edges[bin_index],
                "energy_high_GeV": edges[bin_index + 1],
                "data_counts": inputs.observed_counts[bin_index],
                "background_counts": inputs.published_background_counts[bin_index],
                "signal_plus_background_counts": total[bin_index],
                "nominal_3nu_counts": nominal[joint_offset + bin_index],
                "paper_sin2_theta24_0p018_counts": paper_predictions[0.018][joint_offset + bin_index],
                "paper_sin2_theta24_0p0045_counts": paper_predictions[0.0045][joint_offset + bin_index],
            })
    axes[-1].set_xlabel(r"Reconstructed neutrino energy, $E_\nu^{\mathrm{reco}}$ [GeV]")
    figure.suptitle(
        r"MicroBooNE $\nu_e$ CC FC: BNB and NuMI" + "\n" +
        r"$\Delta m^2_{41}=1.2\,\mathrm{eV}^2$, $\sin^2(2\theta_{\mu e})=0.003$"
    )
    figure.tight_layout()
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(arguments.output, dpi=180, bbox_inches="tight")
    plt.close(figure)
    pd.DataFrame(rows).to_csv(arguments.output.with_suffix(".csv"), index=False, float_format="%.17g")
    arguments.output.with_suffix(".metadata.json").write_text(
        json.dumps({
            "panels": ["BNB nue_cc_fc", "NuMI nue_cc_fc"],
            "paper_coordinates": {
                "delta_m2_41_eV2": DELTA_M2_41_EV2,
                "sin2_2theta_mue": SIN2_2THETA_MUE,
                "sin2_theta24": list(SIN2_THETA24_VALUES),
            },
            "scope": "new experiment-specific two-panel comparison; existing spectrum scripts are not replaced",
        }, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
