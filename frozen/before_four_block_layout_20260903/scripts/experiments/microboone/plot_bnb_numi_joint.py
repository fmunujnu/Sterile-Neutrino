"""Plot fixed BNB+NuMI spectra; this script performs no fit or optimisation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from sterile_fit.experiments.microboone.bnb.binning import BNB_FOUR_CHANNELS
from sterile_fit.experiments.microboone.bnb.workflow import build_strict_bnb_workflow
from sterile_fit.experiments.microboone.joint_bnb_numi import build_joint_microboone_bnb_numi_workflow
from sterile_fit.experiments.microboone.numi.binning import NUMI_FOUR_CHANNELS
from sterile_fit.experiments.microboone.numi.workflow import build_diagnostic_numi_workflow
from sterile_fit.parameters import ThreePlusOneParameters
from sterile_fit.spectrum_plotting import (
    SpectrumCurve,
    SpectrumPanel,
    render_microboone_spectrum_panels,
)


ROOT = Path(__file__).resolve().parents[3]
BNB_CONFIG = ROOT / "configs" / "experiments" / "microboone" / "bnb" / "analysis.yaml"
NUMI_CONFIG = ROOT / "configs" / "experiments" / "microboone" / "numi" / "analysis.yaml"
DEFAULT_OUTPUT = (
    ROOT / "outputs" / "spectra" / "microboone" / "bnb_numi_joint" / "published_reference_comparison.png"
)
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
    parameters = ThreePlusOneParameters(DELTA_M2_41_EV2, float(sin2_theta14), sin2_theta24)
    if not np.isclose(parameters.sin2_2theta_mue_exact, SIN2_2THETA_MUE, rtol=1e-12, atol=1e-15):
        raise RuntimeError("paper parameter conversion failed")
    return parameters


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot fixed published and requested reference spectra.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()

    bnb_document = yaml.safe_load(BNB_CONFIG.read_text(encoding="utf-8"))
    numi_document = yaml.safe_load(NUMI_CONFIG.read_text(encoding="utf-8"))
    my3nu_parameters = _reference_parameters(bnb_document)
    if _reference_parameters(numi_document) != my3nu_parameters:
        raise ValueError("BNB and NuMI reference parameters do not match")
    bnb = build_strict_bnb_workflow(
        ROOT / bnb_document["analysis_inputs"]["kernel"],
        ROOT / bnb_document["analysis_inputs"]["covariance"],
        my3nu_parameters,
        float(bnb_document["baseline_km"]),
    )
    numi = build_diagnostic_numi_workflow(
        ROOT / numi_document["diagnostic_four_channel_events"]["kernel_directory"],
        my3nu_parameters,
        float(numi_document["baseline_km"]),
    )
    joint = build_joint_microboone_bnb_numi_workflow(bnb, numi)

    paper_parameters = {value: _paper_parameters(value) for value in SIN2_THETA24_VALUES}
    my3nu = joint.predict_total_counts(my3nu_parameters)
    paper_predictions = {value: joint.predict_total_counts(point) for value, point in paper_parameters.items()}
    published_total = np.concatenate(
        (bnb.inputs.published_total_prediction_counts, numi.inputs.published_total_prediction_counts)
    )
    observed = np.concatenate((bnb.inputs.observed_counts, numi.inputs.observed_counts))
    error_up = np.concatenate(
        (bnb.inputs.observed_statistical_error_up, numi.inputs.observed_statistical_error_up)
    )
    error_down = np.concatenate(
        (bnb.inputs.observed_statistical_error_down, numi.inputs.observed_statistical_error_down)
    )
    background = np.concatenate(
        (bnb.inputs.published_background_counts, numi.inputs.published_background_counts)
    )

    if not np.allclose(my3nu, published_total, rtol=1e-10, atol=1e-10):
        raise RuntimeError("my3nu empirical reconstruction no longer closes to Signal+Background")

    edges = np.concatenate((np.arange(0.0, 2.6, 0.1), [3.0]))
    panels: list[SpectrumPanel] = []
    rows: list[dict[str, object]] = []
    for beam_row, (beam_name, channels, beam_offset) in enumerate(
        (("BNB", BNB_FOUR_CHANNELS, 0), ("NuMI", NUMI_FOUR_CHANNELS, 104))
    ):
        for channel_column, channel in enumerate(channels):
            start = beam_offset + 26 * channel_column
            stop = start + 26
            panels.append(
                SpectrumPanel(
                    title=f"{beam_name} {channel.identifier}",
                    energy_edges_GeV=edges,
                    observed_counts=observed[start:stop],
                    observed_error_down=error_down[start:stop],
                    observed_error_up=error_up[start:stop],
                    background_counts=background[start:stop],
                    signal_plus_background_counts=published_total[start:stop],
                    comparison_curves=(
                        SpectrumCurve("my3nu", my3nu[start:stop], "tab:purple", ":", 2.0),
                        SpectrumCurve(
                            r"Paper Fig. 1 point: $\sin^2\theta_{24}=0.018$",
                            paper_predictions[0.018][start:stop], "#00A6C8", "-", 2.0,
                        ),
                        SpectrumCurve(
                            r"Paper Fig. 1 point: $\sin^2\theta_{24}=0.0045$",
                            paper_predictions[0.0045][start:stop], "#C24E00", "-", 2.0,
                        ),
                    ),
                    prediction_systematic_sigma=np.sqrt(
                        np.diag(bnb.inputs.systematic_covariance)[start:stop]
                    ) if beam_name == "BNB" else np.sqrt(
                        np.diag(numi.inputs.systematic_covariance)[26 * channel_column:26 * (channel_column + 1)]
                    ),
                )
            )
            for local_bin in range(26):
                index = start + local_bin
                rows.append(
                    {
                        "beam": beam_name,
                        "channel": channel.identifier,
                        "channel_reco_bin": local_bin,
                        "joint_reco_bin": index,
                        "observed_counts": observed[index],
                        "published_background_counts": background[index],
                        "published_signal_plus_background_counts": published_total[index],
                        "my3nu_counts": my3nu[index],
                        "paper_sin2_theta24_0p018_counts": paper_predictions[0.018][index],
                        "paper_sin2_theta24_0p0045_counts": paper_predictions[0.0045][index],
                    }
                )

    render_microboone_spectrum_panels(
        panels,
        arguments.output,
        title="MicroBooNE BNB+NuMI: public four-channel input reproduction\n"
        "(last bin is the released overflow bin)",
    )
    pd.DataFrame(rows).to_csv(arguments.output.with_suffix(".csv"), index=False, float_format="%.17g")

    metadata = {
        "status": "fixed_spectra_only_no_fit_or_optimisation",
        "chi2_definition": "single 208-bin quadratic form including BNB-NuMI cross-covariance",
        "curves": {
            "published_signal_plus_background": {"joint_chi2": joint.likelihood.chi2(published_total)},
            "my3nu": {
                "parameters": {
                    "delta_m2_41_eV2": my3nu_parameters.delta_m2_41_eV2,
                    "sin2_theta14": my3nu_parameters.sin2_theta14,
                    "sin2_theta24": my3nu_parameters.sin2_theta24,
                },
                "joint_chi2": joint.chi2(my3nu_parameters),
                "maximum_difference_from_published_total_counts": float(np.max(np.abs(my3nu - published_total))),
            },
            **{
                f"paper_sin2_theta24_{str(value).replace('.', 'p')}": {
                    "parameters": {
                        "delta_m2_41_eV2": point.delta_m2_41_eV2,
                        "sin2_theta14": point.sin2_theta14,
                        "sin2_theta24": point.sin2_theta24,
                        "sin2_2theta_mue_exact": point.sin2_2theta_mue_exact,
                    },
                    "joint_chi2": joint.chi2(point),
                }
                for value, point in paper_parameters.items()
            },
        },
        "note": "Signal+Background and my3nu coincide by empirical reference construction; this is algebraic closure, not an independent physics validation.",
    }
    arguments.output.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
