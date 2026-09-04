"""experiments/microboone/joint.py: regrouped existing implementations; see docs/ARCHITECTURE.md."""
from __future__ import annotations
from sterile_fit.output import result_directory

from dataclasses import dataclass
from pathlib import Path
from functools import lru_cache
import numpy as np
from numpy.typing import NDArray
from sterile_fit.core.likelihood import PredictionScaledGaussianLikelihood
from sterile_fit.core.three_plus_one import ThreePlusOneParameters
from sterile_fit.experiments.microboone.public_data import bnb_four_channel_indices
from sterile_fit.experiments.microboone.public_data import DEFAULT_COVARIANCE_PATH, read_full_systematic_covariance
from sterile_fit.experiments.microboone.bnb import StrictBnbWorkflow
from sterile_fit.experiments.microboone.public_data import numi_four_channel_published_indices
from sterile_fit.experiments.microboone.numi import DiagnosticNumiWorkflow
# Joint BNB+NuMI four-channel workflow with released cross-beam covariance.


class _ShortBaselineResponseProjector:
    """Cache exact kernel×sin²(phase) projections for one beam.

    This is algebraically the same 3+1 short-baseline probability used by
    ``ThreePlusOneVacuumModel``.  Only quantities independent of the profiled
    mixing coordinates are cached.
    """

    def __init__(self, templates: object, baseline_km: float) -> None:
        self.energy_GeV = np.asarray(getattr(templates, "true_energy_GeV"), dtype=float)
        self.baseline_km = float(baseline_km)
        self.background = np.asarray(
            getattr(templates, "fixed_published_background_counts"), dtype=float
        )
        self.ee = np.asarray(getattr(templates, "beam_nue_to_nue_cc_response_counts")) + np.asarray(
            getattr(templates, "beam_nuebar_to_nuebar_cc_response_counts")
        )
        self.mue = np.asarray(getattr(templates, "beam_numu_to_nue_cc_response_counts")) + np.asarray(
            getattr(templates, "beam_numubar_to_nuebar_cc_response_counts")
        )
        self.emu = np.asarray(getattr(templates, "beam_nue_to_numu_cc_response_counts")) + np.asarray(
            getattr(templates, "beam_nuebar_to_numubar_cc_response_counts")
        )
        self.mumu = np.asarray(getattr(templates, "beam_numu_to_numu_cc_response_counts")) + np.asarray(
            getattr(templates, "beam_numubar_to_numubar_cc_response_counts")
        )
        self.no_oscillation_survival_counts = self.ee.sum(axis=1) + self.mumu.sum(axis=1)

    @lru_cache(maxsize=512)
    def _phase_projections(
        self, delta_m2_41_eV2: float
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        phase = np.sin(
            1.267 * float(delta_m2_41_eV2) * self.baseline_km / self.energy_GeV
        ) ** 2
        return self.ee @ phase, self.mue @ phase, self.emu @ phase, self.mumu @ phase

    def predict(self, parameters: ThreePlusOneParameters) -> NDArray[np.float64]:
        projected_ee, projected_mue, projected_emu, projected_mumu = self._phase_projections(
            parameters.delta_m2_41_eV2
        )
        ue4_squared = parameters.sin2_theta14
        umu4_squared = (1.0 - parameters.sin2_theta14) * parameters.sin2_theta24
        electron_survival_amplitude = 4.0 * ue4_squared * (1.0 - ue4_squared)
        muon_survival_amplitude = 4.0 * umu4_squared * (1.0 - umu4_squared)
        appearance_amplitude = 4.0 * ue4_squared * umu4_squared
        return (
            self.background
            + self.no_oscillation_survival_counts
            - electron_survival_amplitude * projected_ee
            - muon_survival_amplitude * projected_mumu
            + appearance_amplitude * (projected_mue + projected_emu)
        )


def joint_bnb_numi_published_indices() -> tuple[int, ...]:
    """Return BNB then NuMI bins in the joint prediction-vector order."""
    return (*bnb_four_channel_indices(), *numi_four_channel_published_indices())


@dataclass(frozen=True, slots=True)
class JointMicrobooneBnbNumiWorkflow:
    """One 208-bin likelihood; cross terms prevent a unique chi2 split."""

    bnb: StrictBnbWorkflow
    numi: DiagnosticNumiWorkflow
    likelihood: PredictionScaledGaussianLikelihood
    systematic_covariance: NDArray[np.float64]
    bnb_numi_cross_covariance: NDArray[np.float64]
    bnb_projector: _ShortBaselineResponseProjector
    numi_projector: _ShortBaselineResponseProjector

    def predict_total_counts(self, parameters: ThreePlusOneParameters) -> NDArray[np.float64]:
        return np.concatenate((self.bnb_projector.predict(parameters), self.numi_projector.predict(parameters)))

    def chi2(self, parameters: ThreePlusOneParameters) -> float:
        return self.likelihood.chi2(self.predict_total_counts(parameters))


def build_joint_microboone_bnb_numi_workflow(
    bnb: StrictBnbWorkflow,
    numi: DiagnosticNumiWorkflow,
    covariance_path: Path = DEFAULT_COVARIANCE_PATH,
) -> JointMicrobooneBnbNumiWorkflow:
    """Select the full 208x208 block, including both 104x104 cross blocks."""
    full_covariance = read_full_systematic_covariance(covariance_path)
    indices = np.asarray(joint_bnb_numi_published_indices(), dtype=int)
    systematic = full_covariance[np.ix_(indices, indices)]
    cross = systematic[:104, 104:]
    if systematic.shape != (208, 208) or cross.shape != (104, 104):
        raise ValueError("joint MicroBooNE covariance selection has an invalid shape")
    if not np.any(np.abs(cross) > 0.0):
        raise ValueError("released BNB-NuMI cross-covariance block is unexpectedly zero")
    observed = np.concatenate((bnb.inputs.observed_counts, numi.inputs.observed_counts))
    reference = np.concatenate(
        (
            bnb.inputs.published_total_prediction_counts,
            numi.inputs.published_total_prediction_counts,
        )
    )
    return JointMicrobooneBnbNumiWorkflow(
        bnb=bnb,
        numi=numi,
        likelihood=PredictionScaledGaussianLikelihood(observed, reference, systematic),
        systematic_covariance=systematic,
        bnb_numi_cross_covariance=cross,
        bnb_projector=_ShortBaselineResponseProjector(
            bnb.predictor.templates, bnb.predictor.baseline_km
        ),
        numi_projector=_ShortBaselineResponseProjector(
            numi.predictor.kernel, numi.predictor.baseline_km
        ),
    )



# Spectrum payloads belong to this experiment; rendering/writing is shared.
import argparse
import json
import pandas as pd
import yaml
from sterile_fit.output import SpectrumCurve, SpectrumPanel, render_microboone_spectrum_panels, write_csv, write_json
from sterile_fit.experiments.microboone.public_data import BNB_FOUR_CHANNELS, NUMI_FOUR_CHANNELS
from sterile_fit.experiments.microboone.bnb import build_strict_bnb_workflow
from sterile_fit.experiments.microboone.numi import build_diagnostic_numi_workflow

# Plot fixed BNB+NuMI spectra; this script performs no fit or optimisation.


from sterile_fit.paths import REPOSITORY_ROOT as JOINT_PLOT_ROOT
JOINT_PLOT_BNB_CONFIG = JOINT_PLOT_ROOT / "configs" / "experiments" / "microboone" / "bnb" / "analysis.yaml"
JOINT_PLOT_NUMI_CONFIG = JOINT_PLOT_ROOT / "configs" / "experiments" / "microboone" / "numi" / "analysis.yaml"
JOINT_PLOT_DELTA_M2_41_EV2 = 1.2
JOINT_PLOT_SIN2_2THETA_MUE = 0.003
JOINT_PLOT_SIN2_THETA24_VALUES = (0.018, 0.0045)


def joint_plot_reference_parameters(document: dict[str, object]) -> ThreePlusOneParameters:
    values = document["reference_parameters"]
    if not isinstance(values, dict):
        raise ValueError("reference_parameters must be a mapping")
    return ThreePlusOneParameters(**{name: float(value) for name, value in values.items()})


def joint_plot_paper_parameters(sin2_theta24: float) -> ThreePlusOneParameters:
    sin2_2theta14 = JOINT_PLOT_SIN2_2THETA_MUE / sin2_theta24
    sin2_theta14 = (1.0 - np.sqrt(1.0 - sin2_2theta14)) / 2.0
    parameters = ThreePlusOneParameters(JOINT_PLOT_DELTA_M2_41_EV2, float(sin2_theta14), sin2_theta24)
    if not np.isclose(parameters.sin2_2theta_mue_exact, JOINT_PLOT_SIN2_2THETA_MUE, rtol=1e-12, atol=1e-15):
        raise RuntimeError("paper parameter conversion failed")
    return parameters


def plot_joint_spectrum() -> None:
    parser = argparse.ArgumentParser(description="Plot fixed published and requested reference spectra.")
    parser.add_argument("--output", type=Path, default=result_directory("microboone_bnb_numi_joint", "three_plus_one", "spectra_joint") / "published_reference_comparison.png")
    arguments = parser.parse_args()

    bnb_document = yaml.safe_load(JOINT_PLOT_BNB_CONFIG.read_text(encoding="utf-8"))
    numi_document = yaml.safe_load(JOINT_PLOT_NUMI_CONFIG.read_text(encoding="utf-8"))
    my3nu_parameters = joint_plot_reference_parameters(bnb_document)
    if joint_plot_reference_parameters(numi_document) != my3nu_parameters:
        raise ValueError("BNB and NuMI reference parameters do not match")
    bnb = build_strict_bnb_workflow(
        JOINT_PLOT_ROOT / bnb_document["analysis_inputs"]["kernel"],
        JOINT_PLOT_ROOT / bnb_document["analysis_inputs"]["covariance"],
        my3nu_parameters,
        float(bnb_document["baseline_km"]),
    )
    numi = build_diagnostic_numi_workflow(
        JOINT_PLOT_ROOT / numi_document["diagnostic_four_channel_events"]["kernel_directory"],
        my3nu_parameters,
        float(numi_document["baseline_km"]),
    )
    joint = build_joint_microboone_bnb_numi_workflow(bnb, numi)

    paper_parameters = {value: joint_plot_paper_parameters(value) for value in JOINT_PLOT_SIN2_THETA24_VALUES}
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
    write_csv(pd.DataFrame(rows), arguments.output.with_suffix(".csv"), index=False, float_format="%.17g")

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
    write_json(arguments.output.with_suffix(".metadata.json"), metadata)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


# Plot only the BNB and NuMI electron-neutrino CC fully-contained samples.


from sterile_fit.paths import REPOSITORY_ROOT as FIGURE1_ROOT
FIGURE1_BNB_CONFIG = FIGURE1_ROOT / "configs" / "experiments" / "microboone" / "bnb" / "analysis.yaml"
FIGURE1_NUMI_CONFIG = FIGURE1_ROOT / "configs" / "experiments" / "microboone" / "numi" / "analysis.yaml"
FIGURE1_DELTA_M2_41_EV2 = 1.2
FIGURE1_SIN2_2THETA_MUE = 0.003
FIGURE1_SIN2_THETA24_VALUES = (0.018, 0.0045)


def figure1_reference_parameters(document: dict[str, object]) -> ThreePlusOneParameters:
    values = document["reference_parameters"]
    if not isinstance(values, dict):
        raise ValueError("reference_parameters must be a mapping")
    return ThreePlusOneParameters(**{name: float(value) for name, value in values.items()})


def figure1_paper_parameters(sin2_theta24: float) -> ThreePlusOneParameters:
    sin2_2theta14 = FIGURE1_SIN2_2THETA_MUE / sin2_theta24
    sin2_theta14 = (1.0 - np.sqrt(1.0 - sin2_2theta14)) / 2.0
    point = ThreePlusOneParameters(FIGURE1_DELTA_M2_41_EV2, float(sin2_theta14), sin2_theta24)
    if not np.isclose(point.sin2_2theta_mue_exact, FIGURE1_SIN2_2THETA_MUE, rtol=1e-12, atol=1e-15):
        raise RuntimeError("paper parameter conversion failed")
    return point


def plot_figure1_spectrum() -> None:
    parser = argparse.ArgumentParser(description="Plot the two nue CC FC paper-comparison panels.")
    parser.add_argument(
        "--output",
        type=Path,
        default=result_directory("microboone_bnb_numi_joint", "three_plus_one", "spectra_figure1") / "figure1_nue_cc_fc.png",
    )
    arguments = parser.parse_args()

    bnb_document = yaml.safe_load(FIGURE1_BNB_CONFIG.read_text(encoding="utf-8"))
    numi_document = yaml.safe_load(FIGURE1_NUMI_CONFIG.read_text(encoding="utf-8"))
    nominal_parameters = figure1_reference_parameters(bnb_document)
    if figure1_reference_parameters(numi_document) != nominal_parameters:
        raise ValueError("BNB and NuMI nominal parameters do not match")
    bnb = build_strict_bnb_workflow(
        FIGURE1_ROOT / bnb_document["analysis_inputs"]["kernel"],
        FIGURE1_ROOT / bnb_document["analysis_inputs"]["covariance"],
        nominal_parameters,
        float(bnb_document["baseline_km"]),
    )
    numi = build_diagnostic_numi_workflow(
        FIGURE1_ROOT / numi_document["diagnostic_four_channel_events"]["kernel_directory"],
        nominal_parameters,
        float(numi_document["baseline_km"]),
    )
    joint = build_joint_microboone_bnb_numi_workflow(bnb, numi)
    paper_parameters = {value: figure1_paper_parameters(value) for value in FIGURE1_SIN2_THETA24_VALUES}
    nominal = joint.predict_total_counts(nominal_parameters)
    paper_predictions = {value: joint.predict_total_counts(point) for value, point in paper_parameters.items()}

    edges = np.concatenate((np.arange(0.0, 2.6, 0.1), [3.0]))
    rows: list[dict[str, object]] = []
    panels = (("BNB", 0, bnb.inputs), ("NuMI", 104, numi.inputs))
    spectrum_panels = []
    for beam_name, joint_offset, inputs in panels:
        local = slice(0, 26)
        joint_bins = slice(joint_offset, joint_offset + 26)
        total = inputs.published_total_prediction_counts[local]
        sigma = np.sqrt(np.diag(inputs.systematic_covariance)[local])
        spectrum_panels.append(SpectrumPanel(
            title=rf"{beam_name} $\nu_e$ CC FC", energy_edges_GeV=edges,
            observed_counts=inputs.observed_counts[local],
            observed_error_down=inputs.observed_statistical_error_down[local],
            observed_error_up=inputs.observed_statistical_error_up[local],
            background_counts=inputs.published_background_counts[local],
            signal_plus_background_counts=total, prediction_systematic_sigma=sigma,
            comparison_curves=(
                SpectrumCurve(r"$3\nu$ nominal", nominal[joint_bins], "tab:purple", ":", 2.0),
                SpectrumCurve(r"$\sin^2\theta_{24}=0.018$", paper_predictions[0.018][joint_bins], "#00A6C8", "-", 2.0),
                SpectrumCurve(r"$\sin^2\theta_{24}=0.0045$", paper_predictions[0.0045][joint_bins], "#C24E00", "-", 2.0),
            ),
        ))
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
    render_microboone_spectrum_panels(
        spectrum_panels, arguments.output, layout="figure1", title=
        r"MicroBooNE $\nu_e$ CC FC: BNB and NuMI" + "\n" +
        r"$\Delta m^2_{41}=1.2\,\mathrm{eV}^2$, $\sin^2(2\theta_{\mu e})=0.003$"
    )
    write_csv(pd.DataFrame(rows), arguments.output.with_suffix(".csv"), index=False, float_format="%.17g")
    write_json(arguments.output.with_suffix(".metadata.json"), {
            "panels": ["BNB nue_cc_fc", "NuMI nue_cc_fc"],
            "paper_coordinates": {
                "delta_m2_41_eV2": FIGURE1_DELTA_M2_41_EV2,
                "sin2_2theta_mue": FIGURE1_SIN2_2THETA_MUE,
                "sin2_theta24": list(FIGURE1_SIN2_THETA24_VALUES),
            },
            "scope": "new experiment-specific two-panel comparison; existing spectrum scripts are not replaced",
        })


