"""Joint BNB+NuMI four-channel workflow with released cross-beam covariance."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from functools import lru_cache

import numpy as np
from numpy.typing import NDArray

from ...likelihood import PredictionScaledGaussianLikelihood
from ...parameters import ThreePlusOneParameters
from .bnb.binning import bnb_four_channel_indices
from .bnb.published_inputs import DEFAULT_COVARIANCE_PATH, read_full_systematic_covariance
from .bnb.workflow import StrictBnbWorkflow
from .numi.binning import numi_four_channel_published_indices
from .numi.workflow import DiagnosticNumiWorkflow


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
