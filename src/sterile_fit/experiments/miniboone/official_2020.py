"""Official MiniBooNE 2020 nue+nuebar appearance-release adapter.

The release describes six covariance blocks in this exact order:
nu signal, nu electron-like background, nu muon control,
nubar signal, nubar electron-like background, nubar muon control.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
from numpy.typing import NDArray

from sterile_fit.paths import REPOSITORY_ROOT
from sterile_fit.core.three_plus_one import (
    ThreePlusOneParameters,
    short_baseline_appearance_probability,
)

RAW = REPOSITORY_ROOT / "data/experiments/miniboone/shared/raw/official_nue2020_combined"


@dataclass(frozen=True, slots=True)
class MiniBooNE2020Data:
    electron_edges_MeV: NDArray[np.float64]
    nu_e_data: NDArray[np.float64]
    nu_mu_data: NDArray[np.float64]
    nu_e_background: NDArray[np.float64]
    nu_mu_prediction: NDArray[np.float64]
    nu_full_transmutation: NDArray[np.float64]
    nubar_e_data: NDArray[np.float64]
    nubar_mu_data: NDArray[np.float64]
    nubar_e_background: NDArray[np.float64]
    nubar_mu_prediction: NDArray[np.float64]
    nubar_full_transmutation: NDArray[np.float64]
    fractional_covariance: NDArray[np.float64]


def load_release(directory: Path = RAW) -> MiniBooNE2020Data:
    load = lambda name: np.loadtxt(directory / name, dtype=float)
    result = MiniBooNE2020Data(
        electron_edges_MeV=load("miniboone_binboundaries_nue_lowe.txt"),
        nu_e_data=load("miniboone_nuedata_lowe.txt"),
        nu_mu_data=load("miniboone_numudata.txt"),
        nu_e_background=load("miniboone_nuebgr_lowe.txt"),
        nu_mu_prediction=load("miniboone_numu.txt"),
        nu_full_transmutation=load("miniboone_numunuefullosc_ntuple.txt"),
        nubar_e_data=load("miniboone_nuebardata_lowe.txt"),
        nubar_mu_data=load("miniboone_numubardata.txt"),
        nubar_e_background=load("miniboone_nuebarbgr_lowe.txt"),
        nubar_mu_prediction=load("miniboone_numubar.txt"),
        nubar_full_transmutation=load("miniboone_nubarfullosc_ntuple.txt"),
        fractional_covariance=load("miniboone_full_fractcovmatrix_combined_lowe.txt"),
    )
    _validate(result)
    return result


def _validate(data: MiniBooNE2020Data) -> None:
    if data.electron_edges_MeV.shape != (12,):
        raise ValueError("MiniBooNE electron-like sample must have 11 bins")
    for name in ("nu_e_data", "nu_e_background", "nubar_e_data", "nubar_e_background"):
        if getattr(data, name).shape != (11,):
            raise ValueError(f"{name} must have 11 bins")
    for name in ("nu_mu_data", "nu_mu_prediction", "nubar_mu_data", "nubar_mu_prediction"):
        if getattr(data, name).shape != (8,):
            raise ValueError(f"{name} must have 8 bins")
    if data.nu_full_transmutation.shape != (17204, 4):
        raise ValueError("unexpected neutrino-mode full-transmutation table")
    if data.nubar_full_transmutation.shape != (117949, 4):
        raise ValueError("unexpected antineutrino-mode full-transmutation table")
    if data.fractional_covariance.shape != (60, 60):
        raise ValueError("combined MiniBooNE fractional covariance must be 60x60")
    if not np.allclose(data.fractional_covariance, data.fractional_covariance.T, atol=1e-10):
        raise ValueError("MiniBooNE fractional covariance is not symmetric")


def three_plus_one_parameters_from_appearance_amplitude(
    delta_m2_41_eV2: float,
    sin2_2theta_mue: float,
) -> ThreePlusOneParameters:
    """Choose one representative of the appearance-only mixing degeneracy.

    Setting sin²theta14=1/2 and sin²theta24=sin²(2theta_mue) makes the core's
    exact 4|Ue4|²|Umu4|² equal the requested scan coordinate.  MiniBooNE's
    public two-flavour release cannot distinguish other factorizations.
    """
    return ThreePlusOneParameters(delta_m2_41_eV2, 0.5, sin2_2theta_mue)


def three_plus_one_appearance_probability(
    true_energy_MeV: NDArray[np.float64], baseline_cm: NDArray[np.float64],
    delta_m2_eV2: float, sin2_2theta: float,
) -> NDArray[np.float64]:
    parameters = three_plus_one_parameters_from_appearance_amplitude(
        delta_m2_eV2, sin2_2theta
    )
    return short_baseline_appearance_probability(
        parameters,
        np.asarray(true_energy_MeV, dtype=float) / 1000.0,
        np.asarray(baseline_cm, dtype=float) / 100000.0,
    )


def signal_counts(events: NDArray[np.float64], edges_MeV: NDArray[np.float64],
                  delta_m2_eV2: float, sin2_2theta: float) -> NDArray[np.float64]:
    probability = three_plus_one_appearance_probability(
        events[:, 1], events[:, 2], delta_m2_eV2, sin2_2theta
    )
    # Official convention: add P(Etrue,Ltrue)*weight/N into its E_QE bin.
    return np.histogram(
        events[:, 0], bins=edges_MeV, weights=probability * events[:, 3] / len(events)
    )[0]


def prediction_and_covariance_from_signals(
    data: MiniBooNE2020Data,
    signal_nu: NDArray[np.float64],
    signal_nubar: NDArray[np.float64],
):
    """Build the released 38-bin prediction/covariance from two signal vectors."""
    six_block_prediction = np.r_[
        signal_nu, data.nu_e_background, data.nu_mu_prediction,
        signal_nubar, data.nubar_e_background, data.nubar_mu_prediction,
    ]
    full_covariance = data.fractional_covariance * np.outer(
        six_block_prediction, six_block_prediction
    )
    collapse = np.zeros((38, 60), dtype=float)
    collapse[0:11, 0:11] = np.eye(11)
    collapse[0:11, 11:22] = np.eye(11)
    collapse[11:19, 22:30] = np.eye(8)
    collapse[19:30, 30:41] = np.eye(11)
    collapse[19:30, 41:52] = np.eye(11)
    collapse[30:38, 52:60] = np.eye(8)
    covariance = collapse @ full_covariance @ collapse.T
    # The released matrix contains signal systematics but not signal statistics.
    covariance[np.arange(11), np.arange(11)] += signal_nu
    covariance[np.arange(19, 30), np.arange(19, 30)] += signal_nubar
    prediction = np.r_[
        signal_nu + data.nu_e_background, data.nu_mu_prediction,
        signal_nubar + data.nubar_e_background, data.nubar_mu_prediction,
    ]
    observation = np.r_[data.nu_e_data, data.nu_mu_data,
                        data.nubar_e_data, data.nubar_mu_data]
    return observation, prediction, covariance, signal_nu, signal_nubar


def prediction_and_covariance(data: MiniBooNE2020Data, delta_m2_eV2: float,
                              sin2_2theta: float):
    signal_nu = signal_counts(data.nu_full_transmutation, data.electron_edges_MeV,
                              delta_m2_eV2, sin2_2theta)
    signal_nubar = signal_counts(data.nubar_full_transmutation, data.electron_edges_MeV,
                                 delta_m2_eV2, sin2_2theta)
    return prediction_and_covariance_from_signals(data, signal_nu, signal_nubar)


def gaussian_negative_two_log_likelihood_from_signals(
    data: MiniBooNE2020Data,
    signal_nu: NDArray[np.float64],
    signal_nubar: NDArray[np.float64],
) -> float:
    observation, prediction, covariance, _, _ = prediction_and_covariance_from_signals(
        data, signal_nu, signal_nubar
    )
    sign, logdet = np.linalg.slogdet(covariance)
    if sign <= 0:
        raise np.linalg.LinAlgError("MiniBooNE covariance is not positive definite")
    residual = observation - prediction
    return float(residual @ np.linalg.solve(covariance, residual) + logdet)


def gaussian_negative_two_log_likelihood(data: MiniBooNE2020Data,
                                         delta_m2_eV2: float,
                                         sin2_2theta: float) -> float:
    observation, prediction, covariance, signal_nu, signal_nubar = prediction_and_covariance(
        data, delta_m2_eV2, sin2_2theta
    )
    return gaussian_negative_two_log_likelihood_from_signals(data, signal_nu, signal_nubar)
