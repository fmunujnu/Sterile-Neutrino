"""Draw the released 14-channel spectra, covariance, and visible BNB flux.

This is an input-inspection script only.  It does not build an oscillation
prediction, evaluate a likelihood, or modify any scientific input.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from sterile_fit.experiments.microboone.bnb.published_inputs import (  # noqa: E402
    read_full_systematic_covariance,
    read_full_unconstrained_spectrum,
)


SPECTRUM_PATH = (
    ROOT
    / "data"
    / "experiments"
    / "microboone"
    / "shared"
    / "raw"
    / "hepdata_microboone_2025"
    / "HEPData-ins3088922-v1-Unconstrained_14_channels.csv"
)
COVARIANCE_PATH = SPECTRUM_PATH.with_name(
    "HEPData-ins3088922-v1-14_channel_covariance_matrix.csv"
)
BNB_FLUX_PATH = (
    ROOT / "data" / "experiments" / "microboone" / "bnb" / "inputs" / "bnb_flux.csv"
)
OUTPUT_DIRECTORY = ROOT / "outputs" / "spectra" / "microboone" / "public_inputs"

CHANNEL_NAMES = (
    r"$\nu_e$ CC FC",
    r"$\nu_e$ CC PC",
    r"$\nu_\mu$ CC FC",
    r"$\nu_\mu$ CC PC",
    r"$\nu_\mu$ CC $\pi^0$ FC",
    r"$\nu_\mu$ CC $\pi^0$ PC",
    r"NC $\pi^0$",
)
BEAM_NAMES = ("BNB", "NuMI")
RECO_BIN_COUNT = 26


def plot_fourteen_channel_spectra(output_path: Path) -> None:
    spectra = read_full_unconstrained_spectrum(SPECTRUM_PATH)
    data = spectra["observed_counts"]
    error_up = spectra["observed_statistical_error_up"]
    error_down = spectra["observed_statistical_error_down"]
    background = spectra["published_background_counts"]
    total = spectra["published_total_prediction_counts"]

    figure, axes = plt.subplots(7, 2, figsize=(13, 20), sharex=True, constrained_layout=True)
    x = np.arange(RECO_BIN_COUNT)
    for beam_index, beam_name in enumerate(BEAM_NAMES):
        for channel_index, channel_name in enumerate(CHANNEL_NAMES):
            global_channel = beam_index * len(CHANNEL_NAMES) + channel_index
            start = global_channel * RECO_BIN_COUNT
            stop = start + RECO_BIN_COUNT
            axis = axes[channel_index, beam_index]
            axis.step(x, total[start:stop], where="mid", color="tab:blue", label="Signal + Background")
            axis.step(x, background[start:stop], where="mid", color="tab:orange", label="Background")
            axis.errorbar(
                x,
                data[start:stop],
                yerr=np.vstack((error_down[start:stop], error_up[start:stop])),
                fmt="o",
                color="black",
                markersize=2.8,
                linewidth=0.8,
                capsize=1.5,
                label="Data",
            )
            axis.set_title(f"{beam_name}: {channel_name}")
            axis.set_ylabel("Events / bin")
            axis.grid(alpha=0.2)
            if channel_index == len(CHANNEL_NAMES) - 1:
                axis.set_xlabel("Reconstructed-energy bin (25 is overflow)")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_full_covariance(output_path: Path) -> None:
    covariance = read_full_systematic_covariance(COVARIANCE_PATH)
    scale = np.max(np.abs(covariance))
    figure, axis = plt.subplots(figsize=(10, 9), constrained_layout=True)
    image = axis.imshow(covariance, origin="lower", cmap="RdBu_r", vmin=-scale, vmax=scale)
    for boundary in range(RECO_BIN_COUNT, covariance.shape[0], RECO_BIN_COUNT):
        axis.axhline(boundary - 0.5, color="black", linewidth=0.25, alpha=0.5)
        axis.axvline(boundary - 0.5, color="black", linewidth=0.25, alpha=0.5)
    axis.axhline(7 * RECO_BIN_COUNT - 0.5, color="black", linewidth=1.2)
    axis.axvline(7 * RECO_BIN_COUNT - 0.5, color="black", linewidth=1.2)
    axis.set_xlabel("Global reconstructed-bin index")
    axis.set_ylabel("Global reconstructed-bin index")
    axis.set_title("MicroBooNE released 14-channel systematic covariance")
    figure.colorbar(image, ax=axis, label=r"Covariance [events$^2$]")
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_bnb_flux(output_path: Path) -> None:
    table = pd.read_csv(BNB_FLUX_PATH)
    required = {"true_energy_GeV", "numu_flux", "numubar_flux", "nue_flux", "nuebar_flux"}
    if not required.issubset(table.columns) or len(table) != 60:
        raise ValueError("BNB flux must contain the declared four flavours and 60 true-energy bins")

    energy = table["true_energy_GeV"].to_numpy(dtype=float)
    curves = (
        ("numu_flux", r"$\nu_\mu$"),
        ("numubar_flux", r"$\bar{\nu}_\mu$"),
        ("nue_flux", r"$\nu_e$"),
        ("nuebar_flux", r"$\bar{\nu}_e$"),
    )
    figure, axis = plt.subplots(figsize=(9, 6), constrained_layout=True)
    for column, label in curves:
        values = table[column].to_numpy(dtype=float)
        if not np.all(np.isfinite(values)) or np.any(values < 0.0):
            raise ValueError(f"invalid BNB flux values in {column}")
        axis.step(energy, values, where="mid", linewidth=1.5, label=label)
    axis.set_yscale("log")
    axis.set_xlabel(r"True neutrino energy $E_\nu$ [GeV]")
    axis.set_ylabel("Stored flux value per 0.05 GeV bin")
    axis.set_title("Visible BNB flux inputs")
    axis.grid(which="both", alpha=0.25)
    axis.legend(frameon=False, ncol=2)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def main() -> None:
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    products = {
        "14-channel spectra": OUTPUT_DIRECTORY / "microboone_14_channel_spectra.png",
        "14-channel covariance": OUTPUT_DIRECTORY / "microboone_14_channel_covariance.png",
        "BNB flux": OUTPUT_DIRECTORY / "bnb_four_flavour_flux.png",
    }
    plot_fourteen_channel_spectra(products["14-channel spectra"])
    plot_full_covariance(products["14-channel covariance"])
    plot_bnb_flux(products["BNB flux"])
    for name, path in products.items():
        print(f"WROTE {name}: {path}")


if __name__ == "__main__":
    main()
