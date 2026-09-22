"""Compare official/local NuMI-only best-fit points on one common flux model."""

from __future__ import annotations

import json
import mmap
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from sterile_fit.core.three_plus_one import ThreePlusOneParameters  # noqa: E402
from sterile_fit.experiments.microboone.numi import (  # noqa: E402
    NumiEnergyBaselineDistribution,
    build_energy_baseline_numi_workflow,
)


RAW_GRID = ROOT / "data/experiments/microboone/shared/microboone_material_gridscan_numi_dm2_t14_t24_dchi2.txt"
LOCAL_RESULTS = ROOT / "outputs/studies/numi_only_official_comparison/energy_baseline_current"
FLUX_DIRECTORY = ROOT / "data/experiments/microboone/numi/derived/paper_figure3_weighted_flux"
BASELINE_DISTRIBUTION = ROOT / "data/experiments/microboone/numi/derived/public_dk2nu_energy_baseline/psi_exposure_weighted_four_flavours.csv"
CONFIG = ROOT / "configs/experiments/microboone/numi/analysis.yaml"
OUTPUT = ROOT / "outputs/checks/microboone/numi_bestfit_flux_comparison"
FLAVOURS = ("nue", "numu", "nuebar", "numubar")


def official_best_fit() -> ThreePlusOneParameters:
    with RAW_GRID.open("rb") as handle:
        document = mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ)
        position = document.find(b"       0\n")
        if position < 0:
            raise ValueError("official grid has no exact delta-chi2 zero row")
        start = document.rfind(b"\n", 0, position) + 1
        row = document[start:document.find(b"\n", position)].decode("ascii").split()
        document.close()
    if len(row) != 7 or float(row[6]) != 0.0:
        raise ValueError(f"unexpected official minimum row: {row}")
    return ThreePlusOneParameters(float(row[3]), float(row[4]), float(row[5]))


def local_best_fit() -> ThreePlusOneParameters:
    candidates = []
    appearance = pd.read_csv(LOCAL_RESULTS / "fig3a_energy_baseline.csv")
    row = appearance.loc[appearance["local_profile_chi2"].idxmin()]
    candidates.append((float(row.local_profile_chi2), ThreePlusOneParameters(
        float(row.delta_m2_41_eV2), float(row.profiled_sin2_theta14),
        float(row.derived_sin2_theta24),
    )))
    disappearance = pd.read_csv(LOCAL_RESULTS / "fig3b_energy_baseline.csv")
    row = disappearance.loc[disappearance["local_profile_chi2"].idxmin()]
    candidates.append((float(row.local_profile_chi2), ThreePlusOneParameters(
        float(row.delta_m2_41_eV2), float(row.selected_sin2_theta14_branch),
        float(row.profiled_sin2_theta24),
    )))
    return min(candidates, key=lambda item: item[0])[1]


def load_fluxes():
    fluxes = {}
    energy = low = high = None
    column = "no_oscillation_exposure_weighted_flux_per_POT_per_cm2_per_100MeV"
    for flavour in FLAVOURS:
        table = pd.read_csv(FLUX_DIRECTORY / f"numi_exposure_weighted_{flavour}_flux.csv")
        if energy is None:
            low = table.energy_low_GeV.to_numpy(float)
            high = table.energy_high_GeV.to_numpy(float)
            energy = table.energy_center_GeV.to_numpy(float)
        fluxes[flavour] = table[column].to_numpy(float)
    return low, high, energy, fluxes


def oscillated_flux(parameters, energy, source_flux, distribution):
    probability = distribution.three_plus_one_probabilities(parameters, energy)
    return {
        "nue": source_flux["nue"] * probability["nue_to_nue"]
               + source_flux["numu"] * probability["numu_to_nue"],
        "numu": source_flux["numu"] * probability["numu_to_numu"]
                + source_flux["nue"] * probability["nue_to_numu"],
        "nuebar": source_flux["nuebar"] * probability["nuebar_to_nuebar"]
                  + source_flux["numubar"] * probability["numubar_to_nuebar"],
        "numubar": source_flux["numubar"] * probability["numubar_to_numubar"]
                   + source_flux["nuebar"] * probability["nuebar_to_numubar"],
    }, probability


def parameter_record(parameters):
    ue4 = parameters.sin2_theta14
    umu4 = (1.0 - parameters.sin2_theta14) * parameters.sin2_theta24
    return {
        "delta_m2_41_eV2": parameters.delta_m2_41_eV2,
        "sin2_theta14": parameters.sin2_theta14,
        "sin2_theta24": parameters.sin2_theta24,
        "abs_Ue4_squared": ue4,
        "abs_Umu4_squared": umu4,
        "sin2_2theta_ee": 4.0 * ue4 * (1.0 - ue4),
        "sin2_2theta_mumu": 4.0 * umu4 * (1.0 - umu4),
        "sin2_2theta_mue": 4.0 * ue4 * umu4,
    }


def main() -> None:
    official = official_best_fit()
    local = local_best_fit()
    low, high, energy, source = load_fluxes()
    distribution = NumiEnergyBaselineDistribution.from_csv(BASELINE_DISTRIBUTION)
    official_flux, official_probability = oscillated_flux(official, energy, source, distribution)
    local_flux, local_probability = oscillated_flux(local, energy, source, distribution)

    rows = []
    summaries = []
    for flavour in FLAVOURS:
        denominator = source[flavour]
        official_ratio = np.divide(
            official_flux[flavour], denominator, out=np.full_like(denominator, np.nan),
            where=denominator > 0.0,
        )
        local_ratio = np.divide(
            local_flux[flavour], denominator, out=np.full_like(denominator, np.nan),
            where=denominator > 0.0,
        )
        for index in range(len(energy)):
            rows.append({
                "flavour": flavour,
                "energy_low_GeV": low[index], "energy_high_GeV": high[index],
                "energy_center_GeV": energy[index],
                "no_oscillation_flux": denominator[index],
                "official_bestfit_flux_common_model": official_flux[flavour][index],
                "local_bestfit_flux_common_model": local_flux[flavour][index],
                "official_over_no_oscillation": official_ratio[index],
                "local_over_no_oscillation": local_ratio[index],
            })
        summaries.append({
            "flavour": flavour,
            "no_oscillation_integrated_flux": float(np.sum(denominator)),
            "official_bestfit_integrated_flux": float(np.sum(official_flux[flavour])),
            "local_bestfit_integrated_flux": float(np.sum(local_flux[flavour])),
            "official_to_local_integrated_ratio": float(
                np.sum(official_flux[flavour]) / np.sum(local_flux[flavour])
            ),
            "max_absolute_ratio_difference": float(
                np.nanmax(np.abs(official_ratio - local_ratio))
            ),
        })

    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    reference = ThreePlusOneParameters(**{
        key: float(value) for key, value in config["reference_parameters"].items()
    })
    workflow = build_energy_baseline_numi_workflow(
        ROOT / config["diagnostic_four_channel_events"]["kernel_directory"],
        reference, BASELINE_DISTRIBUTION,
    )
    detector_predictions = {
        "official": workflow.predictor.predict_total_counts(official),
        "local": workflow.predictor.predict_total_counts(local),
    }
    channel_rows = []
    for channel_index, channel in enumerate(("nue_cc_fc", "nue_cc_pc", "numu_cc_fc", "numu_cc_pc")):
        selected = slice(26 * channel_index, 26 * (channel_index + 1))
        a = detector_predictions["official"][selected]
        b = detector_predictions["local"][selected]
        channel_rows.append({
            "channel": channel,
            "official_bestfit_predicted_events": float(np.sum(a)),
            "local_bestfit_predicted_events": float(np.sum(b)),
            "difference_events": float(np.sum(a - b)),
            "maximum_absolute_bin_difference": float(np.max(np.abs(a - b))),
        })

    OUTPUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUTPUT / "four_flavour_flux_spectra.csv", index=False)
    pd.DataFrame(summaries).to_csv(OUTPUT / "integrated_flux_comparison.csv", index=False)
    pd.DataFrame(channel_rows).to_csv(OUTPUT / "detector_channel_comparison.csv", index=False)

    labels = {
        "nue": r"$\nu_e$", "numu": r"$\nu_\mu$",
        "nuebar": r"$\bar\nu_e$", "numubar": r"$\bar\nu_\mu$",
    }
    figure, axes = plt.subplots(2, 4, figsize=(18.0, 8.0), sharex="col")
    for column, flavour in enumerate(FLAVOURS):
        top, bottom = axes[:, column]
        positive = source[flavour] > 0.0
        top.plot(energy[positive], source[flavour][positive], color="black", lw=1.5, label="No oscillation")
        top.plot(energy[positive], official_flux[flavour][positive], color="tab:blue", lw=2.0, label="Official best-fit parameters")
        top.plot(energy[positive], local_flux[flavour][positive], color="tab:orange", lw=1.7, ls="--", label="Local best-fit parameters")
        top.set_yscale("log")
        top.set_title(labels[flavour])
        top.set_ylabel(r"flux / POT / cm$^2$ / 100 MeV")
        top.grid(alpha=0.2)
        official_ratio = official_flux[flavour][positive] / source[flavour][positive]
        local_ratio = local_flux[flavour][positive] / source[flavour][positive]
        bottom.plot(energy[positive], official_ratio, color="tab:blue", lw=2.0)
        bottom.plot(energy[positive], local_ratio, color="tab:orange", lw=1.7, ls="--")
        bottom.axhline(1.0, color="black", lw=0.8, alpha=0.6)
        bottom.set_xlabel(r"$E_\nu$ [GeV]")
        bottom.set_ylabel("oscillated / no oscillation")
        bottom.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=8)
    figure.suptitle("NuMI four-flavour spectra on the same public flux and q(L|E, flavour) model")
    figure.tight_layout()
    figure.savefig(OUTPUT / "four_flavour_bestfit_spectra.png", dpi=190)
    plt.close(figure)

    metadata = {
        "official_best_fit": parameter_record(official),
        "local_best_fit": parameter_record(local),
        "local_chi2_at_official_best_fit": float(
            workflow.likelihood.chi2(detector_predictions["official"])
        ),
        "local_chi2_at_local_best_fit": float(
            workflow.likelihood.chi2(detector_predictions["local"])
        ),
        "comparison_definition": "Both parameter points evaluated with the same local public flux and q(L|E,flavour); this isolates parameter-location effects and is not the unpublished official post-oscillation spectrum.",
        "official_minimum_source": str(RAW_GRID),
        "local_minimum_source": str(LOCAL_RESULTS),
        "flux_source": str(FLUX_DIRECTORY),
        "baseline_distribution": str(BASELINE_DISTRIBUTION),
    }
    (OUTPUT / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2), flush=True)
    print(pd.DataFrame(summaries).to_string(index=False), flush=True)
    print(pd.DataFrame(channel_rows).to_string(index=False), flush=True)
    print(OUTPUT, flush=True)


if __name__ == "__main__":
    main()
