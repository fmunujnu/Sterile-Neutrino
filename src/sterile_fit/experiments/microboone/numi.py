"""experiments/microboone/numi.py: regrouped existing implementations; see docs/ARCHITECTURE.md."""
from __future__ import annotations
from sterile_fit.output import result_directory

from dataclasses import dataclass
import json
from pathlib import Path
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sterile_fit.core.three_plus_one import VacuumOscillationModel
from sterile_fit.core.three_plus_one import ThreePlusOneVacuumModel
from sterile_fit.core.three_plus_one import ThreePlusOneParameters
from sterile_fit.core.likelihood import PredictionScaledGaussianLikelihood
from sterile_fit.experiments.microboone.public_data import PublishedNumiFourChannelInputs, load_numi_four_channel_inputs
from hashlib import sha256
from sterile_fit.experiments.microboone.public_data import NUMI_FOUR_CHANNELS
from sterile_fit.experiments.microboone.public_data import DEFAULT_COVARIANCE_PATH, DEFAULT_SPECTRUM_PATH, load_numi_four_channel_inputs
# NuMI four-channel empirical event reweighting, isolated from BNB code.


PROCESS_FIELDS = (
    "beam_nue_to_nue_cc_response_counts",
    "beam_numu_to_nue_cc_response_counts",
    "beam_nue_to_numu_cc_response_counts",
    "beam_numu_to_numu_cc_response_counts",
    "beam_nuebar_to_nuebar_cc_response_counts",
    "beam_numubar_to_nuebar_cc_response_counts",
    "beam_nuebar_to_numubar_cc_response_counts",
    "beam_numubar_to_numubar_cc_response_counts",
)


@dataclass(frozen=True, slots=True)
class NumiFourChannelEmpiricalKernel:
    """Eight source-to-selected-flavour event kernels on 60 true-energy bins.

    These arrays are an algebraic reference-ratio construction, not measured
    cross sections or efficiencies.  The fixed published Background category is
    added exactly once and is not reweighted because its component templates are
    absent from the public release.
    """

    true_energy_GeV: NDArray[np.float64]
    fixed_published_background_counts: NDArray[np.float64]
    beam_nue_to_nue_cc_response_counts: NDArray[np.float64]
    beam_numu_to_nue_cc_response_counts: NDArray[np.float64]
    beam_nue_to_numu_cc_response_counts: NDArray[np.float64]
    beam_numu_to_numu_cc_response_counts: NDArray[np.float64]
    beam_nuebar_to_nuebar_cc_response_counts: NDArray[np.float64]
    beam_numubar_to_nuebar_cc_response_counts: NDArray[np.float64]
    beam_nuebar_to_numubar_cc_response_counts: NDArray[np.float64]
    beam_numubar_to_numubar_cc_response_counts: NDArray[np.float64]

    def __post_init__(self) -> None:
        energy = np.asarray(self.true_energy_GeV, dtype=float)
        if energy.shape != (60,) or np.any(energy <= 0.0) or not np.all(np.diff(energy) > 0.0):
            raise ValueError("true_energy_GeV must be the increasing 60-bin Reco grid")
        background = np.asarray(self.fixed_published_background_counts, dtype=float)
        if background.shape != (104,) or np.any(background < 0.0):
            raise ValueError("fixed_published_background_counts must be non-negative with shape (104,)")
        for name in PROCESS_FIELDS:
            values = np.asarray(getattr(self, name), dtype=float)
            if values.shape != (104, 60) or not np.all(np.isfinite(values)) or np.any(values < 0.0):
                raise ValueError(f"{name} must be finite and non-negative with shape (104, 60)")

    @classmethod
    def from_directory(cls, directory: Path) -> "NumiFourChannelEmpiricalKernel":
        """Load the visible BNB-style NuMI kernel contract."""
        directory = Path(directory)
        metadata_path = directory / "metadata.json"
        energy_path = directory / "true_energy_GeV.csv"
        background_path = directory / "fixed_published_background_counts.csv"
        required = [metadata_path, energy_path, background_path]
        required.extend(directory / f"{name}.csv" for name in PROCESS_FIELDS)
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise ValueError(f"NuMI kernel is missing visible files: {missing}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("format") != "numi_four_channel_empirical_kernel_v1":
            raise ValueError("NuMI kernel metadata format is not recognized")
        energy_table = pd.read_csv(energy_path)
        if list(energy_table.columns) != ["true_bin", "true_energy_GeV"]:
            raise ValueError("NuMI true-energy CSV has unexpected columns")
        if not np.array_equal(energy_table["true_bin"].to_numpy(dtype=int), np.arange(60)):
            raise ValueError("NuMI true-energy bins must be contiguous 0..59")
        background_table = pd.read_csv(background_path)
        if list(background_table.columns) != [
            "local_numi_reco_bin",
            "fixed_published_background_counts",
        ]:
            raise ValueError("NuMI fixed-background CSV has unexpected columns")
        if not np.array_equal(
            background_table["local_numi_reco_bin"].to_numpy(dtype=int), np.arange(104)
        ):
            raise ValueError("NuMI fixed-background bins must be contiguous 0..103")
        values: dict[str, NDArray[np.float64]] = {
            "true_energy_GeV": energy_table["true_energy_GeV"].to_numpy(dtype=float),
            "fixed_published_background_counts": background_table[
                "fixed_published_background_counts"
            ].to_numpy(dtype=float),
        }
        expected_columns = [
            "local_numi_reco_bin",
            *[f"true_bin_{index:03d}" for index in range(60)],
        ]
        for name in PROCESS_FIELDS:
            table = pd.read_csv(directory / f"{name}.csv")
            if list(table.columns) != expected_columns or table.shape != (104, 61):
                raise ValueError(f"{name}.csv has an unexpected visible schema")
            if not np.array_equal(
                table["local_numi_reco_bin"].to_numpy(dtype=int), np.arange(104)
            ):
                raise ValueError(f"{name}.csv local bins must be contiguous 0..103")
            values[name] = table.iloc[:, 1:].to_numpy(dtype=float)
        return cls(**values)

    def component_counts(
        self, model: VacuumOscillationModel, baseline_km: float
    ) -> dict[str, NDArray[np.float64]]:
        energy = self.true_energy_GeV
        probability = model.probability
        return {
            "nue_to_nue": self.beam_nue_to_nue_cc_response_counts
            @ probability(0, 0, energy, baseline_km),
            "numu_to_nue": self.beam_numu_to_nue_cc_response_counts
            @ probability(1, 0, energy, baseline_km),
            "nue_to_numu": self.beam_nue_to_numu_cc_response_counts
            @ probability(0, 1, energy, baseline_km),
            "numu_to_numu": self.beam_numu_to_numu_cc_response_counts
            @ probability(1, 1, energy, baseline_km),
            "nuebar_to_nuebar": self.beam_nuebar_to_nuebar_cc_response_counts
            @ probability(0, 0, energy, baseline_km, antineutrino=True),
            "numubar_to_nuebar": self.beam_numubar_to_nuebar_cc_response_counts
            @ probability(1, 0, energy, baseline_km, antineutrino=True),
            "nuebar_to_numubar": self.beam_nuebar_to_numubar_cc_response_counts
            @ probability(0, 1, energy, baseline_km, antineutrino=True),
            "numubar_to_numubar": self.beam_numubar_to_numubar_cc_response_counts
            @ probability(1, 1, energy, baseline_km, antineutrino=True),
        }

    def predict_total_counts(
        self, model: VacuumOscillationModel, baseline_km: float
    ) -> NDArray[np.float64]:
        components = self.component_counts(model, baseline_km)
        prediction = self.fixed_published_background_counts + sum(components.values())
        if not np.all(np.isfinite(prediction)) or np.any(prediction < 0.0):
            raise FloatingPointError("NuMI event prediction contains invalid counts")
        return prediction


# Parameter-dependent prediction for the diagnostic NuMI four-channel adapter.


@dataclass(frozen=True, slots=True)
class NumiFourChannelPredictor:
    kernel: NumiFourChannelEmpiricalKernel
    baseline_km: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.baseline_km) or self.baseline_km <= 0.0:
            raise ValueError("NuMI baseline_km must be finite and positive")

    def predict_total_counts(self, parameters: ThreePlusOneParameters) -> NDArray[np.float64]:
        model = ThreePlusOneVacuumModel(parameters)
        return self.kernel.predict_total_counts(model, self.baseline_km)

    def validate_reference(
        self,
        parameters: ThreePlusOneParameters,
        published_total_prediction_counts: NDArray[np.float64],
    ) -> None:
        predicted = self.predict_total_counts(parameters)
        published = np.asarray(published_total_prediction_counts, dtype=float)
        if published.shape != (104,):
            raise ValueError("NuMI published reference must have shape (104,)")
        if not np.allclose(predicted, published, rtol=1e-10, atol=1e-10):
            largest = int(np.argmax(np.abs(predicted - published)))
            raise ValueError(f"NuMI reference closure failed at local bin {largest}")


# Disabled-by-default NuMI workflow used by explicit diagnostic scripts.


@dataclass(frozen=True, slots=True)
class DiagnosticNumiWorkflow:
    inputs: PublishedNumiFourChannelInputs
    predictor: NumiFourChannelPredictor
    likelihood: PredictionScaledGaussianLikelihood


def build_diagnostic_numi_workflow(
    kernel_directory: Path,
    reference_parameters: ThreePlusOneParameters,
    baseline_km: float,
) -> DiagnosticNumiWorkflow:
    """Build the explicit approximation without registering a production likelihood."""
    inputs = load_numi_four_channel_inputs()
    kernel = NumiFourChannelEmpiricalKernel.from_directory(kernel_directory)
    predictor = NumiFourChannelPredictor(kernel, baseline_km)
    predictor.validate_reference(reference_parameters, inputs.published_total_prediction_counts)
    return DiagnosticNumiWorkflow(
        inputs=inputs,
        predictor=predictor,
        likelihood=PredictionScaledGaussianLikelihood(
            inputs.observed_counts,
            inputs.published_total_prediction_counts,
            inputs.systematic_covariance,
        ),
    )


# Build four exposure-averaged, oscillation-weighted NuMI flavour flux CSVs.


from sterile_fit.paths import REPOSITORY_ROOT as FLUX_ROOT
FLUX_INPUT_DIRECTORY = FLUX_ROOT / "data" / "experiments" / "microboone" / "numi" / "inputs" / "flux_components"
FLUX_OUTPUT_DIRECTORY = FLUX_ROOT / "data" / "experiments" / "microboone" / "numi" / "derived" / "paper_figure3_weighted_flux"

FLUX_DELTA_M2_41_EV2 = 1.2
FLUX_SIN2_2THETA_MUE = 0.003
FLUX_SIN2_THETA24_VALUES = (0.018, 0.0045)
FLUX_NUMI_BASELINE_KM = 0.680
FLUX_TOTAL_EXPOSURE_POT = 10.54e20
FLUX_FHC_EXPOSURE_FRACTION = 0.308
FLUX_RHC_EXPOSURE_FRACTION = 0.692
FLUX_FLAVOURS = ("numu", "numubar", "nue", "nuebar")


def flux_parameters(sin2_theta24: float) -> ThreePlusOneParameters:
    """Convert the paper's effective angle on its stated small-theta14 branch."""

    sin2_2theta14 = FLUX_SIN2_2THETA_MUE / sin2_theta24
    if not 0.0 <= sin2_2theta14 <= 1.0:
        raise ValueError("sin2(2theta_mue)/sin2(theta24) is outside [0, 1]")
    sin2_theta14 = (1.0 - np.sqrt(1.0 - sin2_2theta14)) / 2.0
    parameters = ThreePlusOneParameters(
        delta_m2_41_eV2=FLUX_DELTA_M2_41_EV2,
        sin2_theta14=float(sin2_theta14),
        sin2_theta24=sin2_theta24,
    )
    if not np.isclose(parameters.sin2_2theta_mue_exact, FLUX_SIN2_2THETA_MUE, rtol=1e-12, atol=1e-15):
        raise RuntimeError("paper effective-angle conversion failed its exact-amplitude check")
    return parameters


def flux_load_fluxes() -> tuple[np.ndarray, dict[str, dict[str, np.ndarray]], dict[str, str]]:
    fluxes: dict[str, dict[str, np.ndarray]] = {"fhc": {}, "rhc": {}}
    hashes: dict[str, str] = {}
    reference_edges: tuple[np.ndarray, np.ndarray] | None = None
    centers: np.ndarray | None = None
    for horn_mode in ("fhc", "rhc"):
        for flavour in FLUX_FLAVOURS:
            path = FLUX_INPUT_DIRECTORY / f"numi_{horn_mode}_{flavour}_flux.csv"
            table = pd.read_csv(path)
            required = {
                "energy_low_GeV",
                "energy_high_GeV",
                "energy_center_GeV",
                "flux_per_POT_per_cm2_per_100MeV",
                "is_censored",
            }
            if not required.issubset(table.columns) or table.shape[0] != 50:
                raise ValueError(f"unexpected NuMI flux input: {path}")
            low = table["energy_low_GeV"].to_numpy(dtype=float)
            high = table["energy_high_GeV"].to_numpy(dtype=float)
            current_centers = table["energy_center_GeV"].to_numpy(dtype=float)
            if reference_edges is None:
                reference_edges = low, high
                centers = current_centers
            elif not (
                np.allclose(low, reference_edges[0], atol=1e-12)
                and np.allclose(high, reference_edges[1], atol=1e-12)
                and np.allclose(current_centers, centers, atol=1e-12)
            ):
                raise ValueError("all eight NuMI flux inputs must use exactly the same grid")
            values = table["flux_per_POT_per_cm2_per_100MeV"].to_numpy(dtype=float)
            if not np.all(np.isfinite(values)) or np.any(values < 0.0):
                raise ValueError(f"invalid flux values: {path}")
            if table["is_censored"].astype(bool).any():
                raise ValueError(f"censored input cannot be used for oscillation weighting: {path}")
            fluxes[horn_mode][flavour] = values
            hashes[path.name] = sha256(path.read_bytes()).hexdigest().upper()
    assert centers is not None
    return centers, fluxes, hashes


def flux_oscillated_mode_flux(
    source: dict[str, np.ndarray],
    energy_GeV: np.ndarray,
    parameters: ThreePlusOneParameters,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    model = ThreePlusOneVacuumModel(parameters)
    probabilities = {
        "nue_to_nue": model.probability(0, 0, energy_GeV, FLUX_NUMI_BASELINE_KM),
        "numu_to_nue": model.probability(1, 0, energy_GeV, FLUX_NUMI_BASELINE_KM),
        "nue_to_numu": model.probability(0, 1, energy_GeV, FLUX_NUMI_BASELINE_KM),
        "numu_to_numu": model.probability(1, 1, energy_GeV, FLUX_NUMI_BASELINE_KM),
        "nuebar_to_nuebar": model.probability(
            0, 0, energy_GeV, FLUX_NUMI_BASELINE_KM, antineutrino=True
        ),
        "numubar_to_nuebar": model.probability(
            1, 0, energy_GeV, FLUX_NUMI_BASELINE_KM, antineutrino=True
        ),
        "nuebar_to_numubar": model.probability(
            0, 1, energy_GeV, FLUX_NUMI_BASELINE_KM, antineutrino=True
        ),
        "numubar_to_numubar": model.probability(
            1, 1, energy_GeV, FLUX_NUMI_BASELINE_KM, antineutrino=True
        ),
    }
    output = {
        "nue": source["nue"] * probabilities["nue_to_nue"]
        + source["numu"] * probabilities["numu_to_nue"],
        "numu": source["numu"] * probabilities["numu_to_numu"]
        + source["nue"] * probabilities["nue_to_numu"],
        "nuebar": source["nuebar"] * probabilities["nuebar_to_nuebar"]
        + source["numubar"] * probabilities["numubar_to_nuebar"],
        "numubar": source["numubar"] * probabilities["numubar_to_numubar"]
        + source["nuebar"] * probabilities["nuebar_to_numubar"],
    }
    return output, probabilities


def prepare_numi_flux() -> None:
    FLUX_OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    energy, source_flux, input_hashes = flux_load_fluxes()
    low = energy - 0.05
    high = energy + 0.05
    no_oscillation = {
        flavour: FLUX_FHC_EXPOSURE_FRACTION * source_flux["fhc"][flavour]
        + FLUX_RHC_EXPOSURE_FRACTION * source_flux["rhc"][flavour]
        for flavour in FLUX_FLAVOURS
    }

    weighted_by_point: dict[float, dict[str, np.ndarray]] = {}
    parameter_metadata = []
    for sin2_theta24 in FLUX_SIN2_THETA24_VALUES:
        parameters = flux_parameters(sin2_theta24)
        fhc_output, _ = flux_oscillated_mode_flux(source_flux["fhc"], energy, parameters)
        rhc_output, _ = flux_oscillated_mode_flux(source_flux["rhc"], energy, parameters)
        weighted_by_point[sin2_theta24] = {
            flavour: FLUX_FHC_EXPOSURE_FRACTION * fhc_output[flavour]
            + FLUX_RHC_EXPOSURE_FRACTION * rhc_output[flavour]
            for flavour in FLUX_FLAVOURS
        }
        parameter_metadata.append(
            {
                "delta_m2_41_eV2": parameters.delta_m2_41_eV2,
                "labelled_sin2_2theta_mue": FLUX_SIN2_2THETA_MUE,
                "sin2_theta14": parameters.sin2_theta14,
                "sin2_theta24": parameters.sin2_theta24,
                "sin2_2theta_mue_exact": parameters.sin2_2theta_mue_exact,
            }
        )

    final_files = []
    for flavour in FLUX_FLAVOURS:
        table = pd.DataFrame(
            {
                "energy_low_GeV": low,
                "energy_high_GeV": high,
                "energy_center_GeV": energy,
                "no_oscillation_exposure_weighted_flux_per_POT_per_cm2_per_100MeV": no_oscillation[flavour],
                "paper_sin2_theta24_0p018_flux_per_POT_per_cm2_per_100MeV": weighted_by_point[0.018][flavour],
                "paper_sin2_theta24_0p0045_flux_per_POT_per_cm2_per_100MeV": weighted_by_point[0.0045][flavour],
            }
        )
        if not np.all(np.isfinite(table.iloc[:, 3:].to_numpy(dtype=float))):
            raise RuntimeError(f"non-finite weighted flux for {flavour}")
        if np.any(table.iloc[:, 3:].to_numpy(dtype=float) < 0.0):
            raise RuntimeError(f"negative weighted flux for {flavour}")
        destination = FLUX_OUTPUT_DIRECTORY / f"numi_exposure_weighted_{flavour}_flux.csv"
        visible_table = table.copy()
        visible_table["energy_low_GeV"] = visible_table["energy_low_GeV"].map(lambda value: f"{value:.1f}")
        visible_table["energy_high_GeV"] = visible_table["energy_high_GeV"].map(lambda value: f"{value:.1f}")
        visible_table["energy_center_GeV"] = visible_table["energy_center_GeV"].map(lambda value: f"{value:.2f}")
        visible_table.to_csv(destination, index=False, float_format="%.17g")
        final_files.append(str(destination))

    metadata = {
        "status": "diagnostic_flux_level_only",
        "final_file_count": len(final_files),
        "final_files": final_files,
        "input_flux_sha256": input_hashes,
        "flux_unit": "neutrinos / POT / cm^2 / 100 MeV",
        "energy_grid": {"minimum_GeV": 0.0, "maximum_GeV": 5.0, "bin_width_GeV": 0.1, "bins": 50},
        "exposure_weighting": {
            "total_NuMI_POT": FLUX_TOTAL_EXPOSURE_POT,
            "FHC_fraction": FLUX_FHC_EXPOSURE_FRACTION,
            "RHC_fraction": FLUX_RHC_EXPOSURE_FRACTION,
            "formula": "combined flux per POT = 0.308 * FHC flux per POT + 0.692 * RHC flux per POT",
        },
        "oscillation_weighting": {
            "baseline_km": FLUX_NUMI_BASELINE_KM,
            "baseline_treatment": "single target-to-detector baseline approximation",
            "parameter_points": parameter_metadata,
            "final_e_formula": "Phi_e * P(e->e) + Phi_mu * P(mu->e)",
            "final_mu_formula": "Phi_mu * P(mu->mu) + Phi_e * P(e->mu)",
            "antineutrino_formula": "same construction using antineutrino probabilities",
        },
        "paper_sources": {
            "parameter_note": "MICROBOONE-NOTE-1132-PUB Figure 3 discussion",
            "parameter_note_url": "https://microboone.fnal.gov/wp-content/uploads/MICROBOONE-NOTE-1132-PUB.pdf",
            "exposure_paper_doi": "10.1038/s41586-025-09757-7",
        },
        "scientific_boundary": (
            "The input fluxes are recovered from displayed PDF vector paths. The fixed 0.680 km baseline "
            "does not integrate over the NuMI decay-position distribution. These files are diagnostic flux-level "
            "weights, not detector-folded event predictions and not inputs that validate a NuMI likelihood."
        ),
    }
    (FLUX_OUTPUT_DIRECTORY / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({"status": "pass", "files": final_files}, indent=2, ensure_ascii=True))


# Build isolated NuMI four-channel empirical kernels and per-bin events.


from sterile_fit.paths import REPOSITORY_ROOT as EVENTS_ROOT
EVENTS_BNB_RESPONSE_DIRECTORY = (
    EVENTS_ROOT
    / "data"
    / "experiments"
    / "microboone"
    / "bnb"
    / "derived"
    / "archival_2022_reco_bnb26_given_true"
)
EVENTS_NUMI_FLUX_DIRECTORY = (
    EVENTS_ROOT
    / "data"
    / "experiments"
    / "microboone"
    / "numi"
    / "derived"
    / "paper_figure3_weighted_flux"
)
EVENTS_NUMI_DATA_DIRECTORY = (
    EVENTS_ROOT
    / "data"
    / "experiments"
    / "microboone"
    / "numi"
)
EVENTS_KERNEL_DIRECTORY = EVENTS_NUMI_DATA_DIRECTORY / "reweighting"
EVENTS_DERIVED_DIRECTORY = EVENTS_NUMI_DATA_DIRECTORY / "derived"

EVENTS_NUMI_BASELINE_KM = 0.680
EVENTS_DELTA_M2_41_EV2 = 1.2
EVENTS_SIN2_2THETA_MUE = 0.003
EVENTS_SIN2_THETA24_VALUES = (0.018, 0.0045)
EVENTS_FLAVOURS = ("nue", "numu", "nuebar", "numubar")
EVENTS_PROCESS_BY_SOURCE_AND_FINAL = {
    ("nue", "nue"): "beam_nue_to_nue_cc_response_counts",
    ("numu", "nue"): "beam_numu_to_nue_cc_response_counts",
    ("nue", "numu"): "beam_nue_to_numu_cc_response_counts",
    ("numu", "numu"): "beam_numu_to_numu_cc_response_counts",
    ("nuebar", "nue"): "beam_nuebar_to_nuebar_cc_response_counts",
    ("numubar", "nue"): "beam_numubar_to_nuebar_cc_response_counts",
    ("nuebar", "numu"): "beam_nuebar_to_numubar_cc_response_counts",
    ("numubar", "numu"): "beam_numubar_to_numubar_cc_response_counts",
}


def events_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest().upper()


def events_load_response(channel_identifier: str) -> np.ndarray:
    path = EVENTS_BNB_RESPONSE_DIRECTORY / f"{channel_identifier}_reco_given_true.csv"
    table = pd.read_csv(path)
    expected_columns = ["reco_bin", *[f"true_bin_{index:03d}" for index in range(60)]]
    if list(table.columns) != expected_columns or table.shape != (26, 61):
        raise ValueError(f"unexpected 26x60 response schema: {path}")
    if not np.array_equal(table["reco_bin"].to_numpy(dtype=int), np.arange(26)):
        raise ValueError(f"reco_bin must be 0..25 in {path}")
    response = table.iloc[:, 1:].to_numpy(dtype=float)
    if np.any(response < 0.0) or not np.all(np.isfinite(response)):
        raise ValueError(f"invalid response values: {path}")
    column_sums = response.sum(axis=0)
    if not np.all(np.isclose(column_sums, 1.0, atol=1e-12) | np.isclose(column_sums, 0.0, atol=1e-12)):
        raise ValueError(f"response columns are neither normalized nor zero: {path}")
    return response


def events_load_true_energy() -> np.ndarray:
    path = EVENTS_BNB_RESPONSE_DIRECTORY / "true_energy_GeV.csv"
    table = pd.read_csv(path)
    if list(table.columns) != ["true_bin", "true_energy_GeV"] or table.shape != (60, 2):
        raise ValueError("Reco true-energy grid must contain 60 visible bins")
    if not np.array_equal(table["true_bin"].to_numpy(dtype=int), np.arange(60)):
        raise ValueError("Reco true_bin must be 0..59")
    energy = table["true_energy_GeV"].to_numpy(dtype=float)
    if not np.allclose(energy, 0.025 + 0.05 * np.arange(60), atol=1e-12):
        raise ValueError("expected the declared 0--3 GeV, 0.05 GeV Reco grid")
    return energy


def events_load_source_flux_on_reco_grid(true_energy_GeV: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    """Conservatively split each 0.1 GeV flux bin into two 0.05 GeV bins."""
    output: dict[str, np.ndarray] = {}
    hashes: dict[str, str] = {}
    column = "no_oscillation_exposure_weighted_flux_per_POT_per_cm2_per_100MeV"
    for flavour in EVENTS_FLAVOURS:
        path = EVENTS_NUMI_FLUX_DIRECTORY / f"numi_exposure_weighted_{flavour}_flux.csv"
        table = pd.read_csv(path)
        if table.shape[0] != 50 or column not in table.columns:
            raise ValueError(f"unexpected NuMI flux schema: {path}")
        low = table["energy_low_GeV"].to_numpy(dtype=float)
        high = table["energy_high_GeV"].to_numpy(dtype=float)
        values = table[column].to_numpy(dtype=float)
        selected = np.empty(60, dtype=float)
        for index, energy in enumerate(true_energy_GeV):
            matches = np.flatnonzero((low <= energy) & (energy < high))
            if matches.size != 1:
                raise ValueError(f"NuMI flux bin is not unique at E={energy:.6g} GeV")
            # Input is per 100 MeV; each Reco true bin is 50 MeV wide.
            selected[index] = 0.5 * values[matches[0]]
        if not np.all(np.isfinite(selected)) or np.any(selected < 0.0):
            raise ValueError(f"invalid resampled flux: {path}")
        output[flavour] = selected
        hashes[path.name] = events_sha256(path)
    return output, hashes


def events_paper_parameters(sin2_theta24: float) -> ThreePlusOneParameters:
    sin2_2theta14 = EVENTS_SIN2_2THETA_MUE / sin2_theta24
    sin2_theta14 = (1.0 - np.sqrt(1.0 - sin2_2theta14)) / 2.0
    parameters = ThreePlusOneParameters(
        delta_m2_41_eV2=EVENTS_DELTA_M2_41_EV2,
        sin2_theta14=float(sin2_theta14),
        sin2_theta24=sin2_theta24,
    )
    if not np.isclose(parameters.sin2_2theta_mue_exact, EVENTS_SIN2_2THETA_MUE, rtol=1e-12, atol=1e-15):
        raise RuntimeError("paper effective-angle conversion failed")
    return parameters


def events_write_matrix(path: Path, matrix: np.ndarray, first_column: str) -> None:
    table = pd.DataFrame(matrix, columns=[f"true_bin_{index:03d}" for index in range(matrix.shape[1])])
    table.insert(0, first_column, np.arange(matrix.shape[0], dtype=int))
    table.to_csv(path, index=False, float_format="%.17g")


def prepare_numi_kernel() -> None:
    EVENTS_EVENT_OUTPUT = result_directory("microboone_numi", "three_plus_one", "prepared_events") / "paper_parameter_event_counts.csv"
    EVENTS_KERNEL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    EVENTS_DERIVED_DIRECTORY.mkdir(parents=True, exist_ok=True)
    EVENTS_EVENT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    inputs = load_numi_four_channel_inputs()
    true_energy = events_load_true_energy()
    source_flux, flux_hashes = events_load_source_flux_on_reco_grid(true_energy)

    reference_parameters = ThreePlusOneParameters(
        delta_m2_41_eV2=EVENTS_DELTA_M2_41_EV2,
        sin2_theta14=0.0,
        sin2_theta24=0.0,
    )
    reference_model = ThreePlusOneVacuumModel(reference_parameters)
    process_arrays = {name: np.zeros((104, 60), dtype=float) for name in PROCESS_FIELDS}

    for local_channel_index, channel in enumerate(NUMI_FOUR_CHANNELS):
        final_name = "nue" if channel.identifier.startswith("nue_") else "numu"
        final_index = 0 if final_name == "nue" else 1
        response = events_load_response(channel.identifier)
        source_probability = {
            "nue": reference_model.probability(0, final_index, true_energy, EVENTS_NUMI_BASELINE_KM),
            "numu": reference_model.probability(1, final_index, true_energy, EVENTS_NUMI_BASELINE_KM),
            "nuebar": reference_model.probability(
                0, final_index, true_energy, EVENTS_NUMI_BASELINE_KM, antineutrino=True
            ),
            "numubar": reference_model.probability(
                1, final_index, true_energy, EVENTS_NUMI_BASELINE_KM, antineutrino=True
            ),
        }
        reference_true_weight = sum(
            source_flux[source] * source_probability[source] for source in EVENTS_FLAVOURS
        )
        denominator = response @ reference_true_weight
        local_start = local_channel_index * 26
        local_stop = local_start + 26
        published_signal = inputs.published_signal_counts[local_start:local_stop]
        if np.any((denominator <= 0.0) & (published_signal > 0.0)):
            raise ValueError(f"zero response support for nonzero {channel.identifier} signal")
        scale = np.divide(
            published_signal,
            denominator,
            out=np.zeros_like(published_signal),
            where=denominator > 0.0,
        )
        for source in EVENTS_FLAVOURS:
            field = EVENTS_PROCESS_BY_SOURCE_AND_FINAL[(source, final_name)]
            process_arrays[field][local_start:local_stop, :] = (
                scale[:, None] * response * source_flux[source][None, :]
            )

    kernel = NumiFourChannelEmpiricalKernel(
        true_energy_GeV=true_energy,
        fixed_published_background_counts=inputs.published_background_counts,
        **process_arrays,
    )
    reference_prediction = kernel.predict_total_counts(reference_model, EVENTS_NUMI_BASELINE_KM)
    reference_residual = reference_prediction - inputs.published_total_prediction_counts
    if not np.allclose(reference_residual, 0.0, rtol=0.0, atol=1e-10):
        largest = int(np.argmax(np.abs(reference_residual)))
        raise RuntimeError(f"reference closure failed at local NuMI bin {largest}")

    paper_predictions: dict[float, np.ndarray] = {}
    paper_components: dict[float, dict[str, np.ndarray]] = {}
    parameter_records: list[dict[str, float]] = []
    for sin2_theta24 in EVENTS_SIN2_THETA24_VALUES:
        parameters = events_paper_parameters(sin2_theta24)
        model = ThreePlusOneVacuumModel(parameters)
        paper_components[sin2_theta24] = kernel.component_counts(model, EVENTS_NUMI_BASELINE_KM)
        paper_predictions[sin2_theta24] = (
            inputs.published_background_counts + sum(paper_components[sin2_theta24].values())
        )
        parameter_records.append(
            {
                "delta_m2_41_eV2": parameters.delta_m2_41_eV2,
                "sin2_theta14": parameters.sin2_theta14,
                "sin2_theta24": parameters.sin2_theta24,
                "sin2_2theta_mue_exact": parameters.sin2_2theta_mue_exact,
            }
        )

    rows: list[dict[str, object]] = []
    for local_channel_index, channel in enumerate(NUMI_FOUR_CHANNELS):
        for channel_reco_bin in range(26):
            local_bin = local_channel_index * 26 + channel_reco_bin
            published_bin = channel.first_published_bin + channel_reco_bin
            row: dict[str, object] = {
                "channel": channel.identifier,
                "released_channel_ordinal": channel.released_channel_ordinal,
                "channel_reco_bin": channel_reco_bin,
                "local_numi_reco_bin": local_bin,
                "published_global_bin": published_bin,
                "reco_energy_low_GeV": 0.1 * channel_reco_bin if channel_reco_bin < 25 else 2.5,
                "reco_energy_high_GeV": 0.1 * (channel_reco_bin + 1) if channel_reco_bin < 25 else 3.0,
                "is_overflow_bin": channel_reco_bin == 25,
                "observed_counts": inputs.observed_counts[local_bin],
                "observed_statistical_error_up": inputs.observed_statistical_error_up[local_bin],
                "observed_statistical_error_down": inputs.observed_statistical_error_down[local_bin],
                "published_background_counts": inputs.published_background_counts[local_bin],
                "published_signal_counts": inputs.published_signal_counts[local_bin],
                "published_total_prediction_counts": inputs.published_total_prediction_counts[local_bin],
                "empirical_reference_total_counts": reference_prediction[local_bin],
                "empirical_reference_closure_residual_counts": reference_residual[local_bin],
            }
            for sin2_theta24 in EVENTS_SIN2_THETA24_VALUES:
                label = "0p018" if sin2_theta24 == 0.018 else "0p0045"
                row[f"paper_sin2_theta24_{label}_total_counts"] = paper_predictions[sin2_theta24][local_bin]
                for component_name, values in paper_components[sin2_theta24].items():
                    row[f"paper_sin2_theta24_{label}_{component_name}_counts"] = values[local_bin]
            rows.append(row)
    pd.DataFrame(rows).to_csv(
        EVENTS_EVENT_OUTPUT,
        index=False,
        float_format="%.17g",
    )

    covariance_columns = [f"local_numi_reco_bin_{index:03d}" for index in range(104)]
    covariance_table = pd.DataFrame(inputs.systematic_covariance, columns=covariance_columns)
    covariance_table.insert(0, "local_numi_reco_bin", np.arange(104, dtype=int))
    covariance_table.to_csv(
        EVENTS_DERIVED_DIRECTORY / "numi_four_channel_systematic_covariance.csv",
        index=False,
        float_format="%.17g",
    )
    block_rows = []
    for row_channel_index, row_channel in enumerate(NUMI_FOUR_CHANNELS):
        for column_channel_index, column_channel in enumerate(NUMI_FOUR_CHANNELS):
            block_rows.append(
                {
                    "block_row": row_channel_index,
                    "block_column": column_channel_index,
                    "row_channel": row_channel.identifier,
                    "column_channel": column_channel.identifier,
                    "local_row_start_inclusive": 26 * row_channel_index,
                    "local_row_stop_exclusive": 26 * (row_channel_index + 1),
                    "local_column_start_inclusive": 26 * column_channel_index,
                    "local_column_stop_exclusive": 26 * (column_channel_index + 1),
                    "numerical_block_shape": "26x26",
                }
            )
    pd.DataFrame(block_rows).to_csv(
        EVENTS_DERIVED_DIRECTORY / "numi_four_channel_covariance_block_map.csv", index=False
    )

    pd.DataFrame(
        {"true_bin": np.arange(60, dtype=int), "true_energy_GeV": true_energy}
    ).to_csv(EVENTS_KERNEL_DIRECTORY / "true_energy_GeV.csv", index=False, float_format="%.17g")
    pd.DataFrame({
        "local_numi_reco_bin": np.arange(104, dtype=int),
        "fixed_published_background_counts": inputs.published_background_counts,
    }).to_csv(
        EVENTS_KERNEL_DIRECTORY / "fixed_published_background_counts.csv",
        index=False,
        float_format="%.17g",
    )
    for name, values in process_arrays.items():
        events_write_matrix(EVENTS_KERNEL_DIRECTORY / f"{name}.csv", values, "local_numi_reco_bin")

    closure_columns = [
        "channel",
        "released_channel_ordinal",
        "channel_reco_bin",
        "local_numi_reco_bin",
        "published_global_bin",
        "observed_counts",
        "published_background_counts",
        "published_signal_counts",
        "published_total_prediction_counts",
        "empirical_reference_total_counts",
        "empirical_reference_closure_residual_counts",
    ]
    pd.DataFrame(rows)[closure_columns].to_csv(
        EVENTS_KERNEL_DIRECTORY / "reference_closure.csv", index=False, float_format="%.17g"
    )

    metadata = {
        "status": "diagnostic_event_reweighting_only",
        "format": "numi_four_channel_empirical_kernel_v1",
        "selected_released_channel_ordinals_one_based": [8, 9, 10, 11],
        "selected_published_global_bins_inclusive": [182, 285],
        "channel_order": [channel.identifier for channel in NUMI_FOUR_CHANNELS],
        "covariance": {
            "numerical_shape": [104, 104],
            "logical_shape": "4x4 channel blocks, each block 26x26 reconstructed bins",
            "source": str(DEFAULT_COVARIANCE_PATH),
            "source_sha256": events_sha256(DEFAULT_COVARIANCE_PATH),
        },
        "spectrum": {
            "source": str(DEFAULT_SPECTRUM_PATH),
            "source_sha256": events_sha256(DEFAULT_SPECTRUM_PATH),
            "published_signal_definition": "Signal + Background minus Background",
        },
        "response": {
            "source_directory": str(EVENTS_BNB_RESPONSE_DIRECTORY),
            "shape_per_channel": [26, 60],
            "true_energy_range_GeV": [0.0, 3.0],
            "true_energy_bin_width_GeV": 0.05,
            "reuse_policy": "read-only reuse; no BNB file was modified",
        },
        "flux": {
            "source_directory": str(EVENTS_NUMI_FLUX_DIRECTORY),
            "source_sha256": flux_hashes,
            "adaptation": "each 0.1 GeV per-bin value is split equally into its two 0.05 GeV Reco-grid bins",
            "used_energy_range_GeV": [0.0, 3.0],
            "unused_flux_range_GeV": [3.0, 5.0],
        },
        "reference": {
            "baseline_km": EVENTS_NUMI_BASELINE_KM,
            "parameters": {
                "delta_m2_41_eV2": EVENTS_DELTA_M2_41_EV2,
                "sin2_theta14": 0.0,
                "sin2_theta24": 0.0,
            },
            "construction": "same per-reconstructed-bin reference-ratio algebra used by BNB",
            "closure": "exact against the selected HEPData Signal + Background vector by construction",
        },
        "paper_parameter_predictions": parameter_records,
        "scientific_limits": [
            "the HEPData total is an empirical anchor and is not asserted to be a published 3nu/null spectrum",
            "the same normalized 2022 BNB response is assumed for NuMI because no NuMI-specific response is available",
            "the response has no true-energy support above 3 GeV, so 3--5 GeV flux does not enter event reweighting",
            "unknown cross section and efficiency are absorbed into one empirical scale per reconstructed bin",
            "the aggregate published Background is frozen because component-level oscillatable templates are unavailable",
            "this artifact does not define a NuMI chi-square or a combined BNB+NuMI likelihood",
        ],
    }
    (EVENTS_KERNEL_DIRECTORY / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    covariance_metadata = {
        "status": "diagnostic_prepared_input",
        "matrix_shape": [104, 104],
        "logical_shape": "4x4 channel blocks, each block 26x26 reconstructed bins",
        "channel_order": [channel.identifier for channel in NUMI_FOUR_CHANNELS],
        "selected_published_global_bins_inclusive": [182, 285],
        "source": str(DEFAULT_COVARIANCE_PATH),
        "source_sha256": events_sha256(DEFAULT_COVARIANCE_PATH),
        "likelihood_status": "disabled",
    }
    (EVENTS_DERIVED_DIRECTORY / "numi_four_channel_systematic_covariance.metadata.json").write_text(
        json.dumps(covariance_metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    event_metadata = {
        "status": "regenerable_spectrum_output",
        "source_kernel": str(EVENTS_KERNEL_DIRECTORY),
        "source_covariance": str(EVENTS_DERIVED_DIRECTORY / "numi_four_channel_systematic_covariance.csv"),
        "parameter_points": parameter_records,
        "event_bins": 104,
        "likelihood_or_chi2_included": False,
        "scientific_limits": metadata["scientific_limits"],
    }
    EVENTS_EVENT_OUTPUT.with_suffix(".metadata.json").write_text(
        json.dumps(event_metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": "pass",
                "event_bins": 104,
                "covariance_shape": [104, 104],
                "maximum_reference_closure_residual": float(np.max(np.abs(reference_residual))),
                "kernel_directory": str(EVENTS_KERNEL_DIRECTORY),
                "event_output": str(EVENTS_EVENT_OUTPUT),
            },
            indent=2,
        )
    )



