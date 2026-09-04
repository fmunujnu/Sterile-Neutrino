"""experiments/microboone/bnb.py: regrouped existing implementations; see docs/ARCHITECTURE.md."""
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
from sterile_fit.core.likelihood import DeclaredTotalCovariance, load_declared_total_covariance, prediction_sha256
from sterile_fit.core.likelihood import PredictionScaledGaussianLikelihood
from sterile_fit.experiments.microboone.public_data import PublishedBnbFourChannelInputs, load_bnb_four_channel_inputs
import yaml
from sterile_fit.core.likelihood import load_declared_total_covariance
from sterile_fit.experiments.microboone.public_data import load_bnb_four_channel_inputs
import argparse
from sterile_fit.experiments.microboone.public_data import BNB_FOUR_CHANNELS
from sterile_fit.core.likelihood import cnp_total_covariance_at_reference, pearson_total_covariance_at_reference
# BNB four-channel oscillation templates.


RESPONSE_TEMPLATE_FIELDS = (
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
class BnbFourChannelOscillationTemplates:
    """Detector-folded, true-energy templates for the active BNB likelihood.

    `fixed_published_background_counts` is the released HEPData ``Background``
    category.  This analysis deliberately holds that whole category fixed; the
    name does not claim that every underlying physical component is intrinsically
    non-oscillatory.

    Each beam response array has shape `(104, n_true_energy_bins)`. Its entry
    `[reconstructed_bin, true_energy_bin]` is the selected count before the
    oscillation probability for one specified source-to-final-flavour process.
    It already contains BNB flux, interaction model, detector efficiency,
    selection and reconstructed-energy migration. Therefore these ingredients
    must not be separately multiplied later.
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
        if energy.ndim != 1 or energy.size == 0 or np.any(energy <= 0.0):
            raise ValueError("true_energy_GeV must be a non-empty positive one-dimensional array")
        if not np.all(np.diff(energy) > 0.0):
            raise ValueError("true_energy_GeV must be strictly increasing")
        fixed_background = np.asarray(self.fixed_published_background_counts, dtype=float)
        if fixed_background.shape != (104,):
            raise ValueError("fixed_published_background_counts must have shape (104,)")
        if not np.all(np.isfinite(fixed_background)) or np.any(fixed_background < 0.0):
            raise ValueError("fixed_published_background_counts must be finite and non-negative")
        for name in RESPONSE_TEMPLATE_FIELDS:
            response = np.asarray(getattr(self, name), dtype=float)
            if response.shape != (104, energy.size):
                raise ValueError(f"{name} must have shape (104, {energy.size})")
            if not np.all(np.isfinite(response)) or np.any(response < 0.0):
                raise ValueError(f"{name} must be finite and non-negative")

    @classmethod
    def from_directory(cls, directory: Path) -> "BnbFourChannelOscillationTemplates":
        """Load a fully visible CSV/JSON template directory."""
        directory = Path(directory)
        required_paths = {
            "metadata.json": directory / "metadata.json",
            "true_energy_GeV.csv": directory / "true_energy_GeV.csv",
            "fixed_published_background_counts": directory / "fixed_published_background_counts.csv",
            **{name: directory / f"{name}.csv" for name in RESPONSE_TEMPLATE_FIELDS},
        }
        missing = [name for name, path in required_paths.items() if not path.is_file()]
        if missing:
            raise ValueError(f"template directory is missing visible files: {missing}")
        metadata = json.loads(required_paths["metadata.json"].read_text(encoding="utf-8"))
        if metadata.get("format") != "bnb_four_channel_text_templates_v1":
            raise ValueError("template metadata has an unknown format")
        energy_table = pd.read_csv(required_paths["true_energy_GeV.csv"])
        if list(energy_table.columns) != ["true_bin", "true_energy_GeV"]:
            raise ValueError("true_energy_GeV.csv must contain true_bin,true_energy_GeV")
        expected_bins = np.arange(len(energy_table), dtype=int)
        if not np.array_equal(energy_table["true_bin"].to_numpy(dtype=int), expected_bins):
            raise ValueError("true_energy_GeV.csv true_bin must be contiguous and zero-based")
        fixed_table = pd.read_csv(required_paths["fixed_published_background_counts"])
        if list(fixed_table.columns) != ["global_reco_bin", "fixed_published_background_counts"]:
            raise ValueError("fixed background CSV has unexpected columns")
        if not np.array_equal(fixed_table["global_reco_bin"].to_numpy(dtype=int), np.arange(104)):
            raise ValueError("fixed background global_reco_bin must be 0..103")
        values: dict[str, NDArray[np.float64]] = {
            "true_energy_GeV": energy_table["true_energy_GeV"].to_numpy(dtype=float),
            "fixed_published_background_counts": fixed_table[
                "fixed_published_background_counts"
            ].to_numpy(dtype=float),
        }
        for name in RESPONSE_TEMPLATE_FIELDS:
            table = pd.read_csv(required_paths[name])
            expected_columns = ["global_reco_bin", *[f"true_bin_{index:03d}" for index in expected_bins]]
            if list(table.columns) != expected_columns:
                raise ValueError(f"{name}.csv has unexpected columns")
            if not np.array_equal(table["global_reco_bin"].to_numpy(dtype=int), np.arange(104)):
                raise ValueError(f"{name}.csv global_reco_bin must be 0..103")
            values[name] = table.iloc[:, 1:].to_numpy(dtype=float)
        return cls(**values)

    def to_directory(self, directory: Path, *, metadata: dict[str, object]) -> None:
        """Write every physical input as inspectable CSV plus JSON metadata."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        energy = np.asarray(self.true_energy_GeV, dtype=float)
        pd.DataFrame({
            "true_bin": np.arange(energy.size, dtype=int),
            "true_energy_GeV": energy,
        }).to_csv(directory / "true_energy_GeV.csv", index=False, float_format="%.17g")
        pd.DataFrame({
            "global_reco_bin": np.arange(104, dtype=int),
            "fixed_published_background_counts": self.fixed_published_background_counts,
        }).to_csv(
            directory / "fixed_published_background_counts.csv",
            index=False,
            float_format="%.17g",
        )
        matrix_columns = [f"true_bin_{index:03d}" for index in range(energy.size)]
        for name in RESPONSE_TEMPLATE_FIELDS:
            table = pd.DataFrame(np.asarray(getattr(self, name), dtype=float), columns=matrix_columns)
            table.insert(0, "global_reco_bin", np.arange(104, dtype=int))
            table.to_csv(directory / f"{name}.csv", index=False, float_format="%.17g")
        document = {
            **metadata,
            "format": "bnb_four_channel_text_templates_v1",
            "shape": {"reconstructed_bins": 104, "true_energy_bins": int(energy.size)},
            "reconstructed_bin_order": "nue_cc_fc,nue_cc_pc,numu_cc_fc,numu_cc_pc; 26 bins each",
        }
        (directory / "metadata.json").write_text(
            json.dumps(document, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def predict_total_counts(self, model: VacuumOscillationModel, baseline_km: float) -> NDArray[np.float64]:
        """Return BNB prediction after applying all oscillation probabilities."""
        energy = self.true_energy_GeV
        probability = model.probability
        predicted = np.asarray(self.fixed_published_background_counts, dtype=float).copy()
        predicted += self.beam_nue_to_nue_cc_response_counts @ probability(0, 0, energy, baseline_km)
        predicted += self.beam_numu_to_nue_cc_response_counts @ probability(1, 0, energy, baseline_km)
        predicted += self.beam_nue_to_numu_cc_response_counts @ probability(0, 1, energy, baseline_km)
        predicted += self.beam_numu_to_numu_cc_response_counts @ probability(1, 1, energy, baseline_km)
        predicted += self.beam_nuebar_to_nuebar_cc_response_counts @ probability(0, 0, energy, baseline_km, antineutrino=True)
        predicted += self.beam_numubar_to_nuebar_cc_response_counts @ probability(1, 0, energy, baseline_km, antineutrino=True)
        predicted += self.beam_nuebar_to_numubar_cc_response_counts @ probability(0, 1, energy, baseline_km, antineutrino=True)
        predicted += self.beam_numubar_to_numubar_cc_response_counts @ probability(1, 1, energy, baseline_km, antineutrino=True)
        if not np.all(np.isfinite(predicted)) or np.any(predicted < 0.0):
            raise FloatingPointError("template prediction contains non-finite or negative counts")
        return predicted


# Strict validation of a BNB template predictor against a published anchor.


@dataclass(frozen=True, slots=True)
class BnbFourChannelPredictor:
    """A parameter-dependent predictor built solely from declared templates."""

    templates: BnbFourChannelOscillationTemplates
    baseline_km: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.baseline_km) or self.baseline_km <= 0.0:
            raise ValueError("baseline_km must be strictly positive")

    def predict_total_counts(self, parameters: ThreePlusOneParameters) -> NDArray[np.float64]:
        return self.templates.predict_total_counts(ThreePlusOneVacuumModel(parameters), self.baseline_km)

    def validate_published_reference(
        self,
        reference_parameters: ThreePlusOneParameters,
        published_reference_total_counts: NDArray[np.float64],
        *,
        relative_tolerance: float = 1e-6,
        absolute_tolerance: float = 1e-8,
    ) -> None:
        """Require the supplied physical templates to reproduce the chosen anchor.

        No per-bin correction is applied. A mismatch is evidence that the
        template definition, reference parameters, or bin order is wrong.
        """
        published = np.asarray(published_reference_total_counts, dtype=float)
        prediction = self.predict_total_counts(reference_parameters)
        if published.shape != (104,):
            raise ValueError("published reference total must have shape (104,)")
        if not np.allclose(prediction, published, rtol=relative_tolerance, atol=absolute_tolerance):
            largest = int(np.argmax(np.abs(prediction - published)))
            raise ValueError(
                "template does not reproduce the published reference; "
                f"largest mismatch at bin {largest}: predicted={prediction[largest]:.8g}, "
                f"published={published[largest]:.8g}"
            )


# Construction of the strict, template-backed BNB four-channel workflow.


@dataclass(frozen=True, slots=True)
class StrictBnbWorkflow:
    """A BNB workflow with declared templates and declared total covariance."""

    inputs: PublishedBnbFourChannelInputs
    predictor: BnbFourChannelPredictor
    likelihood: PredictionScaledGaussianLikelihood
    statistical_treatment: str
    covariance_parameter_dependence: str


def build_strict_bnb_workflow(
    template_path: Path,
    total_covariance_path: Path,
    reference_parameters: ThreePlusOneParameters,
    baseline_km: float,
) -> StrictBnbWorkflow:
    """Build a profile-ready workflow and refuse unvalidated physical inputs."""
    inputs = load_bnb_four_channel_inputs()
    templates = BnbFourChannelOscillationTemplates.from_directory(template_path)
    predictor = BnbFourChannelPredictor(templates, baseline_km=baseline_km)
    predictor.validate_published_reference(reference_parameters, inputs.published_total_prediction_counts)
    total_covariance: DeclaredTotalCovariance = load_declared_total_covariance(total_covariance_path)
    if total_covariance.reference_prediction_sha256 != prediction_sha256(inputs.published_total_prediction_counts):
        raise ValueError("total covariance was not constructed at the published reference prediction used by this workflow")
    return StrictBnbWorkflow(
        inputs=inputs,
        predictor=predictor,
        likelihood=PredictionScaledGaussianLikelihood(
            inputs.observed_counts,
            inputs.published_total_prediction_counts,
            inputs.systematic_covariance,
        ),
        statistical_treatment=total_covariance.statistical_treatment,
        covariance_parameter_dependence="prediction_scaled_fractional_systematics_with_current_Pearson_statistics",
    )


# One-command audit of visible BNB inputs, kernels, covariance and 3+1 physics.


from sterile_fit.paths import REPOSITORY_ROOT as CHECK_ROOT
CHECK_REFERENCE_CONFIG = CHECK_ROOT / "configs" / "experiments" / "microboone" / "bnb" / "analysis.yaml"
CHECK_BNB_DATA_ROOT = CHECK_ROOT / "data" / "experiments" / "microboone" / "bnb"
CHECK_KERNEL_DIRECTORY = CHECK_BNB_DATA_ROOT / "reweighting"
CHECK_COVARIANCE_PATH = CHECK_BNB_DATA_ROOT / "derived" / "bnb_four_channel_total_covariance.csv"
CHECK_RESPONSE_DIRECTORY = CHECK_BNB_DATA_ROOT / "derived" / "archival_2022_reco_bnb26_given_true"


def check_reference_setup() -> tuple[ThreePlusOneParameters, float]:
    document = yaml.safe_load(CHECK_REFERENCE_CONFIG.read_text(encoding="utf-8"))
    values = document["reference_parameters"]
    return (
        ThreePlusOneParameters(**{name: float(value) for name, value in values.items()}),
        float(document["baseline_km"]),
    )


def check_bnb() -> None:
    parameters, baseline_km = check_reference_setup()
    inputs = load_bnb_four_channel_inputs()
    workflow = build_strict_bnb_workflow(CHECK_KERNEL_DIRECTORY, CHECK_COVARIANCE_PATH, parameters, baseline_km)
    reference_prediction = workflow.predictor.predict_total_counts(parameters)
    if not np.allclose(reference_prediction, inputs.published_total_prediction_counts, rtol=1e-12, atol=1e-10):
        raise AssertionError("reference kernel does not reproduce published Signal + Background")
    # This call also proves that the covariance is finite, positive definite,
    # dimensionally compatible and usable by the Cholesky likelihood.
    reference_chi2 = workflow.likelihood.chi2(reference_prediction)
    if not np.isfinite(reference_chi2):
        raise AssertionError("reference chi2 is not finite")
    stored_reference_covariance = load_declared_total_covariance(CHECK_COVARIANCE_PATH).covariance
    active_reference_covariance = workflow.likelihood.covariance_for_prediction(reference_prediction)
    if not np.allclose(active_reference_covariance, stored_reference_covariance, rtol=1e-12, atol=1e-10):
        raise AssertionError("active prediction-scaled covariance does not reproduce the declared reference matrix")
    audit_prediction = workflow.predictor.predict_total_counts(ThreePlusOneParameters(1.2, 0.1, 0.1))
    audit_covariance = workflow.likelihood.covariance_for_prediction(audit_prediction)
    if np.allclose(audit_covariance, active_reference_covariance, rtol=1e-10, atol=1e-8):
        raise AssertionError("active covariance remained fixed when the oscillated prediction changed")
    kernel_metadata = json.loads((CHECK_KERNEL_DIRECTORY / "metadata.json").read_text(encoding="utf-8"))
    if kernel_metadata.get("baseline_km") != baseline_km:
        raise AssertionError("kernel metadata baseline does not match the active config")
    if kernel_metadata.get("reference_parameters", {}) != {
        "delta_m2_41_eV2": parameters.delta_m2_41_eV2,
        "sin2_theta14": parameters.sin2_theta14,
        "sin2_theta24": parameters.sin2_theta24,
        "sin2_2theta_mue_exact": parameters.sin2_2theta_mue_exact,
    }:
        raise AssertionError("kernel metadata reference parameters do not match the active config")
    for channel in ("nue_cc_fc", "nue_cc_pc", "numu_cc_fc", "numu_cc_pc"):
        table = pd.read_csv(CHECK_RESPONSE_DIRECTORY / f"{channel}_reco_given_true.csv")
        response = table.iloc[:, 1:].to_numpy(dtype=float)
        if response.shape != (26, 60) or not np.all(np.isfinite(response)) or np.any(response < 0.0):
            raise AssertionError(f"{channel} response is not a finite non-negative 26x60 matrix")
        sums = response.sum(axis=0)
        if not np.all(np.isclose(sums, 0.0, atol=1e-12) | np.isclose(sums, 1.0, atol=1e-12)):
            raise AssertionError(f"{channel} response columns are not zero-or-one normalized")
    model = ThreePlusOneVacuumModel(parameters)
    energy = np.array([0.2, 0.7, 1.4], dtype=float)
    for initial in range(4):
        total = sum(model.probability(initial, final, energy, baseline_km=baseline_km) for final in range(4))
        if not np.allclose(total, 1.0, atol=1e-12):
            raise AssertionError(f"3+1 probability is not unitary for initial flavour {initial}")
    print("PASS visible public spectrum: 104 bins")
    print("PASS visible covariance: 104 x 104 CSV with JSON metadata")
    print("PASS visible Reco matrices: 4 x (26 x 60), every true column sums to zero or one")
    print("PASS visible reference kernel: published Signal + Background closes bin by bin")
    print("PASS bookkeeping: HEPData Background is added exactly once (its frozen treatment remains a declared approximation)")
    print("PASS scan covariance: reference matrix closes and changes with the current prediction")
    print("PASS 3+1 probability conservation")
    print(f"PASS declared BNB baseline: {baseline_km:.4f} km")
    print(f"PASS reference chi2 is finite: {reference_chi2:.8g}")


# Build a visible, exactly closing BNB reference-reweight kernel.


ANCHOR_PROCESS_FIELDS = {
    ("nue", "nue"): "beam_nue_to_nue_cc_response_counts",
    ("numu", "nue"): "beam_numu_to_nue_cc_response_counts",
    ("nue", "numu"): "beam_nue_to_numu_cc_response_counts",
    ("numu", "numu"): "beam_numu_to_numu_cc_response_counts",
    ("nuebar", "nue"): "beam_nuebar_to_nuebar_cc_response_counts",
    ("numubar", "nue"): "beam_numubar_to_nuebar_cc_response_counts",
    ("nuebar", "numu"): "beam_nuebar_to_numubar_cc_response_counts",
    ("numubar", "numu"): "beam_numubar_to_numubar_cc_response_counts",
}


def anchor_load_reference_setup(path: Path) -> tuple[ThreePlusOneParameters, float]:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    values = document["reference_parameters"]
    parameters = ThreePlusOneParameters(
        delta_m2_41_eV2=float(values["delta_m2_41_eV2"]),
        sin2_theta14=float(values["sin2_theta14"]),
        sin2_theta24=float(values["sin2_theta24"]),
    )
    baseline_km = float(document["baseline_km"])
    if not np.isfinite(baseline_km) or baseline_km <= 0.0:
        raise ValueError("baseline_km must be finite and positive")
    return parameters, baseline_km


def anchor_load_visible_response(directory: Path, channel: str) -> np.ndarray:
    path = directory / f"{channel}_reco_given_true.csv"
    table = pd.read_csv(path)
    expected = ["reco_bin", *[f"true_bin_{index:03d}" for index in range(60)]]
    if list(table.columns) != expected or table.shape != (26, 61):
        raise ValueError(f"unexpected visible 26x60 response table: {path}")
    if not np.array_equal(table["reco_bin"].to_numpy(dtype=int), np.arange(26)):
        raise ValueError(f"reco_bin must be 0..25 in {path}")
    return table.iloc[:, 1:].to_numpy(dtype=float)


def prepare_bnb_kernel() -> None:
    parser = argparse.ArgumentParser(
        description="Build K(reco,true,process) from the public reference prediction using visible inputs."
    )
    parser.add_argument("--reference-config", type=Path, default=Path("configs/experiments/microboone/bnb/analysis.yaml"))
    parser.add_argument("--flux", type=Path, default=Path("data/experiments/microboone/bnb/inputs/bnb_flux.csv"))
    parser.add_argument(
        "--response-directory",
        type=Path,
        default=Path("data/experiments/microboone/bnb/derived/archival_2022_reco_bnb26_given_true"),
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("data/experiments/microboone/bnb/reweighting"),
    )
    arguments = parser.parse_args()

    parameters, baseline_km = anchor_load_reference_setup(arguments.reference_config)
    model = ThreePlusOneVacuumModel(parameters)
    inputs = load_bnb_four_channel_inputs()
    flux_table = pd.read_csv(arguments.flux)
    expected_flux_columns = [
        "true_bin",
        "true_energy_GeV",
        "numu_flux",
        "numubar_flux",
        "nue_flux",
        "nuebar_flux",
    ]
    if list(flux_table.columns) != expected_flux_columns or flux_table.shape != (60, 6):
        raise ValueError("bnb_flux.csv has an unexpected visible schema")
    energy = flux_table["true_energy_GeV"].to_numpy(dtype=float)
    flux = {
        "nue": flux_table["nue_flux"].to_numpy(dtype=float),
        "numu": flux_table["numu_flux"].to_numpy(dtype=float),
        "nuebar": flux_table["nuebar_flux"].to_numpy(dtype=float),
        "numubar": flux_table["numubar_flux"].to_numpy(dtype=float),
    }
    process_arrays = {name: np.zeros((104, 60), dtype=float) for name in ANCHOR_PROCESS_FIELDS.values()}
    closure_rows: list[dict[str, object]] = []

    for channel in BNB_FOUR_CHANNELS:
        final_name = "nue" if channel.identifier.startswith("nue_") else "numu"
        final_index = 0 if final_name == "nue" else 1
        response = anchor_load_visible_response(arguments.response_directory, channel.identifier)
        source_probability = {
            "nue": model.probability(0, final_index, energy, baseline_km=baseline_km),
            "numu": model.probability(1, final_index, energy, baseline_km=baseline_km),
            "nuebar": model.probability(0, final_index, energy, baseline_km=baseline_km, antineutrino=True),
            "numubar": model.probability(1, final_index, energy, baseline_km=baseline_km, antineutrino=True),
        }
        reference_true_weight = sum(flux[name] * source_probability[name] for name in flux)
        denominator = response @ reference_true_weight
        start, stop = channel.first_global_bin, channel.stop_global_bin
        published_signal = inputs.published_signal_counts[start:stop]
        if np.any((denominator <= 0.0) & (published_signal > 0.0)):
            raise ValueError(f"reference response has zero support for a nonzero {channel.identifier} bin")
        scale = np.divide(
            published_signal,
            denominator,
            out=np.zeros_like(published_signal),
            where=denominator > 0.0,
        )
        for source_name in flux:
            field = ANCHOR_PROCESS_FIELDS[(source_name, final_name)]
            process_arrays[field][start:stop, :] = scale[:, None] * response * flux[source_name][None, :]

    templates = BnbFourChannelOscillationTemplates(
        true_energy_GeV=energy,
        fixed_published_background_counts=inputs.published_background_counts,
        **process_arrays,
    )
    reconstructed = templates.predict_total_counts(model, baseline_km=baseline_km)
    residual = reconstructed - inputs.published_total_prediction_counts
    if not np.allclose(reconstructed, inputs.published_total_prediction_counts, rtol=1e-12, atol=1e-10):
        largest = int(np.argmax(np.abs(residual)))
        raise RuntimeError(f"reference kernel failed exact closure at global reco bin {largest}")

    templates.to_directory(
        arguments.output_directory,
        metadata={
            "construction": "per-reconstructed-bin reference-ratio kernel",
            "baseline_km": baseline_km,
            "baseline_interpretation": "BNB target-to-detector distance; one-baseline approximation",
            "formula": "K_r,t,source = S_reference_r * R_r,t * flux_source_t / sum_t,source(R_r,t * flux_source_t * P_source_to_final(reference))",
            "reference_parameters": {
                "delta_m2_41_eV2": parameters.delta_m2_41_eV2,
                "sin2_theta14": parameters.sin2_theta14,
                "sin2_theta24": parameters.sin2_theta24,
                "sin2_2theta_mue_exact": parameters.sin2_2theta_mue_exact,
            },
            "fixed_background_definition": "HEPData published Background block, frozen during reweighting",
            "reference_total_definition": "HEPData published Signal + Background block",
            "assumptions": [
                "unknown cross section and efficiency are not separately identified; their reference-weighted effect is assumed to be absorbed by the chosen response-times-flux true-energy prior and one scale factor per reconstructed bin",
                "neutrino and antineutrino contributions to the same selected final-state channel share that effective ratio",
                "the 2022 normalized migration matrix is used as the declared true-energy prior",
                "the HEPData unconstrained total is imposed as an empirical zero-mixing anchor; the release does not identify it as the paper 3nu/null prediction",
                "this empirical anchor is invalid for claiming a strict reproduction of the paper null prediction",
                "the aggregate HEPData Background block is frozen because public component-level oscillatable templates are unavailable",
            ],
        },
    )
    for channel in BNB_FOUR_CHANNELS:
        for local_bin, global_bin in enumerate(range(channel.first_global_bin, channel.stop_global_bin)):
            closure_rows.append({
                "channel": channel.identifier,
                "channel_reco_bin": local_bin,
                "global_reco_bin": global_bin,
                "published_data_counts": inputs.observed_counts[global_bin],
                "published_background_counts": inputs.published_background_counts[global_bin],
                "published_signal_counts": inputs.published_signal_counts[global_bin],
                "published_total_prediction_counts": inputs.published_total_prediction_counts[global_bin],
                "reconstructed_total_prediction_counts": reconstructed[global_bin],
                "closure_residual_counts": residual[global_bin],
            })
    pd.DataFrame(closure_rows).to_csv(
        arguments.output_directory / "reference_closure.csv",
        index=False,
        float_format="%.17g",
    )
    print(arguments.output_directory)


# Build a fixed-reference BNB total covariance from the released inputs.


def prepare_bnb_covariance() -> None:
    parser = argparse.ArgumentParser(
        description="Add declared data statistics to the released BNB four-channel systematic covariance."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/experiments/microboone/bnb/derived/bnb_four_channel_total_covariance.csv"),
    )
    parser.add_argument(
        "--statistical-method",
        choices=("pearson", "cnp"),
        default="pearson",
        help="Use pearson for the 2025 paper; cnp is retained only for the conflicting HEPData table-header convention.",
    )
    arguments = parser.parse_args()
    inputs = load_bnb_four_channel_inputs()
    provenance = (
        "HEPData 10.17182/hepdata.166435.v1 14-channel covariance, first four BNB channels; "
        "data statistics constructed at the released Signal + Background reference"
    )
    if arguments.statistical_method == "pearson":
        total = pearson_total_covariance_at_reference(
            inputs.systematic_covariance,
            inputs.published_total_prediction_counts,
            provenance=provenance + "; Pearson selected to match the 2025 paper Methods",
        )
    else:
        total = cnp_total_covariance_at_reference(
            inputs.systematic_covariance,
            inputs.observed_counts,
            inputs.published_total_prediction_counts,
            provenance=provenance + "; CNP selected explicitly from the HEPData table header, not as paper default",
        )
    if arguments.output.suffix.lower() != ".csv":
        raise ValueError("--output must be a visible .csv file")
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(arguments.output, total.covariance, delimiter=",", fmt="%.17g")
    metadata = {
        "shape": [104, 104],
        "row_order": "nue_cc_fc,nue_cc_pc,numu_cc_fc,numu_cc_pc; 26 reconstructed bins each",
        "column_order": "same as row_order",
        "statistical_treatment": total.statistical_treatment,
        "parameter_dependence": total.parameter_dependence,
        "reference_prediction_sha256": total.reference_prediction_sha256,
        "provenance": total.provenance,
    }
    arguments.output.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(arguments.output)





# Spectrum payloads belong to this experiment; rendering/writing is shared.
import argparse
import json
import pandas as pd
import yaml
from sterile_fit.output import SpectrumCurve, SpectrumPanel, render_microboone_spectrum_panels, write_csv, write_json
from hashlib import sha256
from sterile_fit.experiments.microboone.public_data import BNB_FOUR_CHANNELS, DEFAULT_SPECTRUM_PATH

# Reproduce the public BNB four-channel data/prediction panels from HEPData.


from sterile_fit.paths import REPOSITORY_ROOT as BNB_PLOT_ROOT
BNB_PLOT_PAPER_FIGURE1_DELTA_M2_41_EV2 = 1.2
BNB_PLOT_PAPER_FIGURE1_SIN2_2THETA_MUE = 0.003
BNB_PLOT_PAPER_FIGURE1_SIN2_THETA24_VALUES = (0.018, 0.0045)


def bnb_plot_parameters_from_appearance_and_sin2_theta24(
    delta_m2_41_eV2: float,
    sin2_2theta_mue: float,
    sin2_theta24: float,
) -> ThreePlusOneParameters:
    """Use the conventional small-theta14 branch of the paper parameterization."""
    sin2_2theta14 = sin2_2theta_mue / sin2_theta24
    if not 0.0 <= sin2_2theta14 <= 1.0:
        raise ValueError("sin2(2theta_mue)/sin2(theta24) must lie in [0, 1]")
    sin2_theta14 = (1.0 - np.sqrt(1.0 - sin2_2theta14)) / 2.0
    parameters = ThreePlusOneParameters(
        delta_m2_41_eV2=delta_m2_41_eV2,
        sin2_theta14=float(sin2_theta14),
        sin2_theta24=sin2_theta24,
    )
    if not np.isclose(parameters.sin2_2theta_mue_exact, sin2_2theta_mue, rtol=1e-12, atol=1e-15):
        raise RuntimeError("paper Figure 1 parameter conversion failed its exact-amplitude check")
    return parameters


def bnb_plot_reference_setup(path: Path) -> tuple[ThreePlusOneParameters, float]:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    parameters = ThreePlusOneParameters(
        **{name: float(value) for name, value in document["reference_parameters"].items()}
    )
    return parameters, float(document["baseline_km"])


def plot_bnb_spectrum() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=result_directory("microboone_bnb", "three_plus_one", "spectra_bnb") / "published_four_channels.png",
        help="PNG destination under the unified outputs directory",
    )
    parser.add_argument(
        "--compare-paper-figure1-points",
        action="store_true",
        help="overlay the two 3+1 parameter points shown in paper Figure 1",
    )
    parser.add_argument(
        "--reference-config",
        type=Path,
        default=BNB_PLOT_ROOT / "configs" / "experiments" / "microboone" / "bnb" / "analysis.yaml",
    )
    parser.add_argument(
        "--kernel",
        type=Path,
        default=BNB_PLOT_ROOT / "data" / "experiments" / "microboone" / "bnb" / "reweighting",
    )
    parser.add_argument(
        "--covariance",
        type=Path,
        default=BNB_PLOT_ROOT / "data" / "experiments" / "microboone" / "bnb" / "derived" / "bnb_four_channel_total_covariance.csv",
    )
    arguments = parser.parse_args()
    inputs = load_bnb_four_channel_inputs()
    comparison_predictions: dict[str, np.ndarray] = {}
    comparison_parameters: dict[str, ThreePlusOneParameters] = {}
    comparison_chi2: dict[str, float] = {}
    workflow = None
    if arguments.compare_paper_figure1_points:
        reference, baseline_km = bnb_plot_reference_setup(arguments.reference_config)
        workflow = build_strict_bnb_workflow(
            arguments.kernel,
            arguments.covariance,
            reference,
            baseline_km,
        )
    if arguments.compare_paper_figure1_points:
        assert workflow is not None
        for sin2_theta24 in BNB_PLOT_PAPER_FIGURE1_SIN2_THETA24_VALUES:
            key = f"paper_figure1_sin2_theta24_{sin2_theta24:g}"
            parameters = bnb_plot_parameters_from_appearance_and_sin2_theta24(
                BNB_PLOT_PAPER_FIGURE1_DELTA_M2_41_EV2,
                BNB_PLOT_PAPER_FIGURE1_SIN2_2THETA_MUE,
                sin2_theta24,
            )
            comparison_parameters[key] = parameters
            comparison_predictions[key] = workflow.predictor.predict_total_counts(parameters)
    # The released table has 25 bins from 0 to 2.5 GeV and one overflow bin.
    edges_GeV = np.append(np.arange(0.0, 2.6, 0.1), 3.0)
    panels: list[SpectrumPanel] = []
    audit_rows: list[dict[str, object]] = []
    for channel in BNB_FOUR_CHANNELS:
        start, stop = channel.first_global_bin, channel.stop_global_bin
        observed = inputs.observed_counts[start:stop]
        total = inputs.published_total_prediction_counts[start:stop]
        background = inputs.published_background_counts[start:stop]
        error_up = inputs.observed_statistical_error_up[start:stop]
        error_down = inputs.observed_statistical_error_down[start:stop]
        prediction_systematic_sigma = np.sqrt(np.diag(inputs.systematic_covariance)[start:stop])
        prediction_lower = np.maximum(total - prediction_systematic_sigma, 0.0)
        prediction_upper = total + prediction_systematic_sigma
        curves: list[SpectrumCurve] = []
        if arguments.compare_paper_figure1_points:
            curves.append(SpectrumCurve(
                r"Paper Fig. 1 point: $\sin^2\theta_{24}=0.018$",
                comparison_predictions["paper_figure1_sin2_theta24_0.018"][start:stop],
                "#00A6C8", "-", 2.0,
            ))
            curves.append(SpectrumCurve(
                r"Paper Fig. 1 point: $\sin^2\theta_{24}=0.0045$",
                comparison_predictions["paper_figure1_sin2_theta24_0.0045"][start:stop],
                "#C24E00", "-", 2.0,
            ))
        panels.append(SpectrumPanel(
            title=channel.identifier,
            energy_edges_GeV=edges_GeV,
            observed_counts=observed,
            observed_error_down=error_down,
            observed_error_up=error_up,
            background_counts=background,
            signal_plus_background_counts=total,
            comparison_curves=tuple(curves),
            prediction_systematic_sigma=prediction_systematic_sigma,
        ))
        for local_bin in range(26):
            row: dict[str, object] = {
                "channel": channel.identifier,
                "channel_reco_bin": local_bin,
                "reco_energy_low_GeV": edges_GeV[local_bin],
                "reco_energy_high_GeV": edges_GeV[local_bin + 1],
                "is_overflow_bin": local_bin == 25,
                "plotted_data_counts": observed[local_bin],
                "plotted_background_counts": background[local_bin],
                "plotted_signal_plus_background_counts": total[local_bin],
                "prediction_systematic_sigma_counts": prediction_systematic_sigma[local_bin],
                "prediction_systematic_lower_counts": prediction_lower[local_bin],
                "prediction_systematic_upper_counts": prediction_upper[local_bin],
                "derived_signal_counts": total[local_bin] - background[local_bin],
            }
            if arguments.compare_paper_figure1_points:
                global_bin = start + local_bin
                row.update({
                    "paper_figure1_sin2_theta24_0p018_counts": comparison_predictions[
                        "paper_figure1_sin2_theta24_0.018"
                    ][global_bin],
                    "paper_figure1_sin2_theta24_0p0045_counts": comparison_predictions[
                        "paper_figure1_sin2_theta24_0.0045"
                    ][global_bin],
                })
            audit_rows.append(row)
    render_microboone_spectrum_panels(
        panels,
        arguments.output,
        title="MicroBooNE BNB: public four-channel input reproduction\n"
        "(last bin is the released overflow bin)",
    )
    audit_path = arguments.output.with_suffix(".csv")
    write_csv(pd.DataFrame(audit_rows), 
        audit_path,
        index=False,
        float_format="%.17g",
    )
    spectrum_source = DEFAULT_SPECTRUM_PATH
    metadata = {
        "plot_semantics": {
            "data": "HEPData Data",
            "background": "HEPData Background, plotted separately",
            "signal_plus_background": "HEPData Signal + Background, used directly; Background is not added again",
            "prediction_uncertainty_band": "plus/minus sqrt of the released systematic covariance diagonal; correlations are used by chi2 but cannot be represented by independent per-bin bars",
            "data_error_bars": "released asymmetric data statistical errors only",
            "derived_signal": "Signal + Background minus Background; audit table only",
        },
        "binning": "25 bins of width 0.1 GeV from 0 to 2.5 GeV, followed by the released overflow bin displayed to 3.0 GeV",
        "spectrum_source": str(spectrum_source),
        "spectrum_source_sha256": sha256(spectrum_source.read_bytes()).hexdigest(),
        "audit_table": str(audit_path),
        "audit_table_sha256": sha256(audit_path.read_bytes()).hexdigest(),
        "fit_point_comparison": None,
        "paper_figure1_comparison": None,
    }
    if arguments.compare_paper_figure1_points:
        metadata["paper_figure1_comparison"] = {
            "source_coordinates": {
                "delta_m2_41_eV2": BNB_PLOT_PAPER_FIGURE1_DELTA_M2_41_EV2,
                "sin2_2theta_mue": BNB_PLOT_PAPER_FIGURE1_SIN2_2THETA_MUE,
                "sin2_theta24_values": list(BNB_PLOT_PAPER_FIGURE1_SIN2_THETA24_VALUES),
            },
            "converted_parameters": {
                name: {
                    "delta_m2_41_eV2": parameters.delta_m2_41_eV2,
                    "sin2_theta14": parameters.sin2_theta14,
                    "sin2_theta24": parameters.sin2_theta24,
                    "sin2_2theta_mue_exact": parameters.sin2_2theta_mue_exact,
                }
                for name, parameters in comparison_parameters.items()
                if name.startswith("paper_figure1_")
            },
            "comparison_scope": "Only nue_cc_fc is a direct comparison with the paper's BNB panel in Figure 1; the other BNB channels are local-model diagnostics.",
            "prediction_provenance": "The coordinates are published; the curves are produced by this repository's empirical BNB kernel and are not digitized collaboration curves.",
        }
    write_json(arguments.output.with_suffix(".metadata.json"), metadata)


