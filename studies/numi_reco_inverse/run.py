"""Infer a constrained *equivalent* NuMI response without changing active code.

The inverse problem is non-identifiable.  This file therefore constructs a
small smooth perturbation family with exact linear invariants.  The expensive
official-surface fit is kept behind ``--mode fit``; ``--mode check`` verifies
the factorisation and every hard constraint without changing scientific data.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from sterile_fit.experiments.microboone.numi import (  # noqa: E402
    NumiFourChannelEmpiricalKernel,
    PROCESS_FIELDS,
)
from sterile_fit.experiments.microboone.public_data import (  # noqa: E402
    NUMI_FOUR_CHANNELS,
    load_numi_four_channel_inputs,
)
from sterile_fit.output import result_directory  # noqa: E402
from sterile_fit.core.profile_three_plus_one import (  # noqa: E402
    profile_s14_s24_at_fixed_sin2_2theta_ee,
    profile_s14_s24_at_fixed_sin2_2theta_mue,
)
from sterile_fit.core.three_plus_one import ThreePlusOneParameters, ThreePlusOneVacuumModel  # noqa: E402


RESPONSE_DIRECTORY = ROOT / "data/experiments/microboone/bnb/derived/archival_2022_reco_bnb26_given_true"
KERNEL_DIRECTORY = ROOT / "data/experiments/microboone/numi/reweighting"
OFFICIAL_DIRECTORY = ROOT / "outputs/studies/official_grid_wilks/legacy/results"


@dataclass(frozen=True)
class ChannelPerturbation:
    name: str
    original: np.ndarray
    basis: np.ndarray  # (coefficient, reco, true)
    coefficient_bound: float
    zero_signal: np.ndarray
    maximum_fractional_change: float

    def response(self, coefficients: np.ndarray) -> np.ndarray:
        direction = np.tensordot(coefficients, self.basis, axes=(0, 0))
        direction_norm = float(np.linalg.norm(direction))
        coordinate_radius = min(
            float(np.linalg.norm(coefficients)) / np.sqrt(coefficients.size), 1.0
        )
        if direction_norm == 0.0 or coordinate_radius == 0.0:
            return self.original.copy()
        target_scale = (
            self.maximum_fractional_change
            * coordinate_radius
            * float(np.linalg.norm(self.original))
            / direction_norm
        )
        decreasing = direction < 0.0
        feasible_scale = np.inf
        if np.any(decreasing):
            feasible_scale = float(
                np.min(self.original[decreasing] / -direction[decreasing])
            )
        scale = min(target_scale, 0.999 * feasible_scale)
        result = self.original + scale * direction
        # Only roundoff-sized negatives are tolerated; clipping is not used to
        # enforce physics because it would break the two exact equalities.
        if np.min(result) < -2e-12:
            raise ValueError(f"{self.name}: coefficient vector violates non-negativity")
        result[np.abs(result) < 2e-15] = 0.0
        return result


def _read_response(channel: str) -> np.ndarray:
    table = pd.read_csv(RESPONSE_DIRECTORY / f"{channel}_reco_given_true.csv")
    response = table.iloc[:, 1:].to_numpy(float)
    if response.shape != (26, 60):
        raise ValueError(f"unexpected response shape for {channel}")
    return response


def _channel_kernel_arrays(kernel: NumiFourChannelEmpiricalKernel, channel_index: int):
    block = slice(26 * channel_index, 26 * (channel_index + 1))
    return [np.asarray(getattr(kernel, field), float)[block] for field in PROCESS_FIELDS]


def _zero_oscillation_kernel(kernel: NumiFourChannelEmpiricalKernel, channel_index: int, channel_name: str):
    """Return only source=final survival templates, since appearance is zero."""
    block = slice(26 * channel_index, 26 * (channel_index + 1))
    if channel_name.startswith("nue_"):
        fields = ("beam_nue_to_nue_cc_response_counts", "beam_nuebar_to_nuebar_cc_response_counts")
    else:
        fields = ("beam_numu_to_numu_cc_response_counts", "beam_numubar_to_numubar_cc_response_counts")
    return sum(np.asarray(getattr(kernel, field), float)[block] for field in fields)


def _make_channel_perturbation(
    channel_index: int,
    channel_name: str,
    kernel: NumiFourChannelEmpiricalKernel,
    *,
    modes_reco: int,
    modes_true: int,
    maximum_fractional_change: float,
) -> ChannelPerturbation:
    r0 = _read_response(channel_name)
    zero_kernel = _zero_oscillation_kernel(kernel, channel_index, channel_name)
    positive = r0 > 0.0
    column_peak = np.max(r0, axis=0, keepdims=True)
    active = positive & (r0 >= 1e-3 * column_peak)
    flat_positive = np.flatnonzero(active.ravel())
    n = flat_positive.size
    if n == 0:
        raise ValueError(f"{channel_name}: empty response")

    # A delta = 0 encodes all 60 column sums and all 26 zero-oscillation
    # reconstructed counts.  The latter use K0/R0 while structural zeros stay
    # absent from the variable vector.
    constraints = np.zeros((86, n), dtype=float)
    rr, tt = np.unravel_index(flat_positive, r0.shape)
    constraints[tt, np.arange(n)] = 1.0
    remaining_weight = np.zeros_like(r0)
    remaining_weight[positive] = zero_kernel[positive] / r0[positive]
    constraints[60 + rr, np.arange(n)] = remaining_weight[rr, tt]
    projector = np.eye(n) - constraints.T @ np.linalg.pinv(constraints @ constraints.T) @ constraints

    reco_coordinate = (np.arange(26) + 0.5) / 26.0
    true_coordinate = (np.arange(60) + 0.5) / 60.0
    candidates = []
    for kr in range(1, modes_reco + 1):
        for kt in range(1, modes_true + 1):
            smooth = np.cos(np.pi * kr * reco_coordinate)[:, None] * np.cos(
                np.pi * kt * true_coordinate
            )[None, :]
            vector = (r0 * smooth).ravel()[flat_positive]
            vector = projector @ vector
            norm = np.linalg.norm(vector)
            if norm > 1e-12:
                candidates.append(vector / norm)
    if not candidates:
        raise ValueError(f"{channel_name}: no nontrivial constrained smooth modes")
    raw = np.stack(candidates, axis=1)
    q, _ = np.linalg.qr(raw)
    basis = np.zeros((q.shape[1], *r0.shape), dtype=float)
    for index in range(q.shape[1]):
        basis[index].ravel()[flat_positive] = q[:, index]

    # The coefficient coordinates only choose a direction and radius.  The
    # response() method computes the exact non-negative step along that
    # direction, so negligible frozen tails cannot globally shrink the basis.
    bound = 1.0
    zero_signal = zero_kernel.sum(axis=1)
    return ChannelPerturbation(
        channel_name, r0, basis, bound, zero_signal, maximum_fractional_change
    )


def _validate(item: ChannelPerturbation, coefficients: np.ndarray) -> dict[str, float]:
    response = item.response(coefficients)
    original_sums = item.original.sum(axis=0)
    new_sums = response.sum(axis=0)
    # Recover the fixed remaining weight from the stored kernel once more.
    kernel = NumiFourChannelEmpiricalKernel.from_directory(KERNEL_DIRECTORY)
    channel_index = [c.identifier for c in NUMI_FOUR_CHANNELS].index(item.name)
    zero_kernel = _zero_oscillation_kernel(kernel, channel_index, item.name)
    weight = np.divide(zero_kernel, item.original, out=np.zeros_like(zero_kernel), where=item.original > 0)
    new_signal = np.sum(weight * response, axis=1)
    return {
        "minimum_element": float(response.min()),
        "maximum_column_sum_residual": float(np.max(np.abs(new_sums - original_sums))),
        "maximum_zero_oscillation_bin_residual": float(np.max(np.abs(new_signal - item.zero_signal))),
        "relative_frobenius_change": float(np.linalg.norm(response - item.original) / np.linalg.norm(item.original)),
    }


def _write_response(path: Path, response: np.ndarray) -> None:
    table = pd.DataFrame(response, columns=[f"true_bin_{i:03d}" for i in range(60)])
    table.insert(0, "reco_bin", np.arange(26))
    table.to_csv(path, index=False, float_format="%.17g")


def _surface(path: Path, x_name: str, value_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    table = pd.read_csv(path)
    pivot = table.pivot(index="delta_m2_41_eV2", columns=x_name, values=value_name)
    return pivot.columns.to_numpy(float), pivot.index.to_numpy(float), pivot.to_numpy(float)


def _candidate_process_arrays(
    kernel: NumiFourChannelEmpiricalKernel,
    items: list[ChannelPerturbation],
    normalized_coefficients: np.ndarray,
) -> dict[str, np.ndarray]:
    result = {field: np.asarray(getattr(kernel, field), float).copy() for field in PROCESS_FIELDS}
    cursor = 0
    for channel_index, item in enumerate(items):
        count = item.basis.shape[0]
        coefficients = normalized_coefficients[cursor:cursor + count] * item.coefficient_bound
        cursor += count
        response = item.response(coefficients)
        ratio = np.divide(response, item.original, out=np.zeros_like(response), where=item.original > 0)
        block = slice(26 * channel_index, 26 * (channel_index + 1))
        for field in PROCESS_FIELDS:
            result[field][block] *= ratio
    return result


def _prediction_function(kernel, items, normalized_coefficients, baseline_km: float):
    arrays = _candidate_process_arrays(kernel, items, normalized_coefficients)

    def predict(parameters: ThreePlusOneParameters) -> np.ndarray:
        model = ThreePlusOneVacuumModel(parameters)
        e = kernel.true_energy_GeV
        p = model.probability
        prediction = np.asarray(kernel.fixed_published_background_counts, float).copy()
        specifications = (
            ("beam_nue_to_nue_cc_response_counts", 0, 0, False),
            ("beam_numu_to_nue_cc_response_counts", 1, 0, False),
            ("beam_nue_to_numu_cc_response_counts", 0, 1, False),
            ("beam_numu_to_numu_cc_response_counts", 1, 1, False),
            ("beam_nuebar_to_nuebar_cc_response_counts", 0, 0, True),
            ("beam_numubar_to_nuebar_cc_response_counts", 1, 0, True),
            ("beam_nuebar_to_numubar_cc_response_counts", 0, 1, True),
            ("beam_numubar_to_numubar_cc_response_counts", 1, 1, True),
        )
        for field, initial, final, anti in specifications:
            prediction += arrays[field] @ p(initial, final, e, baseline_km, antineutrino=anti)
        return prediction

    return predict, arrays


def _parallel_map(function, coordinates, workers: int):
    coordinates = list(coordinates)
    if workers == 1:
        return [function(value) for value in coordinates]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(function, coordinates))


def _profile_surfaces(predict, inputs, mass_a, x_a, mass_b, x_b, workers: int):
    from sterile_fit.core.likelihood import PredictionScaledGaussianLikelihood
    likelihood = PredictionScaledGaussianLikelihood(
        inputs.observed_counts,
        inputs.published_total_prediction_counts,
        inputs.systematic_covariance,
    )
    objective = lambda parameters: likelihood.chi2(predict(parameters))
    coordinates_a = [(m, x) for m in mass_a for x in x_a]
    coordinates_b = [(m, x) for m in mass_b for x in x_b]
    results_a = _parallel_map(
        lambda q: profile_s14_s24_at_fixed_sin2_2theta_mue(
            objective, delta_m2_41_eV2=q[0], sin2_2theta_mue=q[1]
        ), coordinates_a, workers,
    )
    results_b = _parallel_map(
        lambda q: profile_s14_s24_at_fixed_sin2_2theta_ee(
            objective, delta_m2_41_eV2=q[0], sin2_2theta_ee=q[1]
        ), coordinates_b, workers,
    )
    za = np.asarray([r.best_fit.chi2 for r in results_a]).reshape(len(mass_a), len(x_a))
    zb = np.asarray([r.best_fit.chi2 for r in results_b]).reshape(len(mass_b), len(x_b))
    minimum = min(float(za.min()), float(zb.min()))
    return za - minimum, zb - minimum


def _fit(items, kernel, inputs, args, output: Path):
    official_xa, official_ma, official_za = _surface(
        OFFICIAL_DIRECTORY / "fig3a_official_profile.csv", "sin2_2theta_mue", "official_profile_delta_chi2"
    )
    official_xb, official_mb, official_zb = _surface(
        OFFICIAL_DIRECTORY / "fig3b_official_profile.csv", "sin2_2theta_ee", "official_profile_delta_chi2"
    )
    # Sparse log-uniform training coordinates.  Final reporting still uses the
    # requested 100 x 100 axes and is not interpolated from this training mesh.
    finite_a_columns = np.all(np.isfinite(official_za), axis=0)
    official_xa = official_xa[finite_a_columns]
    official_za = official_za[:, finite_a_columns]
    # Exact unit-amplitude boundaries can make a published zero-background bin
    # have zero expectation, outside the active likelihood's strict domain.
    # They are plot boundaries, not relevant to the 5.99 contour fit.
    fit_xa = official_xa[official_xa <= 0.5]
    fit_xb = official_xb[official_xb <= 0.8]
    ma = np.geomspace(official_ma[0], official_ma[-1], args.training_points)
    xa = np.geomspace(fit_xa[0], fit_xa[-1], args.training_points)
    official_b_mass = official_mb[(official_mb >= 0.1) & (official_mb <= 14.0)]
    mb = np.geomspace(official_b_mass[0], official_b_mass[-1], args.training_points)
    xb = np.geomspace(fit_xb[0], fit_xb[-1], args.training_points)
    interp_a = RegularGridInterpolator((np.log(official_ma), np.log(official_xa)), official_za)
    interp_b = RegularGridInterpolator((np.log(official_mb), np.log(official_xb)), official_zb)
    target_a = interp_a(np.stack(np.meshgrid(np.log(ma), np.log(xa), indexing="ij"), axis=-1))
    target_b = interp_b(np.stack(np.meshgrid(np.log(mb), np.log(xb), indexing="ij"), axis=-1))
    dimension = sum(item.basis.shape[0] for item in items)
    x = np.zeros(dimension, dtype=float)
    rng = np.random.default_rng(args.seed)
    history = []

    def loss(vector):
        predict, _ = _prediction_function(kernel, items, vector, args.baseline_km)
        za, zb = _profile_surfaces(predict, inputs, ma, xa, mb, xb, args.workers)
        # Dense high-chi2 regions must not swamp the exclusion boundary.  A
        # smooth weight peaks near 5.99 while retaining global shape information.
        wa = 0.2 + np.exp(-0.5 * ((target_a - 5.99) / 4.0) ** 2)
        wb = 0.2 + np.exp(-0.5 * ((target_b - 5.99) / 4.0) ** 2)
        mismatch = (np.sum(wa * (za - target_a) ** 2) + np.sum(wb * (zb - target_b) ** 2)) / (wa.sum() + wb.sum())
        regularization = args.regularization * float(vector @ vector) / max(1, vector.size)
        return float(mismatch + regularization)

    started = time.perf_counter()
    initial_loss = loss(x)
    history.append({"iteration": 0, "loss": initial_loss, "elapsed_seconds": time.perf_counter() - started})
    print(f"fit 0/{args.iterations}: loss={initial_loss:.6g}", flush=True)
    for iteration in range(1, args.iterations + 1):
        ak = args.learning_rate / (iteration + 5.0) ** 0.602
        ck = args.perturbation_scale / iteration ** 0.101
        direction = rng.choice((-1.0, 1.0), size=dimension)
        plus = np.clip(x + ck * direction, -1.0, 1.0)
        minus = np.clip(x - ck * direction, -1.0, 1.0)
        loss_plus = loss(plus)
        loss_minus = loss(minus)
        gradient = (loss_plus - loss_minus) / (2.0 * ck) * direction
        # SPSA supplies a direction but its raw magnitude inherits arbitrary
        # chi-square units.  Infinity-norm scaling makes learning_rate a stable
        # dimensionless step in the bounded coefficient coordinates.
        scaled_gradient = gradient / max(float(np.max(np.abs(gradient))), 1e-12)
        candidate = np.clip(x - ak * scaled_gradient, -1.0, 1.0)
        candidate_loss = loss(candidate)
        choices = (
            (history[-1]["loss"], x),
            (loss_plus, plus),
            (loss_minus, minus),
            (candidate_loss, candidate),
        )
        accepted_loss, accepted_vector = min(choices, key=lambda item: item[0])
        x = np.asarray(accepted_vector, dtype=float).copy()
        elapsed = time.perf_counter() - started
        remaining = elapsed / iteration * (args.iterations - iteration)
        history.append({"iteration": iteration, "loss": accepted_loss, "elapsed_seconds": elapsed, "estimated_remaining_seconds": remaining})
        np.savetxt(output / "coefficients_checkpoint.csv", x[None, :], delimiter=",", fmt="%.17g")
        pd.DataFrame(history).to_csv(output / "fit_history.csv", index=False)
        print(f"fit {iteration}/{args.iterations}: loss={accepted_loss:.6g}, elapsed={elapsed/60:.1f} min, remaining={remaining/60:.1f} min", flush=True)
    return x, history


def _write_final_surfaces(kernel, items, coefficients, inputs, args, output: Path) -> None:
    predict, _ = _prediction_function(kernel, items, coefficients, args.baseline_km)
    mass_a = np.geomspace(1e-2, 1e2, args.grid_points)
    x_a = np.geomspace(1e-4, np.nextafter(1.0, 0.0), args.grid_points)
    mass_b = np.geomspace(1e-1, 14.0, args.grid_points)
    x_b = np.geomspace(1e-2, np.nextafter(1.0, 0.0), args.grid_points)
    started = time.perf_counter()
    za, zb = _profile_surfaces(predict, inputs, mass_a, x_a, mass_b, x_b, args.workers)
    print(f"final 100x100 profiles complete in {(time.perf_counter()-started)/60:.1f} min", flush=True)
    pd.DataFrame({
        "delta_m2_41_eV2": np.repeat(mass_a, x_a.size),
        "sin2_2theta_mue": np.tile(x_a, mass_a.size),
        "candidate_profile_delta_chi2": za.ravel(),
    }).to_csv(output / "fig3a_candidate_100x100.csv", index=False, float_format="%.17g")
    pd.DataFrame({
        "delta_m2_41_eV2": np.repeat(mass_b, x_b.size),
        "sin2_2theta_ee": np.tile(x_b, mass_b.size),
        "candidate_profile_delta_chi2": zb.ravel(),
    }).to_csv(output / "fig3b_candidate_100x100.csv", index=False, float_format="%.17g")
    official_xa, official_ma, official_za = _surface(OFFICIAL_DIRECTORY / "fig3a_official_profile.csv", "sin2_2theta_mue", "official_profile_delta_chi2")
    official_xb, official_mb, official_zb = _surface(OFFICIAL_DIRECTORY / "fig3b_official_profile.csv", "sin2_2theta_ee", "official_profile_delta_chi2")
    figure, axes = plt.subplots(1, 2, figsize=(13.5, 5.4))
    for axis, x, mass, z, ox, om, oz, label in (
        (axes[0], x_a, mass_a, za, official_xa, official_ma, official_za, r"$\sin^2(2\theta_{\mu e})$"),
        (axes[1], x_b, mass_b, zb, official_xb, official_mb, official_zb, r"$\sin^2(2\theta_{ee})$"),
    ):
        colour = axis.pcolormesh(x, mass, z, shading="auto", cmap="viridis", vmin=0, vmax=25)
        axis.contour(x, mass, z, levels=[5.99], colors="tab:blue", linewidths=2)
        axis.contour(ox, om, oz, levels=[5.99], colors="tab:orange", linestyles="--", linewidths=2)
        axis.set_xscale("log"); axis.set_yscale("log")
        axis.set_xlabel(label); axis.set_ylabel(r"$\Delta m^2_{41}\,[\mathrm{eV}^2]$")
    axes[0].set_xlim(1e-4, 1); axes[0].set_ylim(1e-2, 1e2)
    axes[1].set_xlim(1e-2, 1); axes[1].set_ylim(1e-1, 14)
    figure.colorbar(colour, ax=axes, label=r"candidate profiled $\Delta\chi^2$")
    figure.suptitle("NuMI-only constrained equivalent-Reco study")
    figure.subplots_adjust(left=.08, right=.93, bottom=.12, top=.88, wspace=.26)
    figure.savefig(output / "candidate_vs_official_100x100.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("check", "fit"), default="check")
    parser.add_argument("--grid-points", type=int, default=100)
    parser.add_argument("--modes-reco", type=int, default=4)
    parser.add_argument("--modes-true", type=int, default=4)
    parser.add_argument("--maximum-fractional-change", type=float, default=0.20)
    parser.add_argument("--output-directory", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--training-points", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=12)
    parser.add_argument("--seed", type=int, default=73021)
    parser.add_argument("--learning-rate", type=float, default=0.04)
    parser.add_argument("--perturbation-scale", type=float, default=0.08)
    parser.add_argument("--regularization", type=float, default=0.02)
    parser.add_argument("--baseline-km", type=float, default=0.680)
    args = parser.parse_args()
    if args.grid_points != 100:
        raise ValueError("this reconstruction is fixed to the requested official-style 100 x 100 output grids")
    if not 0.0 < args.maximum_fractional_change < 1.0:
        raise ValueError("maximum fractional change must lie strictly between zero and one")

    kernel = NumiFourChannelEmpiricalKernel.from_directory(KERNEL_DIRECTORY)
    inputs = load_numi_four_channel_inputs()
    expected_signal = inputs.published_total_prediction_counts - inputs.published_background_counts
    items = []
    for index, channel in enumerate(NUMI_FOUR_CHANNELS):
        item = _make_channel_perturbation(
            index,
            channel.identifier,
            kernel,
            modes_reco=args.modes_reco,
            modes_true=args.modes_true,
            maximum_fractional_change=args.maximum_fractional_change,
        )
        block = slice(26 * index, 26 * (index + 1))
        if not np.allclose(item.zero_signal, expected_signal[block], rtol=0.0, atol=2e-10):
            raise RuntimeError(f"{channel.identifier}: stored kernel does not close before inversion")
        items.append(item)

    output = args.output_directory or result_directory("studies", "numi_reco_inverse", args.mode)
    output.mkdir(parents=True, exist_ok=False)
    validation = {}
    for item in items:
        coefficients = np.zeros(item.basis.shape[0])
        validation[item.name] = _validate(item, coefficients)
        _write_response(output / f"{item.name}_original_reco.csv", item.original)
        _write_response(output / f"{item.name}_candidate_reco.csv", item.response(coefficients))
        _write_response(output / f"{item.name}_delta_reco.csv", item.response(coefficients) - item.original)

    if args.mode == "fit":
        coefficients, history = _fit(items, kernel, inputs, args, output)
        _, corrected_arrays = _prediction_function(kernel, items, coefficients, args.baseline_km)
        cursor = 0
        validation = {}
        for item in items:
            count = item.basis.shape[0]
            physical = coefficients[cursor:cursor + count] * item.coefficient_bound
            cursor += count
            candidate = item.response(physical)
            validation[item.name] = _validate(item, physical)
            _write_response(output / f"{item.name}_candidate_reco.csv", candidate)
            _write_response(output / f"{item.name}_delta_reco.csv", candidate - item.original)
        kernel_output = output / "corrected_event_kernel"
        kernel_output.mkdir()
        for field, values in corrected_arrays.items():
            table = pd.DataFrame(values, columns=[f"true_bin_{i:03d}" for i in range(60)])
            table.insert(0, "local_numi_reco_bin", np.arange(104))
            table.to_csv(kernel_output / f"{field}.csv", index=False, float_format="%.17g")
        _write_final_surfaces(kernel, items, coefficients, inputs, args, output)
        status = "fit_complete_candidate_not_unique"
    else:
        history = []
        status = "constraint_family_check_only"

    metadata = {
        "status": status,
        "scope": "NuMI only; BNB unchanged",
        "requested_output_grid": {
            "points_per_axis": 100,
            "fig3a": {"x": [1e-4, 1.0], "dm2_eV2": [1e-2, 1e2]},
            "fig3b": {"x": [1e-2, 1.0], "dm2_eV2": [1e-1, 14.0]},
        },
        "factorisation": "stored event kernel = fixed remaining weight times variable response on original nonzero support",
        "constraints": {
            "column_normalisation": "exact linear invariant",
            "zero_oscillation_bin_closure": "exact linear invariant",
            "nonnegative": "guaranteed by coefficient bounds and unchanged structural zeros",
            "smooth": f"cosine basis {args.modes_reco} x {args.modes_true} before exact nullspace projection",
            "small_correction": f"maximum pointwise fractional envelope {args.maximum_fractional_change}",
        },
        "validation_at_zero_perturbation": validation,
        "fit": {"iterations": args.iterations if args.mode == "fit" else 0, "training_points_per_axis": args.training_points, "workers": args.workers, "history": history},
        "scientific_warning": "the released scalar delta-chi-square grid cannot uniquely recover the collaboration response",
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
