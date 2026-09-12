"""Parallel constrained Gauss--Newton fit of an equivalent NuMI response."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from studies.numi_reco_inverse import run as inverse  # noqa: E402
from sterile_fit.experiments.microboone.numi import NumiFourChannelEmpiricalKernel  # noqa: E402
from sterile_fit.experiments.microboone.public_data import (  # noqa: E402
    NUMI_FOUR_CHANNELS,
    load_numi_four_channel_inputs,
)
from sterile_fit.output import result_directory  # noqa: E402


_STATE = None


def _build_state(modes: int, maximum_change: float, training_points: int):
    kernel = NumiFourChannelEmpiricalKernel.from_directory(inverse.KERNEL_DIRECTORY)
    inputs = load_numi_four_channel_inputs()
    items = [
        inverse._make_channel_perturbation(
            index,
            channel.identifier,
            kernel,
            modes_reco=modes,
            modes_true=modes,
            maximum_fractional_change=maximum_change,
        )
        for index, channel in enumerate(NUMI_FOUR_CHANNELS)
    ]
    xa0, ma0, za0 = inverse._surface(
        inverse.OFFICIAL_DIRECTORY / "fig3a_official_profile.csv",
        "sin2_2theta_mue",
        "official_profile_delta_chi2",
    )
    xb0, mb0, zb0 = inverse._surface(
        inverse.OFFICIAL_DIRECTORY / "fig3b_official_profile.csv",
        "sin2_2theta_ee",
        "official_profile_delta_chi2",
    )
    finite = np.all(np.isfinite(za0), axis=0)
    xa0, za0 = xa0[finite], za0[:, finite]
    fit_xa = xa0[xa0 <= 0.5]
    fit_xb = xb0[xb0 <= 0.8]
    ma = np.geomspace(ma0[0], ma0[-1], training_points)
    xa = np.geomspace(fit_xa[0], fit_xa[-1], training_points)
    allowed_mb = mb0[(mb0 >= 0.1) & (mb0 <= 14.0)]
    mb = np.geomspace(allowed_mb[0], allowed_mb[-1], training_points)
    xb = np.geomspace(fit_xb[0], fit_xb[-1], training_points)
    ia = RegularGridInterpolator((np.log(ma0), np.log(xa0)), za0)
    ib = RegularGridInterpolator((np.log(mb0), np.log(xb0)), zb0)
    ta = ia(np.stack(np.meshgrid(np.log(ma), np.log(xa), indexing="ij"), axis=-1))
    tb = ib(np.stack(np.meshgrid(np.log(mb), np.log(xb), indexing="ij"), axis=-1))
    wa = 0.2 + np.exp(-0.5 * ((ta - 5.99) / 4.0) ** 2)
    wb = 0.2 + np.exp(-0.5 * ((tb - 5.99) / 4.0) ** 2)
    return kernel, inputs, items, ma, xa, mb, xb, ta, tb, wa, wb


def _init_worker(modes, maximum_change, training_points):
    global _STATE
    _STATE = _build_state(modes, maximum_change, training_points)


def _residual_worker(coefficients):
    kernel, inputs, items, ma, xa, mb, xb, ta, tb, wa, wb = _STATE
    predict, _ = inverse._prediction_function(kernel, items, np.asarray(coefficients), 0.680)
    za, zb = inverse._profile_surfaces(predict, inputs, ma, xa, mb, xb, workers=1)
    return np.concatenate((
        np.sqrt(wa).ravel() * (za - ta).ravel(),
        np.sqrt(wb).ravel() * (zb - tb).ravel(),
    ))


def _loss(residual, coefficients, regularization):
    return float(np.mean(residual * residual) + regularization * np.mean(coefficients * coefficients))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--modes", type=int, default=2)
    parser.add_argument("--training-points", type=int, default=6)
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--difference-step", type=float, default=0.03)
    parser.add_argument("--trust-radius", type=float, default=0.15)
    parser.add_argument("--regularization", type=float, default=0.05)
    parser.add_argument("--maximum-fractional-change", type=float, default=0.20)
    parser.add_argument("--output-directory", type=Path)
    args = parser.parse_args()
    if args.workers < 1 or args.workers > 8:
        raise ValueError("workers must be in 1..8")
    if not 0 < args.maximum_fractional_change < 1:
        raise ValueError("maximum-fractional-change must be in (0,1)")
    output = args.output_directory or result_directory("studies", "numi_reco_inverse", "gauss_newton")
    output.mkdir(parents=True, exist_ok=False)
    state = _build_state(args.modes, args.maximum_fractional_change, args.training_points)
    kernel, inputs, items = state[:3]
    dimension = sum(item.basis.shape[0] for item in items)
    coefficients = np.zeros(dimension)
    history = []
    started = time.perf_counter()

    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=_init_worker,
        initargs=(args.modes, args.maximum_fractional_change, args.training_points),
    ) as pool:
        residual = list(pool.map(_residual_worker, [coefficients]))[0]
        current_loss = _loss(residual, coefficients, args.regularization)
        for iteration in range(1, args.iterations + 1):
            vectors = []
            for index in range(dimension):
                plus = coefficients.copy(); plus[index] += args.difference_step
                minus = coefficients.copy(); minus[index] -= args.difference_step
                vectors.extend((np.clip(plus, -1, 1), np.clip(minus, -1, 1)))
            evaluated = list(pool.map(_residual_worker, vectors, chunksize=1))
            jacobian = np.column_stack([
                (evaluated[2*i] - evaluated[2*i+1]) / (2 * args.difference_step)
                for i in range(dimension)
            ])
            normal = jacobian.T @ jacobian + args.regularization * np.eye(dimension)
            gradient = jacobian.T @ residual + args.regularization * coefficients
            step = -np.linalg.solve(normal, gradient)
            norm = float(np.linalg.norm(step))
            if norm > args.trust_radius:
                step *= args.trust_radius / norm
            trial_vectors = [np.clip(coefficients + alpha * step, -1, 1) for alpha in (1.0, 0.5, 0.25, 0.125)]
            trial_residuals = list(pool.map(_residual_worker, trial_vectors))
            choices = [(current_loss, coefficients, residual)] + [
                (_loss(r, c, args.regularization), c, r)
                for c, r in zip(trial_vectors, trial_residuals, strict=True)
            ]
            new_loss, coefficients, residual = min(choices, key=lambda value: value[0])
            accepted = new_loss < current_loss
            current_loss = new_loss
            elapsed = time.perf_counter() - started
            remaining = elapsed / iteration * (args.iterations - iteration)
            history.append({"iteration": iteration, "loss": current_loss, "accepted": accepted, "elapsed_seconds": elapsed, "estimated_remaining_seconds": remaining})
            np.savetxt(output / "coefficients_checkpoint.csv", coefficients[None], delimiter=",", fmt="%.17g")
            pd.DataFrame(history).to_csv(output / "fit_history.csv", index=False)
            print(f"GN {iteration}/{args.iterations}: loss={current_loss:.8g}, accepted={accepted}, elapsed={elapsed/60:.1f} min, remaining={remaining/60:.1f} min", flush=True)

    predict, arrays = inverse._prediction_function(kernel, items, coefficients, 0.680)
    cursor = 0
    validation = {}
    for item in items:
        count = item.basis.shape[0]
        local = coefficients[cursor:cursor+count]
        cursor += count
        response = item.response(local)
        validation[item.name] = inverse._validate(item, local)
        inverse._write_response(output / f"{item.name}_corrected_reco.csv", response)
        inverse._write_response(output / f"{item.name}_delta_reco.csv", response - item.original)
    metadata = {
        "status": "constrained_equivalent_response_candidate",
        "optimizer": "parallel finite-difference Gauss-Newton with ridge and feasible response line search",
        "workers": args.workers,
        "modes_per_axis": args.modes,
        "dimension": dimension,
        "training_grid_per_panel": [args.training_points, args.training_points],
        "final_requested_grid_per_panel": [100, 100],
        "validation": validation,
        "history": history,
        "warning": "non-unique equivalent NuMI response; not a recovered collaboration response",
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(output, flush=True)


if __name__ == "__main__":
    main()
