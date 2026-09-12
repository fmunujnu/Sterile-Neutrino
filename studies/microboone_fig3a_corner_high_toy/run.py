from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import sys
from time import perf_counter

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from studies.microboone_profiled_toy_cls_band200.run import evaluate, initialize_worker  # noqa: E402


SOURCE = ROOT / "outputs/microboone_bnb_numi_joint/three_plus_one/fullgrid_fixed_toy5000_20260904/scan_fig3a_toy/result.csv"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--toys", type=int, default=5000)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260907)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    args.output_directory.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output_directory / "point_summary_checkpoint.csv"
    toy_path = args.output_directory / "profiled_toy_values.csv"
    completed = pd.read_csv(checkpoint) if checkpoint.exists() else pd.DataFrame()
    keys = set(zip(completed.mass, completed.amplitude)) if len(completed) else set()

    frame = pd.read_csv(SOURCE)
    selected = frame[
        frame.fixed_sin2_2theta_mue.between(.7, 1.0)
        & frame.fixed_delta_m2_41_eV2.between(.01, .05)
    ].sort_values(["fixed_delta_m2_41_eV2", "fixed_sin2_2theta_mue"])
    payloads = []
    for index, row in selected.reset_index(drop=True).iterrows():
        mass = float(row.fixed_delta_m2_41_eV2)
        amplitude = float(row.fixed_sin2_2theta_mue)
        if (mass, amplitude) not in keys:
            payloads.append(("fig3a", "appearance-profile", mass, amplitude,
                             args.toys, args.seed + index))

    rows = completed.to_dict("records") if len(completed) else []
    total = len(rows) + len(payloads)
    started = perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers, initializer=initialize_worker) as executor:
        futures = [executor.submit(evaluate, payload) for payload in payloads]
        for future in as_completed(futures):
            summary, toy_rows = future.result()
            rows.append(summary)
            pd.DataFrame(toy_rows).to_csv(
                toy_path, mode="a", header=not toy_path.exists(), index=False
            )
            pd.DataFrame(rows).sort_values(["mass", "amplitude"]).to_csv(checkpoint, index=False)
            done = len(rows)
            elapsed = perf_counter() - started
            new_done = done - len(completed)
            eta = (len(payloads) - new_done) * elapsed / max(new_done, 1)
            print(f"points={done}/{total} elapsed={elapsed:.1f}s ETA={eta:.1f}s", flush=True)
    print(args.output_directory)


if __name__ == "__main__":
    main()
