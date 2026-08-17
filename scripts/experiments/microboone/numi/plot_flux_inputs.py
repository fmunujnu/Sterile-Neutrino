"""Plot only the eight visible MicroBooNE NuMI flux-component CSV files."""

from __future__ import annotations

import argparse
import csv
from hashlib import sha256
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


ROOT = Path(__file__).resolve().parents[4]
INPUT_DIRECTORY = ROOT / "data" / "experiments" / "microboone" / "numi" / "inputs" / "flux_components"
DEFAULT_OUTPUT = ROOT / "outputs" / "spectra" / "microboone" / "numi" / "flux_components.png"
FLAVOURS = ("numu", "numubar", "nue", "nuebar")
HORN_MODES = ("fhc", "rhc")
EXPECTED_COLUMNS = [
    "energy_low_GeV",
    "energy_high_GeV",
    "energy_center_GeV",
    "flux_per_POT_per_cm2_per_100MeV",
    "is_censored",
    "derivation",
]
DISPLAY_NAMES = {
    "numu": r"$\nu_\mu$",
    "numubar": r"$\bar\nu_\mu$",
    "nue": r"$\nu_e$",
    "nuebar": r"$\bar\nu_e$",
}


def _load_one(path: Path) -> pd.DataFrame:
    table = pd.read_csv(path)
    if list(table.columns) != EXPECTED_COLUMNS or table.shape[0] != 50:
        raise ValueError(f"unexpected NuMI flux schema or bin count: {path}")
    expected_low = np.arange(0.0, 5.0, 0.1)
    expected_high = expected_low + 0.1
    if not np.allclose(table["energy_low_GeV"], expected_low, atol=1e-12):
        raise ValueError(f"energy_low_GeV is not the declared 0.1 GeV grid: {path}")
    if not np.allclose(table["energy_high_GeV"], expected_high, atol=1e-12):
        raise ValueError(f"energy_high_GeV is not the declared 0.1 GeV grid: {path}")
    flux = table["flux_per_POT_per_cm2_per_100MeV"].to_numpy(dtype=float)
    if not np.all(np.isfinite(flux)) or np.any(flux < 0.0):
        raise ValueError(f"flux must be finite and non-negative: {path}")
    if table["is_censored"].astype(bool).any():
        raise ValueError(f"censored PDF values cannot enter the NuMI input plot: {path}")
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot the eight visible NuMI flux inputs.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 4, figsize=(15.5, 7.4), sharex=True)
    summary_rows: list[dict[str, object]] = []

    for row, horn_mode in enumerate(HORN_MODES):
        for column, flavour in enumerate(FLAVOURS):
            path = INPUT_DIRECTORY / f"numi_{horn_mode}_{flavour}_flux.csv"
            table = _load_one(path)
            flux = table["flux_per_POT_per_cm2_per_100MeV"].to_numpy(dtype=float)
            edges = np.concatenate(
                [table["energy_low_GeV"].to_numpy(dtype=float), [float(table["energy_high_GeV"].iloc[-1])]]
            )
            values = np.concatenate([flux, [flux[-1]]])
            axis = axes[row, column]
            axis.step(edges, values, where="post", color="#0072B2", linewidth=1.8)
            axis.set_yscale("log")
            axis.set_xlim(0.0, 5.0)
            axis.grid(alpha=0.2, which="both")
            axis.set_title(f"{horn_mode.upper()}  {DISPLAY_NAMES[flavour]}")
            axis.set_xlabel("Neutrino energy [GeV]")
            if column == 0:
                axis.set_ylabel(r"Flux [$\nu$/POT/cm$^2$/100 MeV]")

            summary_rows.append(
                {
                    "horn_mode": horn_mode.upper(),
                    "flavour": flavour,
                    "bins": len(table),
                    "energy_min_GeV": float(table["energy_low_GeV"].iloc[0]),
                    "energy_max_GeV": float(table["energy_high_GeV"].iloc[-1]),
                    "minimum_flux": float(np.min(flux)),
                    "maximum_flux": float(np.max(flux)),
                    "sum_of_0p1_GeV_bin_flux_values": float(np.sum(flux)),
                    "sha256": sha256(path.read_bytes()).hexdigest().upper(),
                    "source_file": str(path),
                }
            )

    figure.suptitle("MicroBooNE NuMI New Flux inputs reconstructed from PDF pages 4-7")
    figure.tight_layout()
    figure.savefig(arguments.output, dpi=240, bbox_inches="tight")
    plt.close(figure)
    with arguments.output.with_suffix(".csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    arguments.output.with_suffix(".metadata.json").write_text(
        json.dumps(
            {
                "status": "regenerable_spectrum_output",
                "numerical_input_directory": str(INPUT_DIRECTORY),
                "plotted_components": 8,
                "bins_per_component": 50,
                "plot_is_numerical_input": False,
                "scientific_boundary": "PDF-vector reconstructed flux inputs; not official unpublished arrays",
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(arguments.output)


if __name__ == "__main__":
    main()
