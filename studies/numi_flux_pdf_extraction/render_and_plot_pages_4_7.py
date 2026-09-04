"""Render source PDF pages 4-7 and independently replot the stored flux CSVs.

The source pages are rendered locally with Poppler.  The four reconstructed
plots read only the eight promoted CSV arrays; they do not call the PDF path
extractor or any model/API.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from sterile_fit.output import result_directory
SOURCE_PDF = (
    ROOT
    / "studies"
    / "numi_flux_pdf_extraction"
    / "data"
    / "microboone_note_1129"
    / "MICROBOONE-NOTE-1129-PUB.pdf"
)
FLUX_DIRECTORY = (
    ROOT / "data" / "experiments" / "microboone" / "numi" / "inputs" / "flux_components"
)
OUTPUT_DIRECTORY = result_directory("studies", "numi_flux_pdf_extraction", "results") / "microboone_pages_4_7" / "page_comparison"
SOURCE_PAGE_DIRECTORY = OUTPUT_DIRECTORY / "source_pdf_pages"
REPLOT_DIRECTORY = OUTPUT_DIRECTORY / "replotted_from_stored_csv"

VALUE_COLUMN = "flux_per_POT_per_cm2_per_100MeV"
REQUIRED_COLUMNS = {
    "energy_low_GeV",
    "energy_high_GeV",
    "energy_center_GeV",
    VALUE_COLUMN,
    "is_censored",
}


def render_source_pages() -> list[Path]:
    SOURCE_PAGE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    products: list[Path] = []
    for page_number in range(4, 8):
        prefix = SOURCE_PAGE_DIRECTORY / f"microboone_note_1129_page_{page_number:02d}"
        subprocess.run(
            [
                "pdftoppm",
                "-f",
                str(page_number),
                "-l",
                str(page_number),
                "-r",
                "180",
                "-png",
                "-singlefile",
                str(SOURCE_PDF),
                str(prefix),
            ],
            check=True,
        )
        output = prefix.with_suffix(".png")
        if not output.is_file() or output.stat().st_size == 0:
            raise RuntimeError(f"Poppler did not produce {output}")
        products.append(output)
    return products


def read_component(horn_mode: str, flavour: str) -> pd.DataFrame:
    path = FLUX_DIRECTORY / f"numi_{horn_mode}_{flavour}_flux.csv"
    table = pd.read_csv(path)
    if not REQUIRED_COLUMNS.issubset(table.columns) or len(table) != 50:
        raise ValueError(f"unexpected stored flux schema or bin count: {path}")
    numeric = table[["energy_low_GeV", "energy_high_GeV", "energy_center_GeV", VALUE_COLUMN]].to_numpy(
        dtype=float
    )
    if not np.all(np.isfinite(numeric)) or np.any(numeric[:, 3] < 0.0):
        raise ValueError(f"stored flux contains invalid values: {path}")
    edges = np.concatenate(
        [table["energy_low_GeV"].to_numpy(dtype=float), [float(table["energy_high_GeV"].iloc[-1])]]
    )
    if not np.allclose(edges, np.linspace(0.0, 5.0, 51), rtol=0.0, atol=1e-12):
        raise ValueError(f"stored flux is not on the declared 0.1 GeV grid: {path}")
    return table


def draw_step(axis: plt.Axes, values: np.ndarray, censored: np.ndarray, *, colour: str) -> None:
    edges = np.linspace(0.0, 5.0, 51)
    axis.step(edges, np.r_[values, values[-1]], where="post", color=colour, linewidth=1.8)
    if np.any(censored):
        centers = (edges[:-1] + edges[1:]) / 2.0
        axis.scatter(
            centers[censored],
            values[censored],
            marker="x",
            s=24,
            color=colour,
            label="PDF-boundary censored",
            zorder=3,
        )


def plot_stored_component(horn_mode: str, flavour: str) -> Path:
    REPLOT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    table = read_component(horn_mode, flavour)
    values = table[VALUE_COLUMN].to_numpy(dtype=float)
    censored = table["is_censored"].astype(bool).to_numpy()
    mode_label = horn_mode.upper()
    flavour_labels = {
        "numu": r"$\nu_\mu$",
        "numubar": r"$\bar{\nu}_\mu$",
        "nue": r"$\nu_e$",
        "nuebar": r"$\bar{\nu}_e$",
    }
    figure, axis = plt.subplots(figsize=(7.5, 5.2), constrained_layout=True)
    draw_step(axis, values, censored, colour="tab:blue")
    axis.set_yscale("log")
    axis.set_xlim(0.0, 5.0)
    axis.set_xlabel(r"Neutrino energy $E_\nu$ [GeV]")
    axis.set_ylabel(r"Flux / POT / cm$^2$ / 100 MeV")
    axis.set_title(f"NuMI {mode_label} {flavour_labels[flavour]} flux")
    axis.grid(which="both", alpha=0.25)
    if np.any(censored):
        axis.legend(frameon=False, fontsize=8)
    output = REPLOT_DIRECTORY / f"numi_{horn_mode}_{flavour}_flux.png"
    figure.savefig(output, dpi=180)
    plt.close(figure)
    return output


def main() -> None:
    source_pages = render_source_pages()
    replots = [
        plot_stored_component(horn_mode, flavour)
        for horn_mode in ("fhc", "rhc")
        for flavour in ("numu", "numubar", "nue", "nuebar")
    ]
    for source in source_pages:
        print(f"WROTE source page: {source}")
    for replot in replots:
        print(f"WROTE stored-CSV replot: {replot}")


if __name__ == "__main__":
    main()
