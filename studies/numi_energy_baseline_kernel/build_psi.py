"""Build a public-dk2nu-informed NuMI psi(E,L,flavour) pilot.

This is a study output only.  It does not modify or feed the active analysis.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys
import time
from typing import Iterable

import awkward as ak
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import uproot


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from sterile_fit.output import (  # noqa: E402
    begin_output_batch,
    finish_output_batch,
    result_directory,
    write_csv,
    write_json,
)


FLAVOUR_PDGS = {"numu": 14, "numubar": -14, "nue": 12, "nuebar": -12}
ENERGY_EDGES_GEV = np.linspace(0.0, 5.0, 51)
BASELINE_EDGES_KM = np.linspace(0.05, 0.75, 71)
BRANCHES = {
    "ntype": "dk2nu/decay/decay.ntype",
    "ptype": "dk2nu/decay/decay.ptype",
    "vx": "dk2nu/decay/decay.vx",
    "vy": "dk2nu/decay/decay.vy",
    "vz": "dk2nu/decay/decay.vz",
    "nimpwt": "dk2nu/decay/decay.nimpwt",
    "ray_energy": "dk2nu/nuray/nuray.E",
    "ray_weight": "dk2nu/nuray/nuray.wgt",
}


def _scalar(tree: uproot.TTree, branch: str) -> np.ndarray:
    return np.asarray(tree[branch].array(library="np"))


def _jagged_column(tree: uproot.TTree, branch: str, index: int) -> np.ndarray:
    return np.asarray(ak.to_numpy(tree[branch].array(library="ak")[:, index]))


def _metadata_value(tree: uproot.TTree, branch: str):
    values = tree[branch].array(library="ak").to_list()
    return values[0]


def read_public_file(url: str) -> tuple[pd.DataFrame, dict]:
    """Read only the fields needed for E-L weighting from one remote ROOT file."""
    root_file = uproot.open(url, timeout=180)
    metadata_tree = root_file["dkmetaTree"]
    names = _metadata_value(metadata_tree, "dkmeta/location/location.name")
    try:
        location_index = names.index("MicroBooNE")
    except ValueError as error:
        raise ValueError(f"MicroBooNE location is absent from {url}") from error

    location = np.array(
        [
            _metadata_value(metadata_tree, "dkmeta/location/location.x")[location_index],
            _metadata_value(metadata_tree, "dkmeta/location/location.y")[location_index],
            _metadata_value(metadata_tree, "dkmeta/location/location.z")[location_index],
        ],
        dtype=float,
    )
    tree = root_file["dk2nuTree"]
    vx = _scalar(tree, BRANCHES["vx"])
    vy = _scalar(tree, BRANCHES["vy"])
    vz = _scalar(tree, BRANCHES["vz"])
    energy = _jagged_column(tree, BRANCHES["ray_energy"], location_index)
    ray_weight = _jagged_column(tree, BRANCHES["ray_weight"], location_index)
    importance_weight = _scalar(tree, BRANCHES["nimpwt"])
    baseline = np.sqrt(
        (vx - location[0]) ** 2 + (vy - location[1]) ** 2 + (vz - location[2]) ** 2
    ) / 100_000.0
    table = pd.DataFrame(
        {
            "neutrino_pdg": _scalar(tree, BRANCHES["ntype"]).astype(int),
            "parent_pdg": _scalar(tree, BRANCHES["ptype"]).astype(int),
            "energy_GeV": energy,
            "baseline_km": baseline,
            "geometry_importance_weight": ray_weight * importance_weight,
        }
    )
    metadata = {
        "url": url,
        "entries": int(tree.num_entries),
        "microboone_location_index": int(location_index),
        "microboone_location_numi_cm": location.tolist(),
        "beamsim": _metadata_value(metadata_tree, "dkmeta/beamsim"),
        "physics": _metadata_value(metadata_tree, "dkmeta/physics"),
        "target": _metadata_value(metadata_tree, "dkmeta/tgtcfg"),
        "horn": _metadata_value(metadata_tree, "dkmeta/horncfg"),
        "decay_volume": _metadata_value(metadata_tree, "dkmeta/dkvolcfg"),
        "simulated_pot": float(_metadata_value(metadata_tree, "dkmeta/pots")),
    }
    return table, metadata


def weighted_effective_entries(weights: np.ndarray) -> float:
    denominator = float(np.square(weights).sum())
    return 0.0 if denominator == 0.0 else float(weights.sum() ** 2 / denominator)


def conditional_baseline_histogram(
    events: pd.DataFrame,
    neutrino_pdg: int,
    energy_edges: np.ndarray = ENERGY_EDGES_GEV,
    baseline_edges: np.ndarray = BASELINE_EDGES_KM,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Return q(L-bin|E-bin, flavour), with explicit nearest-bin fallback."""
    selected = events.loc[
        (events["neutrino_pdg"] == neutrino_pdg)
        & np.isfinite(events["energy_GeV"])
        & np.isfinite(events["baseline_km"])
        & np.isfinite(events["geometry_importance_weight"])
        & (events["geometry_importance_weight"] > 0.0)
    ]
    histogram, _, _ = np.histogram2d(
        selected["energy_GeV"],
        selected["baseline_km"],
        bins=(energy_edges, baseline_edges),
        weights=selected["geometry_importance_weight"],
    )
    counts, _, _ = np.histogram2d(
        selected["energy_GeV"],
        selected["baseline_km"],
        bins=(energy_edges, baseline_edges),
    )
    weighted_square, _, _ = np.histogram2d(
        selected["energy_GeV"],
        selected["baseline_km"],
        bins=(energy_edges, baseline_edges),
        weights=np.square(selected["geometry_importance_weight"]),
    )
    row_sum = histogram.sum(axis=1)
    nonempty = np.flatnonzero(row_sum > 0.0)
    if nonempty.size == 0:
        raise ValueError(f"no usable public dk2nu events for PDG {neutrino_pdg}")
    source_rows = np.arange(histogram.shape[0])
    fallback = np.zeros(histogram.shape[0], dtype=bool)
    for row in np.flatnonzero(row_sum == 0.0):
        source = int(nonempty[np.argmin(np.abs(nonempty - row))])
        histogram[row] = histogram[source]
        source_rows[row] = source
        fallback[row] = True
    probability = histogram / histogram.sum(axis=1, keepdims=True)
    raw_sum = row_sum
    square_sum = weighted_square.sum(axis=1)
    effective = np.divide(
        np.square(raw_sum),
        square_sum,
        out=np.zeros_like(raw_sum),
        where=square_sum > 0.0,
    )
    diagnostics = pd.DataFrame(
        {
            "energy_low_GeV": energy_edges[:-1],
            "energy_high_GeV": energy_edges[1:],
            "raw_event_count_in_EL_window": counts.sum(axis=1).astype(int),
            "weighted_effective_entries": effective,
            "used_nearest_energy_fallback": fallback,
            "conditional_source_energy_bin": source_rows,
        }
    )
    return probability, diagnostics


def load_flux(horn: str, flavour: str, directory: Path) -> pd.DataFrame:
    path = directory / f"numi_{horn}_{flavour}_flux.csv"
    table = pd.read_csv(path)
    if not np.allclose(table["energy_low_GeV"], ENERGY_EDGES_GEV[:-1]) or not np.allclose(
        table["energy_high_GeV"], ENERGY_EDGES_GEV[1:]
    ):
        raise ValueError(f"unexpected energy binning in {path}")
    return table


def make_psi_table(
    flux: pd.DataFrame,
    conditional_mass: np.ndarray,
    diagnostics: pd.DataFrame,
    *,
    horn: str,
    flavour: str,
    shape_flavour: str,
    shape_method: str,
) -> pd.DataFrame:
    delta_l = np.diff(BASELINE_EDGES_KM)
    phi = flux["flux_per_POT_per_cm2_per_100MeV"].to_numpy(dtype=float)
    density = phi[:, None] * conditional_mass / delta_l[None, :]
    rows = []
    for energy_index in range(phi.size):
        for baseline_index in range(delta_l.size):
            rows.append(
                {
                    "horn_mode": horn.upper(),
                    "flavour": flavour,
                    "energy_low_GeV": ENERGY_EDGES_GEV[energy_index],
                    "energy_high_GeV": ENERGY_EDGES_GEV[energy_index + 1],
                    "energy_center_GeV": 0.5 * (ENERGY_EDGES_GEV[energy_index] + ENERGY_EDGES_GEV[energy_index + 1]),
                    "baseline_low_km": BASELINE_EDGES_KM[baseline_index],
                    "baseline_high_km": BASELINE_EDGES_KM[baseline_index + 1],
                    "baseline_center_km": 0.5 * (BASELINE_EDGES_KM[baseline_index] + BASELINE_EDGES_KM[baseline_index + 1]),
                    "conditional_probability_mass": conditional_mass[energy_index, baseline_index],
                    "psi_per_POT_per_cm2_per_100MeV_per_km": density[energy_index, baseline_index],
                    "published_phi_per_POT_per_cm2_per_100MeV": phi[energy_index],
                    "shape_flavour_from_public_rhc": shape_flavour,
                    "shape_method": shape_method,
                    "weighted_effective_entries": diagnostics.loc[energy_index, "weighted_effective_entries"],
                    "used_nearest_energy_fallback": diagnostics.loc[energy_index, "used_nearest_energy_fallback"],
                }
            )
    return pd.DataFrame(rows)


def closure_table(psi: pd.DataFrame) -> pd.DataFrame:
    work = psi.copy()
    work["integrated_phi"] = work["psi_per_POT_per_cm2_per_100MeV_per_km"] * (
        work["baseline_high_km"] - work["baseline_low_km"]
    )
    grouped = work.groupby(["horn_mode", "flavour", "energy_low_GeV", "energy_high_GeV"], as_index=False).agg(
        integrated_phi=("integrated_phi", "sum"),
        target_phi=("published_phi_per_POT_per_cm2_per_100MeV", "first"),
    )
    grouped["absolute_difference"] = grouped["integrated_phi"] - grouped["target_phi"]
    grouped["relative_difference"] = np.divide(
        grouped["absolute_difference"],
        grouped["target_phi"],
        out=np.zeros(len(grouped), dtype=float),
        where=grouped["target_phi"].to_numpy() != 0.0,
    )
    return grouped


def plot_exposure_weighted(psi: pd.DataFrame, output_path: Path, fractions: dict[str, float]) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True, constrained_layout=True)
    for axis, flavour in zip(axes.flat, FLAVOUR_PDGS):
        subset = psi.loc[psi["flavour"] == flavour].copy()
        subset["exposure_weighted_density"] = subset["psi_per_POT_per_cm2_per_100MeV_per_km"] * subset[
            "horn_mode"
        ].str.lower().map(fractions)
        grid = subset.groupby(["energy_center_GeV", "baseline_center_km"])["exposure_weighted_density"].sum().unstack()
        positive = grid.to_numpy()[grid.to_numpy() > 0.0]
        floor = positive.min() if positive.size else 1.0
        colour = axis.pcolormesh(
            ENERGY_EDGES_GEV,
            BASELINE_EDGES_KM,
            np.log10(np.maximum(grid.to_numpy().T, floor)),
            shading="auto",
            cmap="viridis",
        )
        axis.set_title(flavour)
        axis.set_xlabel(r"$E_\nu$ [GeV]")
        axis.set_ylabel(r"$L$ [km]")
        figure.colorbar(colour, ax=axis, label=r"$\log_{10}\psi(E_\nu,L)$")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.suptitle("MicroBooNE NuMI public-flux-anchored energy--baseline pilot")
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_conditional_slices(psi: pd.DataFrame, output_path: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True, constrained_layout=True)
    requested = [0.25, 0.75, 1.25, 2.05]
    for axis, flavour in zip(axes.flat, FLAVOUR_PDGS):
        rhc = psi.loc[(psi["flavour"] == flavour) & (psi["horn_mode"] == "RHC")]
        for energy in requested:
            centers = rhc["energy_center_GeV"].unique()
            selected_energy = centers[np.argmin(np.abs(centers - energy))]
            row = rhc.loc[rhc["energy_center_GeV"] == selected_energy]
            axis.step(
                row["baseline_center_km"],
                row["conditional_probability_mass"],
                where="mid",
                label=f"E={selected_energy:.2f} GeV",
            )
        axis.set_title(f"RHC {flavour}")
        axis.set_xlabel(r"$L$ [km]")
        axis.set_ylabel(r"$P(L\;\mathrm{bin}\mid E,\nu)$")
        axis.legend(fontsize=8)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def parent_summary(events: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for flavour, pdg in FLAVOUR_PDGS.items():
        selected = events.loc[events["neutrino_pdg"] == pdg]
        for parent, group in selected.groupby("parent_pdg"):
            weights = group["geometry_importance_weight"].to_numpy(dtype=float)
            rows.append(
                {
                    "flavour": flavour,
                    "parent_pdg": int(parent),
                    "raw_events": int(len(group)),
                    "weight_sum": float(weights.sum()),
                    "weighted_effective_entries": weighted_effective_entries(weights),
                }
            )
    result = pd.DataFrame(rows)
    result["weight_fraction_within_flavour"] = result["weight_sum"] / result.groupby("flavour")["weight_sum"].transform("sum")
    return result


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--files", type=int, default=4, help="Number of public RHC dk2nu files, starting at index zero")
    parser.add_argument("--batch", default=None, help="Optional output batch label")
    parser.add_argument("--source-config", type=Path, default=Path(__file__).with_name("sources.json"))
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    if arguments.files < 1 or arguments.files > 100:
        raise ValueError("--files must be between 1 and 100")
    source_config = json.loads(arguments.source_config.read_text(encoding="utf-8"))
    base_url = source_config["public_dk2nu_directory"]
    template = source_config["file_template"]
    urls = [base_url + template.format(index=index) for index in range(arguments.files)]
    event_tables = []
    file_metadata = []
    failed_urls = []
    for ordinal, url in enumerate(urls, start=1):
        print(f"Reading public dk2nu file {ordinal}/{len(urls)}: {url}", flush=True)
        for attempt in range(1, 4):
            try:
                table, metadata = read_public_file(url)
                event_tables.append(table)
                file_metadata.append(metadata)
                break
            except (OSError, TimeoutError, TypeError) as error:
                if attempt == 3:
                    failed_urls.append({"url": url, "error": repr(error)})
                    print(f"Skipping after 3 failed reads: {url}", flush=True)
                else:
                    print(f"Remote read failed; retry {attempt}/2: {error!r}", flush=True)
                    time.sleep(2.0)
    if not event_tables:
        raise RuntimeError("No public dk2nu file could be read")
    events = pd.concat(event_tables, ignore_index=True)

    expected = source_config["beam_metadata_expected"]
    for metadata in file_metadata:
        for key, value in expected.items():
            if metadata[key] != value:
                raise ValueError(f"public dk2nu metadata mismatch for {key}: {metadata[key]!r} != {value!r}")

    conditionals = {}
    diagnostics = {}
    for flavour, pdg in FLAVOUR_PDGS.items():
        conditionals[flavour], diagnostics[flavour] = conditional_baseline_histogram(events, pdg)

    flux_directory = REPOSITORY_ROOT / source_config["microboone_flux_directory"]
    proxy = source_config["fhc_conditional_proxy"]
    psi_tables = []
    for horn in ("rhc", "fhc"):
        for flavour in FLAVOUR_PDGS:
            shape_flavour = flavour if horn == "rhc" else proxy[flavour]
            method = "direct_public_rhc_dk2nu" if horn == "rhc" else "charge_conjugate_public_rhc_proxy"
            psi_tables.append(
                make_psi_table(
                    load_flux(horn, flavour, flux_directory),
                    conditionals[shape_flavour],
                    diagnostics[shape_flavour],
                    horn=horn,
                    flavour=flavour,
                    shape_flavour=shape_flavour,
                    shape_method=method,
                )
            )
    psi = pd.concat(psi_tables, ignore_index=True)
    closure = closure_table(psi)

    begin_output_batch(arguments.batch)
    output_directory = result_directory("studies", "numi_energy_baseline_kernel", "results")
    output_directory.mkdir(parents=True, exist_ok=True)
    write_csv(psi, output_directory / "psi_energy_baseline_all_modes.csv")
    for (horn, flavour), table in psi.groupby(["horn_mode", "flavour"], sort=False):
        write_csv(table, output_directory / f"psi_{horn.lower()}_{flavour}.csv")
    fractions = source_config["exposure_fractions"]
    combined = psi.copy()
    combined["exposure_fraction"] = combined["horn_mode"].str.lower().map(fractions)
    combined["weighted_psi"] = combined["psi_per_POT_per_cm2_per_100MeV_per_km"] * combined["exposure_fraction"]
    combined = combined.groupby(
        ["flavour", "energy_low_GeV", "energy_high_GeV", "energy_center_GeV", "baseline_low_km", "baseline_high_km", "baseline_center_km"],
        as_index=False,
    )["weighted_psi"].sum().rename(columns={"weighted_psi": "psi_per_POT_per_cm2_per_100MeV_per_km"})
    write_csv(combined, output_directory / "psi_exposure_weighted_four_flavours.csv")
    write_csv(closure, output_directory / "marginal_closure.csv")
    write_csv(parent_summary(events), output_directory / "public_rhc_parent_summary.csv")
    for flavour, table in diagnostics.items():
        write_csv(table, output_directory / f"public_rhc_{flavour}_energy_statistics.csv")
    plot_exposure_weighted(psi, output_directory / "psi_exposure_weighted_heatmaps.png", fractions)
    plot_conditional_slices(psi, output_directory / "rhc_conditional_baseline_slices.png")

    max_closure = float(np.max(np.abs(closure["relative_difference"])))
    metadata = {
        "scientific_status": "geometry-informed public-dk2nu pilot; not an official MicroBooNE Nature-analysis input",
        "definition": "psi is a baseline density whose L integral exactly equals the registered MicroBooNE PDF-extracted phi(E) in each horn/flavour/energy bin",
        "energy_edges_GeV": ENERGY_EDGES_GEV.tolist(),
        "baseline_edges_km": BASELINE_EDGES_KM.tolist(),
        "public_files_read": file_metadata,
        "public_files_failed": failed_urls,
        "source_config": source_config,
        "source_config_sha256": sha256(arguments.source_config.read_bytes()).hexdigest(),
        "maximum_absolute_relative_marginal_closure_error": max_closure,
        "limitations": [
            "public dk2nu sample is Geant4 9.2 medium-energy -200 kA RHC, not the final updated MicroBooNE production",
            "no PPFX central-value event weights were available in the public file",
            "FHC conditional shapes use charge-conjugate RHC flavour distributions as an explicit proxy",
            "empty public-sample energy rows use the nearest populated row of the same shape flavour",
            "the registered absolute MicroBooNE phi(E) curves originate from vector extraction of Note 1129 figures",
            "outputs are not connected to the active oscillation analysis",
        ],
    }
    write_json(output_directory / "metadata.json", metadata)
    finish_output_batch(sys.argv)
    print(f"Wrote {output_directory}")
    print(f"Maximum marginal closure error: {max_closure:.3e}")


if __name__ == "__main__":
    main()
