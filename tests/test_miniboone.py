import csv
import hashlib

import numpy as np
import pandas as pd

from sterile_fit.experiments.miniboone.official_2020 import (
    RAW,
    load_release,
    prediction_and_covariance,
    signal_counts,
    three_plus_one_appearance_probability,
    three_plus_one_parameters_from_appearance_amplitude,
)


def test_official_release_shapes_and_likelihood_minimum():
    data = load_release()
    assert data.fractional_covariance.shape == (60, 60)
    surface = pd.read_csv(RAW / "likihood_surface_contNunubar.txt", sep=r"\s+")
    best = surface.loc[surface["-2ln(L)"].idxmin()]
    assert len(surface) == 36100
    assert np.isclose(best.dm2, 0.0431696)
    assert np.isclose(best.sintheta, 0.807239)
    assert np.isclose(best["-2ln(L)"], 344.519)


def test_download_manifest_matches_every_raw_file():
    with (RAW / "SHA256SUMS.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 19
    for row in rows:
        assert row["url"].startswith("https://rtayloe.pages.iu.edu/MB/data-releases/nue2020/")
        assert hashlib.sha256((RAW / row["file"]).read_bytes()).hexdigest() == row["sha256"]


def test_probability_units_and_signal_linearity():
    probability = three_plus_one_appearance_probability(
        np.array([500.0]), np.array([50000.0]), 1.2, 0.003
    )
    assert np.allclose(probability, 0.003 * np.sin(1.267 * 1.2) ** 2)
    parameters = three_plus_one_parameters_from_appearance_amplitude(1.2, 0.003)
    assert np.isclose(parameters.sin2_2theta_mue_exact, 0.003)
    data = load_release()
    low = signal_counts(data.nu_full_transmutation, data.electron_edges_MeV, 1.2, 0.003)
    unit = signal_counts(data.nu_full_transmutation, data.electron_edges_MeV, 1.2, 1.0)
    assert np.allclose(low, 0.003 * unit, rtol=1e-12, atol=1e-12)


def test_collapsed_covariance_is_symmetric_positive_definite():
    data = load_release()
    observation, prediction, covariance, _, _ = prediction_and_covariance(
        data, 0.0431696, 0.807239
    )
    assert observation.shape == prediction.shape == (38,)
    assert covariance.shape == (38, 38)
    assert np.allclose(covariance, covariance.T, atol=1e-10)
    assert np.linalg.eigvalsh(covariance).min() > 0
