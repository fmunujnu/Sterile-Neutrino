import csv
import hashlib

import numpy as np
import pandas as pd

from sterile_fit.experiments.miniboone.official_2020 import (
    RAW,
    chi_square_from_signals,
    load_release,
    prediction_and_covariance,
    signal_counts,
    three_plus_one_appearance_probability,
    three_plus_one_parameters_from_appearance_amplitude,
)
from sterile_fit.core.one_plus_three_plus_one import OnePlusThreePlusOneVacuumModel
from sterile_fit.experiments.miniboone.one_plus_three_plus_one import (
    build_fixed_mass_templates,
    event_appearance_probability,
    scan_fixed_mass_product_plane,
    scan_fixed_mass_slices,
    scan_profiled_mixing_plane,
    signals_from_templates,
    symmetric_parameters_from_appearance_amplitudes,
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


def test_one_plus_three_plus_one_event_probability_uses_released_true_baseline():
    data = load_release()
    events = data.nu_full_transmutation[:12]
    parameters = symmetric_parameters_from_appearance_amplitudes(
        0.1, 1.0, 0.01, 0.02, 0.7
    )
    vectorised = event_appearance_probability(
        events[:, 1], events[:, 2], parameters, antineutrino=False
    )
    model = OnePlusThreePlusOneVacuumModel(parameters)
    scalar_reference = np.asarray([
        model.probability(
            1, 0, np.asarray([energy_MeV / 1000.0]), baseline_cm / 100000.0
        )[0]
        for energy_MeV, baseline_cm in zip(events[:, 1], events[:, 2])
    ])
    assert np.allclose(vectorised, scalar_reference, rtol=0.0, atol=2e-15)


def test_one_plus_three_plus_one_single_state_limit_recovers_miniboone_three_plus_one():
    data = load_release()
    events = data.nu_full_transmutation[:30]
    parameters = symmetric_parameters_from_appearance_amplitudes(
        1.2, 7.0, 0.003, 0.0, 0.0
    )
    candidate = event_appearance_probability(
        events[:, 1], events[:, 2], parameters, antineutrino=False
    )
    reference = three_plus_one_appearance_probability(
        events[:, 1], events[:, 2], 1.2, 0.003
    )
    assert np.allclose(candidate, reference, rtol=0.0, atol=2e-15)


def test_one_plus_three_plus_one_templates_equal_direct_event_histograms():
    data = load_release()
    parameters = symmetric_parameters_from_appearance_amplitudes(
        0.1, 1.0, 0.01, 0.02, 0.7
    )
    templates = build_fixed_mass_templates(data, 0.1, 1.0)
    templated_nu, templated_nubar = signals_from_templates(
        templates, 0.01, 0.02, 0.7
    )

    def direct(events, *, antineutrino):
        probability = event_appearance_probability(
            events[:, 1], events[:, 2], parameters, antineutrino=antineutrino
        )
        return np.histogram(
            events[:, 0],
            bins=data.electron_edges_MeV,
            weights=probability * events[:, 3] / len(events),
        )[0]

    assert np.allclose(
        templated_nu, direct(data.nu_full_transmutation, antineutrino=False),
        rtol=2e-12, atol=2e-11,
    )
    assert np.allclose(
        templated_nubar, direct(data.nubar_full_transmutation, antineutrino=True),
        rtol=2e-12, atol=2e-11,
    )


def test_one_plus_three_plus_one_small_no_toy_scan_is_complete():
    data = load_release()
    result = scan_fixed_mass_slices(
        data,
        [(0.1, 1.0)],
        np.asarray([0.003, 0.01]),
        phase_grid_points=9,
        report_every=0,
    )
    assert len(result) == 4
    assert np.all(np.isfinite(result["negative_2_log_likelihood"]))
    assert np.isclose(
        result["delta_negative_2_log_likelihood_within_mass_slice"].min(), 0.0
    )


def test_one_plus_three_plus_one_profiles_masses_for_each_mixing_point():
    data = load_release()
    result = scan_profiled_mixing_plane(
        data,
        np.asarray([0.003, 0.01]),
        np.asarray([0.1, 1.0]),
        coarse_phase_points=5,
        refined_mass_candidates=2,
        refined_phase_grid_points=9,
        report_every=0,
    )
    assert len(result) == 4
    assert set(result["profiled_delta_m2_41_absolute_eV2"]).issubset({0.1, 1.0})
    assert set(result["profiled_delta_m2_51_eV2"]).issubset({0.1, 1.0})
    assert np.all(np.isfinite(result["chi_square"]))
    assert np.isclose(result["delta_chi_square"].min(), 0.0)


def test_one_plus_three_plus_one_fixed_paper_mass_scan_uses_product_axes():
    data = load_release()
    products = np.asarray([0.001, 0.01])
    result = scan_fixed_mass_product_plane(
        data,
        products,
        delta_m2_41_absolute_eV2=0.9,
        delta_m2_51_eV2=0.5,
        cp_phase_mue_rad=0.0,
        report_every=0,
    )
    assert len(result) == 4
    assert np.allclose(
        result["appearance_amplitude_state4"],
        4.0 * result["absolute_Ue4_Umu4"] ** 2,
    )
    assert np.allclose(
        result["appearance_amplitude_state5"],
        4.0 * result["absolute_Ue5_Umu5"] ** 2,
    )
    assert np.all(result["cp_phase_mue_rad"] == 0.0)
    assert np.all(np.isfinite(result["chi_square"]))
    assert np.isclose(result["delta_chi_square"].min(), 0.0)


def test_declared_miniboone_chi_square_is_the_quadratic_term_only():
    data = load_release()
    _, _, _, signal_nu, signal_nubar = prediction_and_covariance(
        data, 1.2, 0.003
    )
    observation, prediction, covariance, _, _ = prediction_and_covariance(
        data, 1.2, 0.003
    )
    residual = observation - prediction
    expected = float(residual @ np.linalg.solve(covariance, residual))
    assert chi_square_from_signals(data, signal_nu, signal_nubar) == expected
