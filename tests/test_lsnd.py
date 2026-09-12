import csv
import json
import subprocess
import sys

import numpy as np

from sterile_fit.experiments.lsnd.final_2001 import (
    PUBLISHED, core_mapping_rows, load_published_summary,
    three_plus_one_dar_appearance_probability,
    three_plus_one_parameters_from_appearance_amplitude,
)
from sterile_fit.experiments.lsnd.public_rate_approximation import (
    LSNDDARRateInputs,
    rate_negative_two_log_likelihood,
    three_plus_one_average_probability,
)
from sterile_fit.paths import REPOSITORY_ROOT


def test_public_summary_shape_and_printed_best_fit():
    with PUBLISHED.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    summary = load_published_summary()
    assert len(rows) == 9
    assert summary.dar_excess_events == 87.9
    assert summary.dar_probability_percent == 0.264
    assert summary.best_fit_delta_m2_eV2 == 1.2
    assert summary.best_fit_sin2_2theta_mue == 0.003


def test_source_manifest_matches_the_retained_validated_pdf():
    manifest = PUBLISHED.parent.parent / "sources" / "MANIFEST.csv"
    with manifest.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    primary = next(row for row in rows if row["source_id"] == "aguilar_2001_final")
    assert len(rows) == 3
    assert primary["url"] == "https://arxiv.org/pdf/hep-ex/0104049"
    assert primary["local_file"] == (
        "data/experiments/lsnd/sources/aguilar_2001_hep-ex_0104049.pdf"
    )
    assert primary["sha256"] == (
        "0c62d411eefb93a6146403be31439d250b22b1c9af9f627fabd26b990c5a09f2"
    )


def test_dar_units_and_probability_zero_point():
    probability = three_plus_one_dar_appearance_probability(
        np.array([40.0]), np.array([30.0]), 1.2, 0.003
    )
    assert np.allclose(probability, 0.003 * np.sin(1.267 * 1.2 * 30.0 / 40.0) ** 2)
    assert np.array_equal(
        three_plus_one_dar_appearance_probability(np.array([20.0, 52.8]), 30.0, 1.2, 0.0),
        np.zeros(2),
    )


def test_effective_amplitude_is_exact_common_core_identity():
    point = three_plus_one_parameters_from_appearance_amplitude(1.2, 0.003)
    assert np.isclose(point.sin2_2theta_mue_exact, 0.003)


def test_core_mapping_is_parseable_and_appearance_only():
    rows = core_mapping_rows(load_published_summary())
    assert len(rows) == 5
    assert all(20.0 <= row["energy_MeV"] <= 52.8 for row in rows)
    assert all(0.0 <= row["appearance_probability"] <= 0.003 for row in rows)


def test_no_unpublished_covariance_or_surface_is_claimed():
    source_text = (PUBLISHED.parent.parent / "README.md").read_text(encoding="utf-8")
    assert "does **not** publish an event table" in source_text
    assert "It is not an active raw input" in source_text
    assert "not promoted" in source_text


def test_registered_entry_writes_parseable_core_mapping_output(tmp_path):
    output = tmp_path / "lsnd-core-mapping"
    subprocess.run(
        [sys.executable, "-B", "run.py", "lsnd", "--kind", "core-mapping",
         "--output-directory", str(output)],
        cwd=REPOSITORY_ROOT, check=True, capture_output=True, text=True,
    )
    with (output / "result.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    assert len(rows) == 5
    assert metadata["local_likelihood_surface"].startswith("not computed")


def test_public_rate_kernel_preserves_zero_and_amplitude_linearity():
    zero = three_plus_one_average_probability(1.2, 0.0)
    unit = three_plus_one_average_probability(1.2, 1.0)
    test = three_plus_one_average_probability(1.2, 0.003)
    assert zero == 0.0
    assert np.isclose(test, 0.003 * unit, rtol=1e-12, atol=1e-15)


def test_published_best_fit_is_compatible_with_public_rate_measurement():
    inputs = LSNDDARRateInputs()
    prediction = three_plus_one_average_probability(1.2, 0.003, inputs)
    assert 0.0 < prediction < 0.003
    assert rate_negative_two_log_likelihood(prediction, inputs) < 1.0


def test_rate_likelihood_is_minimal_at_the_published_average_probability():
    inputs = LSNDDARRateInputs()
    assert rate_negative_two_log_likelihood(
        inputs.observed_average_probability, inputs
    ) == 0.0
    assert rate_negative_two_log_likelihood(0.0, inputs) > 10.0
