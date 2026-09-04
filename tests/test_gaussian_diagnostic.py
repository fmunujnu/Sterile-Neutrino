"""Small diagnostic tests; no parameter scan and no new profiled Toys."""
import importlib.util
import numpy as np
import pytest
from sterile_fit.paths import REPOSITORY_ROOT
from sterile_fit.output import plot_statistic_calibration, plot_gaussian_toy_grid

spec = importlib.util.spec_from_file_location("gaussian_diagnostic", REPOSITORY_ROOT / "studies/three_plus_one_toy_distribution_fit/compare_covariance_gaussian.py")
diagnostic = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnostic)


def test_empty_tail_is_not_reported_as_exact_zero():
    score = diagnostic.score_distribution(np.array([-1., 0., 1.]), 0., 1., 4.)
    assert score['tail_count'] == 0
    assert score['toy_p_raw'] == 0
    assert score['toy_p_smoothed'] == .25
    assert score['tail_ci95_high'] > 0
    assert score['gaussian_mean'] == 0
    assert score['gaussian_variance'] == 1


def test_invalid_width_rejected():
    with pytest.raises(ValueError):
        diagnostic.score_distribution([1., 2.], 0., 0., 0.)


def test_shared_distribution_renderer(tmp_path):
    panel = dict(label="test", profiled_T=np.linspace(-2, 2, 30), fixed_T=np.linspace(-1.9, 2.1, 30), mean=0., sigma=1., observed_T=1.)
    output = tmp_path / "comparison.png"
    plot_statistic_calibration([panel, panel], output, title="Diagnostic test")
    assert output.stat().st_size > 0


def test_combined_plot_has_only_density_and_tail(tmp_path):
    import pandas as pd
    samples = pd.DataFrame([dict(point_index=p, generator=g, test_statistic=t)
                            for p in (0, 1) for g in ("3nu", "4nu") for t in np.linspace(-2, 2, 30)])
    scores = pd.DataFrame([dict(point_index=p, generator=g, variant="reprofiled",
                               gaussian_mean=0., gaussian_sigma=1., observed_T=.5,
                               delta_m2_41_eV2=1., sin2_2theta_mue=.003)
                           for p in (0, 1) for g in ("3nu", "4nu")])
    target = tmp_path / "combined.png"
    plot_gaussian_toy_grid(samples, scores, target)
    assert target.stat().st_size > 0


def test_plot_only_never_builds_physics(tmp_path, monkeypatch):
    import pandas as pd
    import sys
    pd.DataFrame({"test_statistic": [1.]}).to_csv(tmp_path/"samples_with_gaussian_p.csv", index=False)
    pd.DataFrame({"gaussian_mean": [0.]}).to_csv(tmp_path/"distribution_comparison.csv", index=False)
    def forbidden(*args, **kwargs):
        raise AssertionError("Plot-only must not rebuild analysis")
    captured = []
    monkeypatch.setattr(diagnostic, "build_three_plus_one_analysis", forbidden)
    monkeypatch.setattr(diagnostic, "plot_gaussian_toy_grid", lambda samples, scores, path: captured.append(path))
    monkeypatch.setattr(sys, "argv", ["compare", "--plot-only", str(tmp_path), "--layout", "combined"])
    diagnostic.main()
    assert captured == [tmp_path/"all_points_gaussian_vs_toy.png"]


def test_individual_adapter_exports_deviation_table(tmp_path, monkeypatch):
    import pandas as pd
    samples = pd.DataFrame([dict(point_index=0, generator=g, test_statistic=t, fixed_T_same_draw=t)
                            for g in ("3nu", "4nu") for t in [-1., 0., 1.]])
    scores = pd.DataFrame([dict(point_index=0, generator=g, variant="reprofiled", sample_size=3,
        gaussian_mean=0., gaussian_sigma=1., observed_T=.5, delta_m2_41_eV2=1., sin2_2theta_mue=.003,
        toy_mean=.2, toy_sigma=1.1, toy_p_raw=1/3, gaussian_p_at_observed=.3)
        for g in ("3nu", "4nu")])
    captured = []
    monkeypatch.setattr(diagnostic, "plot_statistic_calibration", lambda panels, path, **kwargs: captured.append(path))
    diagnostic.render_comparison(samples, scores, tmp_path)
    assert captured == [tmp_path/"point_00_gaussian_vs_toy.png"]
    result = pd.read_csv(tmp_path/"gaussian_deviations.csv")
    np.testing.assert_allclose(result.mean_bias_Toy_minus_Gaussian, .2)
    np.testing.assert_allclose(result.sigma_ratio_Toy_over_Gaussian, 1.1)
    np.testing.assert_allclose(result.tail_bias_raw_Toy_minus_Gaussian, 1/3-.3)
