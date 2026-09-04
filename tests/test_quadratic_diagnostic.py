"""Independent analytic special cases for the isolated CF-inversion study."""
import importlib.util
from pathlib import Path
import numpy as np
from scipy.stats import norm, ncx2
from sterile_fit.core.calibration import GaussianHypothesis, _quadratic_difference_moments
from sterile_fit.output import plot_statistic_calibration

spec = importlib.util.spec_from_file_location("quadratic_pilot", Path(__file__).resolve().parents[1]/"studies/quadratic_toy_check/quadratic.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_pure_linear_gaussian_by_cf_inversion():
    law = module.QuadraticLaw(2., [0., 0.], [1., 2.])
    x = np.linspace(-4, 8, 35)
    pdf, sf, info = law.evaluate(x)
    np.testing.assert_allclose(sf, norm.sf(x, 2, np.sqrt(5)), atol=3e-7)
    np.testing.assert_allclose(pdf, norm.pdf(x, 2, np.sqrt(5)), atol=3e-7)


def test_noncentral_chi_square_special_case():
    # sum(z_i+.5)^2, eight independent components: df=8, noncentrality=2.
    law = module.QuadraticLaw(2., np.ones(8), np.ones(8))
    x = np.linspace(.3, 32, 50)
    pdf, sf, info = law.evaluate(x)
    np.testing.assert_allclose(sf, ncx2.sf(x, 8, 2), atol=3e-7)
    np.testing.assert_allclose(pdf, ncx2.pdf(x, 8, 2), atol=3e-7)


def test_hypothesis_moments_match_unchanged_core():
    null = GaussianHypothesis(np.array([4., 5., 6.]), np.diag([2., 4., 5.]))
    alt = GaussianHypothesis(np.array([5., 3., 7.]), np.diag([3., 2., 6.]))
    for h, generated in enumerate((null, alt)):
        law = module.from_hypotheses([(null, alt)], h)
        mean, variance = _quadratic_difference_moments(generated, null, alt)
        np.testing.assert_allclose([law.mean, law.sigma**2], [mean, variance], rtol=1e-12)


def test_indefinite_symmetric_distribution():
    # Difference of independent chi2_8: symmetry is an exact identity.
    law = module.QuadraticLaw(0., np.r_[np.ones(8), -np.ones(8)], np.zeros(16))
    x = np.linspace(-20, 20, 51)
    pdf, sf, _ = law.evaluate(x)
    np.testing.assert_allclose(sf+sf[::-1], 1., atol=3e-7)
    np.testing.assert_allclose(pdf, pdf[::-1], atol=3e-7)
    assert np.all(np.diff(sf) <= 1e-9)


def test_candidate_shared_renderer(tmp_path):
    import pandas as pd
    x = np.linspace(-4, 4, 100)
    panel = dict(label="CF test", profiled_T=x[20:80], fixed_T=x[20:80], mean=0., sigma=1., observed_T=1.,
        candidate=pd.DataFrame(dict(T=x, pdf=norm.pdf(x), sf=norm.sf(x))))
    path = tmp_path/"candidate.png"
    plot_statistic_calibration([panel], path, title="Candidate overlay")
    assert path.stat().st_size > 0
