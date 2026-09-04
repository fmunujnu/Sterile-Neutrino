"""Study report conventions, distinct from physical inference tests."""
import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm

spec=importlib.util.spec_from_file_location("quad_assess",Path(__file__).resolve().parents[1]/"studies/quadratic_toy_check/assess.py")
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_no_extrapolation_and_multiple_crossings():
    assert module.crossings([0,1,2],[3,2,1]) == []
    assert len(module.crossings([0,1,2,3],[1,-1,1,-1])) == 2
    assert module.crossings([0,1],[1,-1]) == [(0.5,0.,1.)]


def test_zero_tail_interval_positive_upper_limit():
    lo,hi=module.cp_interval(0,1000)
    assert lo==0 and 0<hi<.01


def test_identical_candidate_has_zero_ks_improvement():
    values=np.linspace(-2,2,100)
    curve=pd.DataFrame(dict(T_values=values,sf=norm.sf(values)))
    result=module.distribution_metrics(values,curve,0.,1.,.5,np.random.default_rng(4),20)
    assert abs(result['ks_improvement'])<1e-14
    assert abs(result['ks_improvement_bootstrap95_low'])<1e-14
    assert abs(result['ks_improvement_bootstrap95_high'])<1e-14


def test_unresolved_tail_is_not_a_precise_probability():
    values=np.linspace(-2,2,100)
    curve=pd.DataFrame(dict(T_values=values,sf=norm.sf(values)))
    result=module.distribution_metrics(values,curve,0.,1.,2.,np.random.default_rng(4),10,numerical_resolution=.1)
    assert np.isnan(result['p_quadratic'])
    assert result['p_quadratic_numerical_upper'] > result['p_quadratic_raw_integral']


def test_ee_coordinate_audit_records_both_branches():
    data=pd.DataFrame(dict(toy_sin2_theta14=[.1,.9],toy_sin2_theta24=[.2,.7]))
    row=module.profile_coordinate_audit(data,.36,"electron-disappearance-profile",.9)
    assert row['opposite_generation_s14_half_count']==1
    assert row['low_s14_branch_count']==row['high_s14_branch_count']==1


def test_coordinate_audit_rejects_wrong_amplitude():
    import pytest
    data=pd.DataFrame(dict(toy_sin2_theta14=[.1],toy_sin2_theta24=[.2]))
    with pytest.raises(AssertionError):
        module.profile_coordinate_audit(data,.36,"appearance-profile",.1)
