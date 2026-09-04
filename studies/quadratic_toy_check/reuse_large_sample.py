"""Recompute fixed-quadratic curves for saved 10,000-Toy samples, no new Toys."""
import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
import argparse
from pathlib import Path
import sys
from hashlib import sha256
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/"src"),str(ROOT)]
from studies.quadratic_toy_check.quadratic import from_hypotheses
from studies.three_plus_one_toy_distribution_fit.run import _build_analysis
from sterile_fit.adapter import _hypothesis_pairs
from sterile_fit.core.three_plus_one import ThreePlusOneParameters
from sterile_fit.output import begin_output_batch, result_directory, write_csv, write_json, finish_output_batch


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--batch", required=True)
    args=parser.parse_args()
    begin_output_batch(args.batch)
    out=result_directory("studies","quadratic_toy_check","results")
    if out.exists(): raise FileExistsError("No overwrite")
    samples=pd.read_csv(args.source/"samples_with_gaussian_p.csv",float_precision="round_trip")
    old=pd.read_csv(args.source/"distribution_comparison.csv",float_precision="round_trip")
    analysis=_build_analysis(ROOT/"configs/analyses/microboone_bnb_numi.yaml")
    null=ThreePlusOneParameters(1.,0.,0.)
    scores=[]
    for index,group in samples.groupby("point_index"):
        first=group.iloc[0]
        tested=ThreePlusOneParameters(first.fixed_delta_m2_41_eV2,first.profiled_observed_sin2_theta14,first.derived_observed_sin2_theta24)
        np.testing.assert_allclose(analysis.objective.chi2(tested),first.observed_profiled_chi2_4nu,rtol=1e-10,atol=1e-8)
        observed=analysis.objective.chi2(tested)-analysis.objective.chi2(null)
        pairs=_hypothesis_pairs(analysis,null,tested)
        for h,generator in enumerate(("3nu","4nu")):
            law=from_hypotheses(pairs,h)
            values=group[group.generator==generator]
            ref=old[(old.point_index==index)&(old.generator==generator)&(old.variant=="reprofiled")].iloc[0]
            np.testing.assert_allclose([law.mean,law.sigma,observed],[ref.gaussian_mean,ref.gaussian_sigma,ref.observed_T],rtol=1e-9,atol=1e-8)
            low=min(values.test_statistic.min(),values.fixed_T_same_draw.min(),law.mean-5*law.sigma,observed)
            high=max(values.test_statistic.max(),values.fixed_T_same_draw.max(),law.mean+5*law.sigma,observed)
            grid=np.unique(np.r_[np.linspace(low,high,2001),observed])
            pdf,sf,diagnostics=law.evaluate(grid)
            # Independent midpoint check of linear interpolation used in saved-data assessment.
            mids=(grid[:-1:20]+grid[1::20])/2
            _,sf_mid,_=law.evaluate(mids)
            interp_error=float(np.max(np.abs(sf_mid-np.interp(mids,grid,sf))))
            if interp_error>1e-5: raise ArithmeticError("Curve interpolation too coarse")
            write_csv(pd.DataFrame(dict(T=grid,pdf=pdf,sf=sf)),out/f"point_{index:02d}_{generator}_curve.csv")
            scores.append(dict(point_index=index,generator=generator,variant="reprofiled",mean=law.mean,sigma=law.sigma,
                constant=law.constant,mass=first.fixed_delta_m2_41_eV2,amplitude=first.fixed_sin2_2theta_mue,
                region="reused_large_sample",observed_T=observed,interpolation_midpoint_error=interp_error,**diagnostics))
        print(f"Reused 10000-Toy point {index}: prediction and analytic moments verified",flush=True)
    write_csv(pd.DataFrame(scores),out/"scores.csv")
    write_csv(samples.rename(columns={"fixed_T_same_draw":"fixed_T"}),out/"samples.csv")
    write_json(out/"metadata.json",dict(new_toys=0,reused_toys=len(samples),
        validation="Observed chi2 and both analytic moments match current construction; historical fixed-T replay validation retained in source metadata",
        source=str(args.source.resolve()),source_hashes={p.name:sha256(p.read_bytes()).hexdigest() for p in args.source.glob("*.csv")},
        note="Uniform saved-curve interpolation midpoint check <=1e-5; not full uniform bound"))
    finish_output_batch(sys.argv[1:])


if __name__=="__main__": main()
