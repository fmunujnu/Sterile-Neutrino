"""Replay saved Toys and verify the real scan dispatcher before reusing results."""
import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key]="1"
import argparse
import json
from pathlib import Path
from hashlib import sha256
import sys
import numpy as np
import pandas as pd
from scipy.linalg import cholesky

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/"src"),str(ROOT)]
from studies.quadratic_toy_check.run import _build_analysis
from sterile_fit.adapter import _hypothesis_pairs
from sterile_fit.core.three_plus_one import ThreePlusOneParameters
from sterile_fit.core.calibration import _draw_gaussian_toys,prepare_fixed_hypothesis_chi2
from sterile_fit.scan import _profile_toy_at_scan_point,_stable_point_seed
from sterile_fit.output import begin_output_batch,result_directory,write_csv,write_json,finish_output_batch


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",type=Path,required=True)
    parser.add_argument("--mode",choices=("appearance-profile","electron-disappearance-profile"),required=True)
    parser.add_argument("--batch",required=True)
    args=parser.parse_args()
    begin_output_batch(args.batch)
    out=result_directory("studies","quadratic_toy_check","reuse_verification")
    if out.exists(): raise FileExistsError("No overwrite")
    meta=json.loads((args.source/"metadata.json").read_text(encoding="utf-8"))
    if meta.get("scan_mode",args.mode)!=args.mode: raise ValueError("Wrong reuse mode")
    config=ROOT/"configs/analyses/microboone_bnb_numi.yaml"
    if meta["config_sha256"]!=sha256(config.read_bytes()).hexdigest(): raise ValueError("Config changed")
    for name,digest in meta["core_sha256"].items():
        if sha256((ROOT/name).read_bytes()).hexdigest()!=digest: raise ValueError(f"Core changed: {name}")
    science=[p for p in (ROOT/"data").rglob("*") if p.is_file() and p.suffix.lower() in {".csv",".json",".yaml",".txt"}]
    data_hashes={str(p.relative_to(ROOT)):sha256(p.read_bytes()).hexdigest() for p in science}
    samples=pd.read_csv(args.source/"samples.csv",float_precision="round_trip")
    scores=pd.read_csv(args.source/"scores.csv",float_precision="round_trip")
    analysis=_build_analysis(config)
    null=ThreePlusOneParameters(1.,0.,0.)
    checks=[]
    for index,group in samples.groupby("point_index"):
        score=scores[scores.point_index==index].iloc[0]
        if "generation_sin2_theta14" in score.index:
            tested=ThreePlusOneParameters(score.mass,score.generation_sin2_theta14,score.generation_sin2_theta24)
        else:
            tested=ThreePlusOneParameters(score.mass,score.profiled_sin2_theta14,score.derived_sin2_theta24)
        pairs=_hypothesis_pairs(analysis,null,tested)
        hs=[tuple(p[h] for p in pairs) for h in (0,1)]
        fixed=[prepare_fixed_hypothesis_chi2(h) for h in hs]
        seeds=np.random.SeedSequence(_stable_point_seed(meta["seed"],int(index))).spawn(2)
        for h,generator in enumerate(("3nu","4nu")):
            saved=group[group.generator==generator].sort_values("toy_index")
            draws=_draw_gaussian_toys(hs[h],tuple(cholesky(x.covariance,lower=True) for x in hs[h]),len(saved),
                tuple(np.random.default_rng(s) for s in seeds[h].spawn(len(hs[h]))))
            indices=np.unique(np.linspace(0,len(saved)-1,5,dtype=int))
            for j in indices:
                dataset=tuple(d[j] for d in draws)
                fit=_profile_toy_at_scan_point(analysis,dataset,mode=args.mode,tested_parameters=tested)
                chi3=fixed[0](dataset)
                expected=saved.iloc[j]
                np.testing.assert_allclose([chi3,fit.chi2,fixed[1](dataset)-chi3],
                    [expected.chi2_3nu,expected.chi2_4nu,expected.fixed_T],rtol=1e-10,atol=1e-8)
                checks.append(dict(point_index=index,generator=generator,toy_index=int(j),
                    chi2_profile_difference=fit.chi2-expected.chi2_4nu,chi2_null_difference=chi3-expected.chi2_3nu))
        print(f"Official scan dispatcher matched sampled records at point {index}",flush=True)
    for name,digest in data_hashes.items():
        if sha256((ROOT/name).read_bytes()).hexdigest()!=digest: raise ValueError(f"Data changed: {name}")
    write_csv(pd.DataFrame(checks),out/"dispatcher_checks.csv")
    write_json(out/"metadata.json",dict(mode=args.mode,source=str(args.source.resolve()),
        retained_samples=len(samples),reprofiled_checks=len(checks),all_rows_reprofiled=False,
        data_sha256=data_hashes,source_sha256={p.name:sha256(p.read_bytes()).hexdigest() for p in (args.source/"samples.csv",args.source/"scores.csv")},
        statement="No new random ensemble. Five stored draws per point/hypothesis replayed through official scan dispatcher; old full-ensemble results reused, not represented as a new full simulation."))
    finish_output_batch(sys.argv[1:])


if __name__=="__main__":main()
