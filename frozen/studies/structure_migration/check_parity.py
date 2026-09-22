"""Low-resource, visible before/after fixtures; never execute frozen code."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

import numpy as np
from sterile_fit.adapter import build_three_plus_one_analysis
from sterile_fit.adapter import load_analysis_selection
from sterile_fit.core.three_plus_one import ThreePlusOneParameters
from sterile_fit.core.three_plus_one import ThreePlusOneVacuumModel
from sterile_fit.core.profile_three_plus_one import profile_s14_s24_at_fixed_sin2_2theta_mue, profile_s14_s24_at_fixed_sin2_2theta_ee
from sterile_fit.core.one_plus_three_plus_one import OnePlusThreePlusOneParameters, OnePlusThreePlusOneVacuumModel
from sterile_fit.adapter import build_one_plus_three_plus_one_analysis
from sterile_fit.core.calibration import asymptotic_cls, toy_cls, prepare_fixed_hypothesis_chi2
from sterile_fit.adapter import _hypothesis_pairs, _objective_for_toy


def fixtures():
    result = {"inputs_sha256": {}, "calculations": {}}
    for path in sorted((ROOT / "data").rglob("*")):
        if path.is_file() and path.suffix in {".csv", ".json", ".yaml"}:
            result["inputs_sha256"][path.relative_to(ROOT).as_posix()] = sha256(path.read_bytes()).hexdigest()
    p3 = [ThreePlusOneParameters(1.2, 0, 0), ThreePlusOneParameters(1.2, 0.04, 0.018), ThreePlusOneParameters(0.63, 0.8, 0.25)]
    energy = np.array([0.025, 0.1, 0.55, 1.2, 2.975])
    for i, p in enumerate(p3):
        result["calculations"][f"probability3_{i}"] = [ThreePlusOneVacuumModel(p).probability(a,b,energy,0.4685,antineutrino=anti).tolist() for anti in (False,True) for a in (0,1) for b in (0,1)]
    p5 = OnePlusThreePlusOneParameters(0.7, 1.3, 0.02, 0.01, 0.015, 0.025, 0.6)
    result["calculations"]["probability5"] = [OnePlusThreePlusOneVacuumModel(p5).probability(a,b,energy,0.680,antineutrino=anti).tolist() for anti in (False,True) for a in (0,1) for b in (0,1)]
    for name in ("microboone_bnb", "microboone_bnb_numi"):
        selection = load_analysis_selection(ROOT / "configs" / "analyses" / f"{name}.yaml", repository_root=ROOT)
        analysis = build_three_plus_one_analysis(selection, repository_root=ROOT)
        for i,p in enumerate(p3):
            exp = analysis.experiments[0]
            prediction = exp.predict_counts(p)
            result["calculations"][f"{name}_{i}"] = {"prediction":prediction.tolist(), "covariance":exp.covariance_for_prediction(prediction).tolist(), "chi2":analysis.objective.chi2(p)}
        extended = build_one_plus_three_plus_one_analysis(selection, repository_root=ROOT)
        result["calculations"][name+"_extended"] = {"prediction":extended.experiments[0].predict_counts(p5).tolist(), "chi2":extended.objective.chi2(p5)}
        for mode, profile_fn, axis in (("mue",profile_s14_s24_at_fixed_sin2_2theta_mue,"sin2_2theta_mue"),("ee",profile_s14_s24_at_fixed_sin2_2theta_ee,"sin2_2theta_ee")):
            profile = profile_fn(analysis.objective.chi2, delta_m2_41_eV2=1.2, **{axis:0.003 if mode=="mue" else 0.1})
            tested = profile.best_fit.parameters
            pairs = _hypothesis_pairs(analysis,p3[0],tested)
            statistic = profile.best_fit.chi2-analysis.objective.chi2(p3[0])
            result["calculations"][name+mode] = {"parameters":asdict(tested),"chi2":profile.best_fit.chi2,"analytic":asdict(asymptotic_cls(statistic,pairs))}
            nulls = tuple(p[0] for p in pairs)
            tests = tuple(p[1] for p in pairs)
            null_objective = prepare_fixed_hypothesis_chi2(nulls)
            def evaluate(dataset):
                obj=_objective_for_toy(analysis,dataset)
                return profile_fn(obj,delta_m2_41_eV2=1.2,**{axis:0.003 if mode=="mue" else 0.1}).best_fit.chi2-null_objective(dataset)
            toy=toy_cls(statistic,nulls,tests,evaluate,number_of_toys=8,seed=20250821,workers=1,batch_size=4)
            result["calculations"][name+mode+"_toy"]=asdict(toy)
        print("Captured",name,flush=True)
    return result


def serializable(value):
    if isinstance(value,np.ndarray): return value.tolist()
    if isinstance(value,np.generic): return value.item()
    raise TypeError(type(value).__name__)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--record",action="store_true")
    parser.add_argument("--baseline", type=Path, required=True, help="Explicit immutable before.json")
    parser.add_argument("--output-directory", type=Path, required=True)
    args=parser.parse_args()
    destination=args.output_directory
    baseline=args.baseline
    actual=json.loads(json.dumps(fixtures(),default=serializable,allow_nan=False))
    destination.mkdir(parents=True,exist_ok=True)
    if args.record:
        if baseline.exists(): raise RuntimeError("Refusing to replace baseline")
        baseline.write_text(json.dumps(actual,ensure_ascii=False,indent=2),encoding="utf-8")
        print("Saved immutable before fixture")
    else:
        expected=json.loads(baseline.read_text(encoding="utf-8"))
        if actual!=expected:
            (destination/"after_mismatch.json").write_text(json.dumps(actual,default=serializable,indent=2),encoding="utf-8")
            raise AssertionError("Before/after numerical fixtures differ")
        (destination/"parity.json").write_text(json.dumps({"exact_match":True,"input_files":len(actual["inputs_sha256"]),"calculation_groups":len(actual["calculations"]),"toys_per_hypothesis":8,"workers":1},indent=2),encoding="utf-8")
        print("PASS exact input hashes, predictions, covariances, profiles, analytic CLs and profiled Toy statistics")


if __name__=="__main__": main()
