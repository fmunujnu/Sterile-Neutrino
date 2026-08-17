import numpy as np
from pathlib import Path

from sterile_fit.analysis.registry import build_three_plus_one_analysis
from sterile_fit.analysis.selection import load_analysis_selection
from sterile_fit.experiments.microboone.bnb.binning import bnb_four_channel_indices
from sterile_fit.experiments.microboone.joint_bnb_numi import (
    joint_bnb_numi_published_indices,
)
from sterile_fit.experiments.microboone.numi.binning import (
    numi_four_channel_published_indices,
)
from sterile_fit.experiments.microboone.bnb.published_inputs import read_full_systematic_covariance
from sterile_fit.experiments.microboone.bnb.workflow import build_strict_bnb_workflow
from sterile_fit.experiments.microboone.joint_bnb_numi import build_joint_microboone_bnb_numi_workflow
from sterile_fit.experiments.microboone.numi.workflow import build_diagnostic_numi_workflow
from sterile_fit.parameters import ThreePlusOneParameters
import yaml


def test_joint_order_is_bnb_then_numi_and_keeps_cross_covariance() -> None:
    indices = joint_bnb_numi_published_indices()
    assert indices == (*bnb_four_channel_indices(), *numi_four_channel_published_indices())
    full = read_full_systematic_covariance()
    selected = full[np.ix_(indices, indices)]
    assert selected.shape == (208, 208)
    assert np.allclose(selected, selected.T)
    assert np.count_nonzero(selected[:104, 104:]) > 0


def test_joint_analysis_registry_builds_one_cross_covariance_contribution() -> None:
    root = Path(__file__).resolve().parents[1]
    selection = load_analysis_selection(
        root / "configs" / "analyses" / "microboone_bnb_numi.yaml",
        repository_root=root,
    )
    analysis = build_three_plus_one_analysis(selection, repository_root=root)
    assert len(analysis.experiments) == 1
    assert analysis.experiments[0].metadata["covariance_includes_bnb_numi_cross_blocks"] is True
    reference = ThreePlusOneParameters(1.2, 0.0, 0.0)
    np.testing.assert_allclose(
        analysis.objective.chi2(reference), 152.21602635970766, rtol=1e-10
    )


def test_cached_joint_predictor_matches_original_beam_predictors_point_by_point() -> None:
    root = Path(__file__).resolve().parents[1]
    bnb_document = yaml.safe_load(
        (root / "configs/experiments/microboone/bnb/analysis.yaml").read_text(encoding="utf-8")
    )
    numi_document = yaml.safe_load(
        (root / "configs/experiments/microboone/numi/analysis.yaml").read_text(encoding="utf-8")
    )
    reference = ThreePlusOneParameters(**{
        name: float(value) for name, value in bnb_document["reference_parameters"].items()
    })
    bnb = build_strict_bnb_workflow(
        root / bnb_document["analysis_inputs"]["kernel"],
        root / bnb_document["analysis_inputs"]["covariance"],
        reference,
        float(bnb_document["baseline_km"]),
    )
    numi = build_diagnostic_numi_workflow(
        root / numi_document["diagnostic_four_channel_events"]["kernel_directory"],
        reference,
        float(numi_document["baseline_km"]),
    )
    joint = build_joint_microboone_bnb_numi_workflow(bnb, numi)
    points = (
        ThreePlusOneParameters(0.03, 0.001, 0.7),
        ThreePlusOneParameters(1.2, 0.043564535412361605, 0.018),
        ThreePlusOneParameters(17.0, 0.8, 0.3),
    )
    for point in points:
        original = np.concatenate(
            (bnb.predictor.predict_total_counts(point), numi.predictor.predict_total_counts(point))
        )
        np.testing.assert_allclose(joint.predict_total_counts(point), original, rtol=2e-13, atol=2e-10)
