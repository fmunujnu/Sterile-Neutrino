from pathlib import Path

import pytest

from sterile_fit.analysis.combination import ChiSquareContribution, CombinedChiSquare
from sterile_fit.analysis.selection import load_analysis_selection
from sterile_fit.paths import REPOSITORY_ROOT


def test_default_selection_includes_only_microboone_bnb() -> None:
    selection = load_analysis_selection(
        REPOSITORY_ROOT / "configs" / "analyses" / "microboone_bnb.yaml",
        repository_root=REPOSITORY_ROOT,
    )
    assert [item.experiment_id for item in selection.included] == [
        "microboone.bnb.four_channel"
    ]


def test_selection_rejects_unavailable_included_experiment(tmp_path: Path) -> None:
    configuration = tmp_path / "experiment.yaml"
    configuration.write_text("available: false\n", encoding="utf-8")
    selection_path = tmp_path / "selection.yaml"
    selection_path.write_text(
        "\n".join([
            'analysis_name: "invalid"',
            "experiments:",
            '  - experiment_id: "microboone.numi"',
            "    include: true",
            '    status: "inputs_unavailable"',
            '    correlation_group: "microboone_2025_release"',
            f'    configuration: "{configuration.name}"',
        ]) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="inputs are unavailable"):
        load_analysis_selection(selection_path, repository_root=tmp_path)


def test_combined_chi_square_exposes_breakdown_and_sum() -> None:
    objective = CombinedChiSquare((
        ChiSquareContribution("experiment.a", "independent.a", lambda value: value**2),
        ChiSquareContribution("experiment.b", "independent.b", lambda value: 2.0 * value),
    ))
    assert objective.breakdown(3.0) == {"experiment.a": 9.0, "experiment.b": 6.0}
    assert objective.chi2(3.0) == 15.0


def test_combination_rejects_separately_summed_correlated_inputs() -> None:
    with pytest.raises(ValueError, match="correlated datasets"):
        CombinedChiSquare((
            ChiSquareContribution("microboone.bnb", "microboone", lambda value: value),
            ChiSquareContribution("microboone.numi", "microboone", lambda value: value),
        ))
