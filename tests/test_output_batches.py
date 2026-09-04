"""Output paths are bookkeeping, not scientific settings."""
import pytest
from sterile_fit import output


def test_batch_groups_products_without_creating_directories(tmp_path, monkeypatch):
    monkeypatch.setattr(output, "REPOSITORY_ROOT", tmp_path)
    output.begin_output_batch("run_test")
    a = output.result_directory("microboone_bnb_numi_joint_diagnostic", "three_plus_one", "scan_appearance-profile_analytic")
    b = output.result_directory("microboone_bnb_numi_joint", "three_plus_one", "spectra_figure1")
    assert a.parent == b.parent
    assert a.name == "scan_fig3a_analytic"
    assert not a.exists()
    output.begin_output_batch("run_next")
    assert output.result_directory("microboone_bnb", "three_plus_one", "spectra").parent.name == "run_next"


@pytest.mark.parametrize("name", ["../elsewhere", "a/b", "a\\b", "..", "C:", "CON", "foo."])
def test_batch_rejects_unsafe_names(name):
    with pytest.raises(ValueError):
        output.begin_output_batch(name)


def test_empty_invocation_does_not_write_provenance(tmp_path, monkeypatch):
    monkeypatch.setattr(output, "REPOSITORY_ROOT", tmp_path)
    output.begin_output_batch("help_test")
    output.result_directory("source", "model", "product")
    output.finish_output_batch(["--help"])
    assert list(tmp_path.iterdir()) == []


def test_provenance_is_visible_and_records_each_invocation(tmp_path, monkeypatch):
    import json
    monkeypatch.setattr(output, "REPOSITORY_ROOT", tmp_path)
    (tmp_path / "run.py").write_text("# fixture", encoding="utf-8")
    output.begin_output_batch("same_batch")
    product = output.result_directory("source", "model", "spectra")
    product.mkdir(parents=True)
    output.finish_output_batch(["--kind", "bnb"])
    output.finish_output_batch(["--kind", "joint"])
    records = list((product.parent / "provenance").glob("*.json"))
    assert len(records) == 2
    for path in records:
        document = json.loads(path.read_text(encoding="utf-8"))
        assert document["products"] == ["spectra"]
        assert len(document["code_and_config_sha256"]["run.py"]) == 64
