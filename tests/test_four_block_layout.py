"""Structural equivalence and entrypoint contracts; no expensive scan here."""
import ast
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

from sterile_fit.paths import REPOSITORY_ROOT
from sterile_fit.output import SpectrumPanel, render_microboone_spectrum_panels, write_csv, write_json


def test_experiment_adapters_own_all_experiment_entrypoints():
    package = REPOSITORY_ROOT / "src/sterile_fit"
    assert not (package / "adapter.py").exists()
    assert not (package / "miniboone.py").exists()
    assert not (package / "lsnd.py").exists()
    assert (package / "experiments/interface.py").is_file()
    for experiment in ("microboone", "miniboone", "lsnd"):
        assert (package / f"experiments/{experiment}/adapter.py").is_file()


@pytest.mark.parametrize("old,new", [
    ("parameters.py", "core/three_plus_one.py"),
    ("models/three_plus_one.py", "core/three_plus_one.py"),
    ("one_plus_three_plus_one/parameters.py", "core/one_plus_three_plus_one.py"),
    ("one_plus_three_plus_one/model.py", "core/one_plus_three_plus_one.py"),
    ("fitting.py", "core/profile_three_plus_one.py"),
    ("one_plus_three_plus_one/fitting.py", "core/profile_one_plus_three_plus_one.py"),
    ("covariance.py", "core/likelihood.py"),
    ("likelihood.py", "core/likelihood.py"),
    ("statistics/asymptotic_cls.py", "core/calibration.py"),
        # Toy evaluation is intentionally optimized after the layout freeze;
        # exact scalar/batched equivalence is covered by test_toy_cls.py.
    ("experiments/microboone/joint_bnb_numi.py", "experiments/microboone/joint.py"),
    ("experiments/microboone/bnb/templates.py", "experiments/microboone/bnb.py"),
    ("experiments/microboone/numi/event_prediction.py", "experiments/microboone/numi.py"),
])
def test_numerical_function_bodies_are_unchanged(old, new):
    """Read archived source as evidence only; never import or execute it."""
    before = REPOSITORY_ROOT / "frozen/before_four_block_layout_20260903/src/sterile_fit" / old
    after = REPOSITORY_ROOT / "src/sterile_fit" / new
    def definitions(path):
        return {node.name: ast.dump(node, include_attributes=False)
                for node in ast.parse(path.read_text(encoding="utf-8")).body
                if isinstance(node, (ast.FunctionDef, ast.ClassDef))
                and not node.name.startswith("prefit_")}
    expected, actual = definitions(before), definitions(after)
    for name, body in expected.items():
        assert name in actual
        assert actual[name] == body, f"Numerical implementation changed: {old}:{name}"


def _entrypoint():
    spec = importlib.util.spec_from_file_location("profile_entrypoint", REPOSITORY_ROOT / "run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_entrypoint_forwards_calibration_and_preserves_overrides(monkeypatch):
    import sterile_fit.scan as scan
    captured = []
    monkeypatch.setattr(scan, "scan_three_plus_one", lambda: captured.extend(sys.argv[1:]))
    _entrypoint().main(["scan", "--preset", "fig3b", "--calibration", "toy", "--number-of-toys", "8", "--delta-m2-min-eV2", "0.2"])
    assert captured[captured.index("--cls-calibration") + 1] == "toy"
    assert captured[-2:] == ["--delta-m2-min-eV2", "0.2"]
    assert "electron-disappearance-profile" in captured


def test_fig3b_registered_mass_range_reaches_40_ev2(monkeypatch):
    import sterile_fit.scan as scan
    captured = []
    monkeypatch.setattr(scan, "scan_three_plus_one", lambda: captured.extend(sys.argv[1:]))
    _entrypoint().main(["scan", "--preset", "fig3b", "--calibration", "analytic"])
    position = captured.index("--delta-m2-max-eV2")
    assert captured[position + 1] == "40"


def test_entrypoint_rejects_removed_prefit(monkeypatch):
    with pytest.raises(SystemExit) as exc:
        _entrypoint().main(["scan", "--mode", "prefit"])
    assert exc.value.code == 2


def test_entrypoint_rejects_unimplemented_extended_adaptive():
    with pytest.raises(SystemExit) as exc:
        _entrypoint().main(["scan", "--model", "1+3+1", "--calibration", "adaptive-toy"])
    assert exc.value.code == 2


@pytest.mark.parametrize("layout", ["channels", "figure1"])
def test_spectrum_layouts_share_renderer(tmp_path, layout):
    panel = SpectrumPanel("test", np.array([0.0, 0.1, 0.2]), np.array([2., 3.]),
                          np.ones(2), np.ones(2), np.ones(2), np.array([2., 3.]))
    target = tmp_path / (layout + ".png")
    render_microboone_spectrum_panels([panel, panel], target, title="test", layout=layout)
    assert target.stat().st_size > 0


def test_visible_result_writers_roundtrip(tmp_path):
    values = np.array([0.12345678901234567, 1.2345678901234567])
    write_csv(pd.DataFrame({"value": values}), tmp_path / "table.csv")
    actual = pd.read_csv(tmp_path / "table.csv", float_precision="round_trip")["value"].to_numpy()
    np.testing.assert_array_equal(actual, values)
    write_json(tmp_path / "metadata.json", {"method": "profile", "note": "可见"})
    assert "可见" in (tmp_path / "metadata.json").read_text(encoding="utf-8")
