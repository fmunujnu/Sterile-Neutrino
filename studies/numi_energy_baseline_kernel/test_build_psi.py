from pathlib import Path
import importlib.util

import numpy as np
import pandas as pd


MODULE_PATH = Path(__file__).with_name("build_psi.py")
SPEC = importlib.util.spec_from_file_location("numi_energy_baseline_kernel", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_conditional_rows_normalize_and_empty_energy_uses_nearest() -> None:
    events = pd.DataFrame(
        {
            "neutrino_pdg": [14, 14],
            "parent_pdg": [211, 211],
            "energy_GeV": [0.05, 0.25],
            "baseline_km": [0.15, 0.25],
            "geometry_importance_weight": [1.0, 2.0],
        }
    )
    probability, diagnostics = MODULE.conditional_baseline_histogram(
        events,
        14,
        energy_edges=np.array([0.0, 0.1, 0.2, 0.3]),
        baseline_edges=np.array([0.1, 0.2, 0.3]),
    )
    assert np.allclose(probability.sum(axis=1), 1.0)
    assert diagnostics["used_nearest_energy_fallback"].tolist() == [False, True, False]
    assert np.allclose(probability[1], probability[0])


def test_psi_integrates_to_input_flux() -> None:
    flux = pd.DataFrame(
        {
            "energy_low_GeV": MODULE.ENERGY_EDGES_GEV[:-1],
            "energy_high_GeV": MODULE.ENERGY_EDGES_GEV[1:],
            "flux_per_POT_per_cm2_per_100MeV": np.linspace(1.0, 2.0, 50),
        }
    )
    conditional = np.full((50, 70), 1.0 / 70.0)
    diagnostics = pd.DataFrame(
        {
            "weighted_effective_entries": np.ones(50),
            "used_nearest_energy_fallback": np.zeros(50, dtype=bool),
        }
    )
    psi = MODULE.make_psi_table(
        flux,
        conditional,
        diagnostics,
        horn="rhc",
        flavour="numu",
        shape_flavour="numu",
        shape_method="test",
    )
    closure = MODULE.closure_table(psi)
    assert np.allclose(closure["integrated_phi"], closure["target_phi"])

