"""Extract the four declared historical BNB flux vectors into visible CSV.

The historical module is parsed as syntax and is never imported or executed.
This is a one-time provenance conversion, not a compatibility layer.
"""

from __future__ import annotations

import argparse
import ast
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd


HISTORICAL_NAMES = {
    "numu_flux": "numu_flux",
    "numub_flux": "numubar_flux",
    "nue_flux": "nue_flux",
    "nueb_flux": "nuebar_flux",
}


def _safe_numeric_literal(node: ast.AST) -> object:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, (ast.List, ast.Tuple)):
        values: list[object] = []
        for element in node.elts:
            if isinstance(element, ast.Starred):
                expanded = _safe_numeric_literal(element.value)
                if not isinstance(expanded, list):
                    raise ValueError("starred flux literal must expand to a list")
                values.extend(expanded)
            else:
                values.append(_safe_numeric_literal(element))
        return values
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
        left = _safe_numeric_literal(node.left)
        right = _safe_numeric_literal(node.right)
        if isinstance(left, list) and isinstance(right, int):
            return left * right
        if isinstance(right, list) and isinstance(left, int):
            return right * left
    raise ValueError(f"unsupported historical flux syntax: {ast.dump(node, include_attributes=False)}")


def _read_flux_literals(path: Path) -> dict[str, np.ndarray]:
    module = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    load_function = next(
        node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "load_experiment"
    )
    extracted: dict[str, np.ndarray] = {}
    for statement in load_function.body:
        if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
            continue
        target = statement.targets[0]
        if not isinstance(target, ast.Name) or target.id not in HISTORICAL_NAMES or target.id in extracted:
            continue
        value = statement.value
        if not (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Attribute)
            and isinstance(value.func.value, ast.Name)
            and value.func.value.id == "np"
            and value.func.attr == "array"
            and value.args
        ):
            raise ValueError(f"historical {target.id} is not an explicit np.array literal")
        extracted[target.id] = np.asarray(_safe_numeric_literal(value.args[0]), dtype=float)[:60]
    if set(extracted) != set(HISTORICAL_NAMES):
        raise ValueError(f"could not find all historical flux vectors: {sorted(extracted)}")
    return extracted


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("frozen/baseline_v1/testchonggou1.py"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/inputs/bnb_flux.csv"),
    )
    arguments = parser.parse_args()
    extracted = _read_flux_literals(arguments.source)
    table: dict[str, object] = {
        "true_bin": np.arange(60, dtype=int),
        "true_energy_GeV": np.arange(60, dtype=float) * 0.05 + 0.025,
    }
    for historical_name, visible_name in HISTORICAL_NAMES.items():
        table[visible_name] = extracted[historical_name]
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(table).to_csv(arguments.output, index=False, float_format="%.17g")
    metadata = {
        "format": "bnb_flux_true_energy_60x0.05_GeV",
        "source": str(arguments.source.as_posix()),
        "source_sha256": sha256(arguments.source.read_bytes()).hexdigest(),
        "units": "historical source units; common overall normalization cancels in anchor reweight ratios",
        "warning": "shape provenance is preserved; absolute flux normalization is not independently asserted",
    }
    arguments.output.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(arguments.output)


if __name__ == "__main__":
    main()
