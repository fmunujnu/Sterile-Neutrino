from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from studies.microboone_fig3a_corner_high_toy.render_convergence import crossing


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--current-summary", type=Path, required=True)
    parser.add_argument("--current-contour", type=Path, required=True)
    parser.add_argument("--formal-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    heatmap = pd.read_csv(args.current_summary)
    current = pd.read_csv(args.current_contour)
    current = current[current.toy_count == 1000].sort_values("amplitude")
    formal = pd.read_csv(args.formal_result)
    formal = formal[
        formal.fixed_sin2_2theta_mue.between(.7, 1.0)
        & formal.fixed_delta_m2_41_eV2.between(.01, .05)
    ]
    formal_rows = []
    for amplitude, group in formal.groupby("fixed_sin2_2theta_mue"):
        formal_rows.append({
            "amplitude": amplitude,
            "mass": crossing(
                group.fixed_delta_m2_41_eV2.to_numpy(float),
                group.cls_quadratic.to_numpy(float),
                reference=.022,
            ),
        })
    formal_line = pd.DataFrame(formal_rows).sort_values("amplitude")
    formal_line.to_csv(args.output.with_name("formal_quadratic_local_contour.csv"), index=False)

    fig, ax = plt.subplots(figsize=(7.2, 5.4), constrained_layout=True)
    image = ax.scatter(
        heatmap.amplitude, heatmap.mass, c=heatmap.cls_corrected,
        cmap="turbo", norm=Normalize(0, .1), marker="s", s=180,
        edgecolors="none", label="1000-Toy evaluated points",
    )
    ax.plot(current.amplitude, current.bootstrap_median_mass,
            color="white", linewidth=4.0)
    ax.plot(current.amplitude, current.bootstrap_median_mass,
            color="tab:blue", linewidth=2.5,
            label="1000-Toy, every Toy profiled")
    ax.plot(formal_line.amplitude, formal_line.mass,
            color="white", linewidth=4.0)
    ax.plot(formal_line.amplitude, formal_line.mass,
            color="tab:red", linewidth=2.5, linestyle="--",
            label="Formal profile + quadratic-calibration contour")
    ax.set(xlim=(.7, 1.0), ylim=(.01, .05),
           xlabel=r"$\sin^2(2\theta_{\mu e})$",
           ylabel=r"$\Delta m^2_{41}\;[\mathrm{eV}^2]$")
    ax.set_title("Fig. 3a corner: production profile vs per-Toy profile")
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label(r"1000 per-Toy-profile $CL_s$")
    ax.legend(fontsize=8); ax.grid(alpha=.15)
    fig.savefig(args.output, dpi=180)


if __name__ == "__main__":
    main()
