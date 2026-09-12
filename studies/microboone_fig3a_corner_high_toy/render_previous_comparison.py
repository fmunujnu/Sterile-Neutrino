from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--current-summary", type=Path, required=True)
    parser.add_argument("--current-contour", type=Path, required=True)
    parser.add_argument("--previous-contour", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    heatmap = pd.read_csv(args.current_summary)
    current = pd.read_csv(args.current_contour)
    current = current[current.toy_count == 1000].sort_values("amplitude")
    previous = pd.read_csv(args.previous_contour)
    previous = previous[previous.figure == "fig3a"].sort_values("mass")
    # Keep neighboring x-outside vertices so Matplotlib can clip the segment
    # where the previous contour actually enters this viewport.
    previous = previous[previous.mass.between(.01, .05)]

    fig, ax = plt.subplots(figsize=(7.2, 5.4), constrained_layout=True)
    image = ax.scatter(
        heatmap.amplitude, heatmap.mass, c=heatmap.cls_corrected,
        cmap="turbo", norm=Normalize(.0, .1), marker="s", s=180,
        edgecolors="none", label="1000-Toy evaluated points",
    )
    ax.plot(current.amplitude, current.bootstrap_median_mass,
            color="white", linewidth=4.0)
    ax.plot(current.amplitude, current.bootstrap_median_mass,
            color="tab:blue", linewidth=2.5,
            label="Current 1000-Toy profiled contour")
    ax.plot(previous.predicted_amplitude, previous.mass,
            color="white", linewidth=4.0)
    ax.plot(previous.predicted_amplitude, previous.mass,
            color="tab:red", linewidth=2.5, linestyle="--",
            label="Previous 200-Toy profiled contour")
    ax.set(xlim=(.7, 1.0), ylim=(.01, .05),
           xlabel=r"$\sin^2(2\theta_{\mu e})$",
           ylabel=r"$\Delta m^2_{41}\;[\mathrm{eV}^2]$")
    ax.set_title("Fig. 3a corner: profiled-Toy contour comparison")
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label(r"Current 1000-Toy $CL_s$")
    ax.legend(fontsize=8); ax.grid(alpha=.15)
    fig.savefig(args.output, dpi=180)


if __name__ == "__main__":
    main()
