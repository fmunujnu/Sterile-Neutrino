from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--points", type=Path, required=True)
    parser.add_argument("--contours", type=Path, required=True)
    parser.add_argument("--figure", choices=("fig3a", "fig3b"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    points = pd.read_csv(args.points)
    points = points[points.figure == args.figure]
    contours = pd.read_csv(args.contours)
    contours = contours[contours.figure == args.figure].sort_values("mass").dropna(
        subset=["lower_amplitude", "upper_amplitude"]
    )
    xlabel = r"$\sin^2(2\theta_{\mu e})$" if args.figure == "fig3a" else r"$\sin^2(2\theta_{ee})$"

    fig, ax = plt.subplots(figsize=(7.3, 5.5), constrained_layout=True)
    image = ax.scatter(
        points.amplitude, points.mass, c=points.cls_corrected,
        cmap="turbo", norm=Normalize(.01, .07), marker="s", s=58,
        edgecolors="none", label="Evaluated parameter points",
    )
    ax.plot(contours.lower_amplitude, contours.mass, color="white", lw=2.4,
            label="95% pointwise limits")
    ax.plot(contours.upper_amplitude, contours.mass, color="white", lw=2.4)
    ax.plot(contours.lower_amplitude, contours.mass, color="black", lw=.7)
    ax.plot(contours.upper_amplitude, contours.mass, color="black", lw=.7)
    ax.set(xscale="log", yscale="log", xlabel=xlabel,
           ylabel=r"$\Delta m^2_{41}\;[\mathrm{eV}^2]$")
    ax.set_title(f"MicroBooNE {args.figure}: 200-Toy profiled $CL_s$ samples")
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label(r"$CL_s$ (colors clipped to 0.01--0.07)")
    ax.legend(fontsize=8); ax.grid(alpha=.12)
    fig.savefig(args.output, dpi=180)


if __name__ == "__main__":
    main()
