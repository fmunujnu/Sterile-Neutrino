from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def crossing(
    mass: np.ndarray, cls: np.ndarray, target: float = .05, reference: float | None = None
) -> float:
    order = np.argsort(mass)
    mass, q = mass[order], cls[order] - target
    hits = np.flatnonzero(q[:-1] * q[1:] <= 0)
    if not len(hits):
        return np.nan
    candidates = []
    for index in hits:
        fraction = -q[index] / (q[index + 1] - q[index]) if q[index + 1] != q[index] else .5
        candidates.append(float(10 ** (
            np.log10(mass[index])
            + fraction * (np.log10(mass[index + 1]) - np.log10(mass[index]))
        )))
    if reference is not None and np.isfinite(reference):
        return min(candidates, key=lambda value: abs(np.log(value / reference)))
    return min(candidates, key=lambda value: abs(np.log(value / np.median(mass))))


def build(toys: pd.DataFrame, toy_count: int, draws: int, rng: np.random.Generator) -> pd.DataFrame:
    subset = toys[toys.toy_index < toy_count]
    counts = subset.groupby(["amplitude", "mass", "generating_hypothesis"]).is_right_tail.sum().unstack()
    rows = []
    for amplitude, group in counts.reset_index().groupby("amplitude"):
        group = group.sort_values("mass")
        mass = group.mass.to_numpy(float)
        k3, k4 = group["3nu"].to_numpy(int), group["4nu"].to_numpy(int)
        cls = np.minimum(1.0, (k4 + 1) / np.maximum(k3 + 1, 1))
        central = crossing(mass, cls)
        rate3, rate4 = (k3 + .5) / (toy_count + 1), (k4 + .5) / (toy_count + 1)
        roots = []
        for _ in range(draws):
            b3 = rng.binomial(toy_count, rate3)
            b4 = rng.binomial(toy_count, rate4)
            root = crossing(
                mass,
                np.minimum(1.0, (b4 + 1) / np.maximum(b3 + 1, 1)),
                reference=central,
            )
            if np.isfinite(root):
                roots.append(root)
        lower, median, upper = np.quantile(roots, [.025, .5, .975])
        rows.append({"toy_count": toy_count, "amplitude": amplitude,
                     "central_crossing_mass": central,
                     "bootstrap_lower_mass": lower,
                     "bootstrap_median_mass": median,
                     "bootstrap_upper_mass": upper,
                     "valid_bootstrap_fraction": len(roots) / draws})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--toy-values", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=10000)
    args = parser.parse_args()
    args.output_directory.mkdir(parents=True, exist_ok=True)
    toys = pd.read_csv(args.toy_values)
    rng = np.random.default_rng(20260907)
    results = pd.concat([
        build(toys, 500, args.bootstrap_draws, rng),
        build(toys, 750, args.bootstrap_draws, rng),
        build(toys, 1000, args.bootstrap_draws, rng),
    ], ignore_index=True)
    results.to_csv(args.output_directory / "nested_500_750_1000_contours.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.0, 5.2), constrained_layout=True)
    styles = {500: ("tab:orange", ":"), 750: ("tab:green", "--"), 1000: ("tab:blue", "-")}
    for count, group in results.groupby("toy_count"):
        group = group.sort_values("amplitude")
        color, style = styles[count]
        ax.plot(group.amplitude, group.bootstrap_lower_mass, color=color, ls=style, lw=1.6,
                label=f"{count} Toys: 95% limits")
        ax.plot(group.amplitude, group.bootstrap_upper_mass, color=color, ls=style, lw=1.6)
        ax.plot(group.amplitude, group.bootstrap_median_mass, color=color, ls=style, lw=2.3,
                label=f"{count} Toys: median contour")
    ax.set(xlim=(.7, 1.0), ylim=(.01, .05),
           xlabel=r"$\sin^2(2\theta_{\mu e})$",
           ylabel=r"$\Delta m^2_{41}\;[\mathrm{eV}^2]$")
    ax.set_title("Fig. 3a corner: nested profiled-Toy convergence")
    ax.legend(fontsize=8); ax.grid(alpha=.2)
    fig.savefig(args.output_directory / "nested_500_750_1000_contours.png", dpi=180)
    plt.close(fig)

    wide = results.pivot(index="amplitude", columns="toy_count")
    comparison = pd.DataFrame({
        "amplitude": wide.index,
        "median_shift_1000_minus_500": wide.bootstrap_median_mass[1000] - wide.bootstrap_median_mass[500],
        "full_width_500": wide.bootstrap_upper_mass[500] - wide.bootstrap_lower_mass[500],
        "full_width_1000": wide.bootstrap_upper_mass[1000] - wide.bootstrap_lower_mass[1000],
    }).reset_index(drop=True)
    comparison["width_ratio_1000_over_500"] = comparison.full_width_1000 / comparison.full_width_500
    comparison.to_csv(args.output_directory / "convergence_metrics.csv", index=False)
    print(results.to_string(index=False))
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
