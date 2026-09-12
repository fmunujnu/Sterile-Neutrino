# NuMI response-matrix inverse study

This isolated study asks whether a small, smooth perturbation of the borrowed
BNB response can reduce the difference between the local NuMI-only surface and
the released NuMI-only grid.  It never modifies the BNB response, the active
NuMI kernel, or any production analysis entry point.

The two reported planes use the official plotting ranges and 100 x 100 points:

- Fig. 3a: `sin^2(2 theta_mue) = 1e-4 .. 1`, `dm2 = 1e-2 .. 1e2 eV2`;
- Fig. 3b: `sin^2(2 theta_ee) = 1e-2 .. 1`, `dm2 = 1e-1 .. 14 eV2`.

## Meaning of "fixed kernel"

The repository's stored event kernel already contains the response matrix.
For this study it is factorised as

`event_kernel[r,t,source] = remaining_weight[r,t,source] * R0[r,t]`.

`remaining_weight` is held fixed and only `R0` is perturbed.  This is an
algebraic factorisation of the existing empirical kernel, not a separately
measured flux-times-cross-section-times-efficiency array.

## Constraints

For every channel the perturbation basis is projected into the exact nullspace
of both:

1. every non-empty true-energy column sum;
2. every reconstructed-bin zero-oscillation signal count.

Structural zeros remain zero.  Coefficient bounds guarantee non-negativity.
Smooth low-frequency basis functions and an L2 penalty select a small,
energy-smooth correction from the non-unique solution family.

The official scalar delta-chi-square surface does not uniquely identify a
response matrix.  Therefore outputs are an **equivalent response hypothesis**,
not a recovered collaboration detector response.  Training and held-out errors
must both be reported before it may be used even diagnostically in 1+3+1.

Run a cheap construction/constraint check:

```text
python studies/numi_reco_inverse/run.py --mode check
```

The full fit is intentionally explicit:

```text
python studies/numi_reco_inverse/run.py --mode fit --grid-points 100
```

