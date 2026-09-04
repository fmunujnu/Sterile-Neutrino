# 1+3+1 parallel scan

This directory is deliberately parallel to the established 3+1 entry points.
It does not refactor or replace `scripts/scan.py`.

## Reused without modification

- BNB and NuMI visible event-response kernels;
- published observations and fixed published background convention;
- prediction-scaled systematic covariance and Pearson statistical diagonal;
- the joint 208-bin BNB-NuMI covariance, including cross blocks;
- analytic and Toy-MC `CLs` utilities.

## New physics coordinates

The isolated lower state is represented by the positive magnitude
`delta_m2_41_absolute_eV2`, so its signed splitting is negative.  The upper
state uses positive `delta_m2_51_eV2`.  At each mass pair the current scanner
profiles

`abs_Ue4_squared`, `abs_Umu4_squared`, `abs_Ue5_squared`,
`abs_Umu5_squared`, and `cp_phase_mue_rad`.

The phase convention is
`cp_phase_mue_rad = arg(U_mu4* U_e4 U_mu5 U_e5*)`; the antineutrino amplitude
uses the complex-conjugate value.  Recording this sign convention is required
before comparing a fitted phase to another paper.

The implementation enforces both flavour-row normalisations and the condition
that the selected heavy-state row fragments can be embedded in a unitary 5x5
matrix.  It explicitly tests the state-4-decoupled, state-5-decoupled and 3nu
boundaries.  The parameterisation follows the standard two-sterile
short-baseline form used by Karagiorgi et al. (hep-ph/0609177) and Kopp et al.
(arXiv:1303.3011); it is not a newly invented effective angle.

`theta34` and `theta35` are fixed to zero only for this first e/mu CC stage.
`theta45` is not scanned because a sterile-sterile basis rotation is redundant
for the four e/mu probabilities consumed by the current kernels.  Tau and NC
data would require a later, explicit extension rather than silently profiling
these absent coordinates.

## Commands

Fast analytic calibration with the same default BNB selection as 3+1:

```powershell
python scripts/one_plus_three_plus_one/run1_non_toy.py
```

Joint BNB-NuMI diagnostic selection:

```powershell
python scripts/one_plus_three_plus_one/run1_non_toy.py --analysis-config configs/analyses/microboone_bnb_numi.yaml
```

Toy-MC calibration (100 toys per hypothesis and mass pair by default):

```powershell
python scripts/one_plus_three_plus_one/run2_toy_mc.py --toy-workers 1
```

This first development scanner is sequential across mass pairs.  Do not call it
a production Toy scan or increase nested Toy threads blindly; point-level
process parallelism should be added only after numerical equivalence tests.

Use explicit comma-separated mass grids when needed:

```powershell
python scripts/one_plus_three_plus_one/run1_non_toy.py `
  --delta-m2-41-absolute-grid-eV2 0.01,0.1,1,10,100 `
  --delta-m2-51-grid-eV2 0.01,0.1,1,10,100
```

## Interpretation boundary

The default two-dimensional mass plane profiles every mixing to zero if that is
preferred.  Since 3nu is then nested inside 1+3+1, this plot is a diagnostic
preference map and cannot alone answer whether the complete 1+3+1 parameter
space is excluded.  The next analysis stage must define nonzero physical
mixing slices (or add signal experiments) before presenting exclusion claims.
