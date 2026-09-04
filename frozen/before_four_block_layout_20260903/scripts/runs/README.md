# Stable run entry points

- `run1_non_toy.py`: fast analytic moment-Gaussian approximation to pointwise
  CLs. It does not generate pseudo-experiments and must not be labelled Toy MC.
- `run2_toy_mc.py`: empirical pointwise CLs under both 3nu and 4nu hypotheses,
  with the same physical profile repeated for each pseudo-experiment. The
  default is 100 toys per hypothesis and is diagnostic rather than final-tail
  precision.

Both files only set the statistical calibration and default output directory.
All prediction, covariance, profiling, CLs and plotting logic remains in the
single canonical `scripts/scan.py` engine.
