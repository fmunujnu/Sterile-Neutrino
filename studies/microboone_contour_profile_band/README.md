# Low-Toy profiled contour band

This diagnostic consumes tail counts produced by individually profiling every
Toy in `microboone_profile_toy_block_shift`. It estimates repeat-run Monte Carlo
uncertainty by binomially resampling those observed tail rates. It does not use
the fixed-hypothesis quadratic Toy shortcut and does not alter active inference.

The two bounding curves are pointwise Monte Carlo uncertainty limits, not an
experimental confidence region or a simultaneous-coverage guarantee.

