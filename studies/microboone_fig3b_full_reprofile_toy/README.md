# MicroBooNE Fig.3b full-grid per-Toy reprofile

This research runner uses the active BNB+NuMI prediction, registered
`psi(E,L,flavour)` input, covariance and Fig.3b constrained-profile definition.
It does not replace the active fixed-hypothesis Toy implementation.

The grid is 74 logarithmic mass points from 0.1 to 40 eV2 and 61 logarithmic
`sin2(2theta_ee)` points from 0.01 to 1.  At every plotted point it generates
the requested number of Toys under each of the 3nu and tested 4nu hypotheses.
Every Toy repeats the same fixed-coordinate profile over `sin2(theta24)` and
both physical `sin2(theta14)` branches.  Summary and raw Toy statistics are
appended immediately, so completed points survive an interrupted worker.

One hundred Toys per hypothesis give only about five expected samples in a
0.05 tail and are therefore a qualitative cross-check, not a stable exclusion
calibration.
