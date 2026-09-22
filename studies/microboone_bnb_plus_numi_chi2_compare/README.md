# BNB plus NuMI saved-surface comparison

This diagnostic adds the saved BNB chi-square surface separately to the
official NuMI-only profile and to the local NuMI profile.  The local input must
be the registered `q(L|E,nu)` energy--baseline weighted result.  The script
deliberately refuses the historical `local_numi_only` fixed-baseline filenames.

The local `q(L|E,nu)` surface is the same numerical surface used in the
NuMI-only comparison; this is not a second baseline-averaging algorithm.  The
curves can look closer here because the same BNB surface is added to both NuMI
alternatives and each combined surface is shifted to its own minimum.

It performs no fit or Toy generation.  Each sum is shifted to its own minimum
and the common diagnostic level `Delta chi2 = 5.99` is drawn.  This is not the
official correlated BNB+NuMI likelihood and is not a CLs result.
