# LSND public-spectrum reweighting study

This isolated study extracts the vector coordinates of Fig. 16 and Fig. 24
from the final LSND paper and builds a transparent, reduced 3+1 appearance
likelihood. It does not replace or claim to reproduce LSND's four-variable
event likelihood.

Run from the repository root:

```powershell
python -B studies/lsnd_public_spectrum_reweighting/run.py
```

The primary model-portable approximation is the Fig. 24 `L/E` representation.
Its public low-mass signal template is divided by the small-phase `x^2` shape,
normalized with the paper's 100%-transmutation count and `R_gamma > 10`
efficiency, and reweighted with the common short-baseline appearance
probability. The resulting independent-bin Gaussian likelihood has no public
bin-to-bin covariance and must not be labelled as the official likelihood.

At every scan point the current study analytically profiles two paper-supported
normalization nuisances: `sqrt(10%^2 + 7%^2) = 12.2%` for the selected signal,
and `2.3/16.9 = 13.6%` for the common neutrino-background stack. It does not
invent separate bin-shape pulls or a covariance matrix.

The script performs fail-fast validation. Fig. 16 must close to the printed
clean-sample totals (`49.1` beam excess, `16.9` neutrino background and `32.2`
signal excess), and the published `(1.2 eV2, 0.003)` best-fit coordinate must
remain within `Delta chi2 < 2.3` of both reduced scans. Fig. 16 and Fig. 24 are
also overlaid as an internal stability check; their low-mass 90% regions are
close, while their high-mass fine structure is not treated as reliable because
of coarse binning and missing resolution/covariance information.
