# NuMI-only official-grid comparison

This isolated study compares two versions of the repository's approximate
NuMI-only detector adapter with the collaboration's released NuMI-only observed
delta-chi-square grid: the original fixed 0.680 km baseline and an explicit
public-dk2nu-informed energy--baseline average.  All three curves use a fixed
`delta chi-square = 5.99` diagnostic criterion; neither CLs nor Toy MC is used.

The local calculation reuses the active 3+1 prediction, likelihood and profile
functions.  It does not alter or enable the production analysis registry.  Its
detector model remains approximate because it borrows the BNB response, freezes
the aggregate published background, and has no response support from 3--5 GeV.
The E--L input is additionally limited by an old public RHC production and an
explicit charge-conjugate proxy for FHC.  Flux is not multiplied twice: only
the conditional baseline mass is used to average each oscillation probability.

Run:

```text
python studies/numi_only_official_comparison/run.py --workers 8
```
