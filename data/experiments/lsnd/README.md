# LSND final 2001 public record

This directory is deliberately narrower than `data/experiments/miniboone`: the
LSND Collaboration final publication makes the detector and statistical method
auditable, but does **not** publish an event table, the 4-D component PDFs,
background-variation inputs, covariance, or a machine-readable likelihood
surface/contour. The active optional `rate-scan` is calculated from printed
scalar measurements and an explicit analytic DAR weighting approximation.

An isolated research path in `studies/lsnd_public_spectrum_reweighting/`
additionally extracts the vector coordinates of final-paper Fig. 16 and Fig. 24.
It is not an active raw input: its CSV files and scans remain under `outputs/`
because the plotted beam-excess errors do not supply a bin covariance and the
low-mass reference signal is not the collaboration's full-transmutation MC.

`published/final_2001_summary.csv` transcribes only scalar values printed in
Table X, Table XI, and Sec. IX.H of Aguilar *et al.*, Phys. Rev. D 64, 112007
(2001), arXiv:hep-ex/0104049. It is a source-indexed fact table, not a binned
dataset and not sufficient to calculate a likelihood.

`sources/MANIFEST.csv` records URLs, paper locations, purpose, access date and
the explicit absence of a local PDF hash. A local byte copy was not retained:
the download request was cancelled. Consequently its SHA-256 field is blank,
rather than fabricated. The raw scientific inputs are logically read-only.

## What the final paper establishes

- Primary result: `anti-nu_mu -> anti-nu_e` from `mu+` DAR, observed as
  `anti-nu_e p -> e+ n`; source-to-detector centre is 30 m and the DAR
  electron energy selection is 20--60 MeV (paper pp. 3, 6, 35).
- The full oscillation fit uses 5697 beam-on events, four variables
  `(Ee, Rgamma, cos(theta_nu), z)`, `20<Ee<200 MeV`, and combines primary DAR
  with secondary `nu_mu -> nu_e` from pion DIF (paper Sec. IX, pp. 22--27).
- Signal and background PDFs were MC-derived except beam-unrelated background
  from beam-off data; background variations are Gaussian weighted (Sec. IX.B--E,
  pp. 23--24). This is neither a released Gaussian covariance chi-square nor a
  published Poisson-bin likelihood.
- The paper describes Feldman--Cousins generated-data calibration, but says it
  was not followed for the final paper because of CPU cost. It publishes
  constant-log-likelihood slices, `Lmax-L<2.3` and `<4.6`, for 90% and 99%
  regions (Sec. IX.F, pp. 24--25; Fig. 27, p. 64). Thus the contours must not
  be relabelled as a newly reproduced FC/Toy coverage calculation.

## Deliberate limits

The exact public-record modes export (a) the published scalar record and (b) an
auditable call to the project 3+1 short-baseline appearance probability at the
published best-fit coordinate. The additional `rate-scan` mode constructs a
model-portable **approximate** DAR rate likelihood from the published average
probability. It integrates the analytic anti-muon-neutrino Michel spectrum,
leading IBD phase space, and the published detector length/distance. It assumes
constant selection efficiency and lacks the source extent, transverse detector
geometry, energy migration and four-variable event PDFs. Consequently it can
test the 3+1 units and broad allowed-band trend and can later accept a 1+3+1
probability, but it is not an official-surface reproduction or coverage result.

## Public binned-spectrum study

The final paper contains two useful vector figures for the clean
`Rgamma > 10` DAR subset. Fig. 16 has five reconstructed-positron-energy bins;
Fig. 24 has eleven reconstructed `L/E` bins. Both show beam-on minus beam-off
points, remaining neutrino-background components, and a low-mass oscillation
reference stack. The study extracts these paths without raster point picking,
reconstructs their asymmetric plotted errors, and verifies the Fig. 16 sums
against the printed `49.1` beam excess, `16.9` neutrino background and `32.2`
signal excess.

For a model-portable diagnostic it divides the low-mass signal by the
bin-averaged small-phase `x^2` shape and normalizes the resulting `L/E` kernel
with `33300 * 0.39`, the printed full-transmutation count and correlated-gamma
efficiency. The likelihood is an independent-bin Gaussian for the plotted
beam-excess points, with paper-supported signal and total-background
normalization nuisances profiled at every scan point. It is more informative than the one-rate approximation and
can accept a future 1+3+1 appearance probability, but it still is not promoted
as official LSND input because it lacks covariance, nuisance constraints and
the four-dimensional event information.
