# MiniBooNE public inputs

`shared/raw/official_nue2020_combined/` is an unchanged local copy of the
MiniBooNE 2020 combined neutrino/antineutrino electron-appearance release,
corrected by the collaboration on 2021-02-23.  Every file is plain text or
HTML.  `SHA256SUMS.csv` records the byte-level copy and its direct URL.

The release contains 11 electron-like and 8 muon-control bins in each horn
polarity, two event-level full-transmutation samples, a 60x60 fractional
covariance, the collaboration's 190x190 likelihood surface, and four
frequency-calibrated contours.  Raw files are read-only inputs; no rebinning,
interpolation, digitisation, or PDF extraction was used.

The local adapter converts the six covariance blocks to the 38 observed bins
exactly as described by the release.  The `scan` mode reconstructs
`chi2 + log|M|`; it is a validation calculation and is not labelled as the
collaboration's frequentist coverage calculation.
