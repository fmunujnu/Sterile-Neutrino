# Public-dk2nu energy--baseline pilot

`psi_exposure_weighted_four_flavours.csv` is a reusable, visible derived input
for the NuMI-only diagnostic.  Its energy marginal is the registered
MicroBooNE PDF-extracted flux, while its conditional baseline shapes come from
two publicly readable RHC NuMI dk2nu files.  FHC shapes use the documented
charge-conjugate RHC proxy.

This is not a collaboration release and is not enabled in the declared joint
analysis.  The active NuMI adapter uses it only when a caller explicitly passes
the CSV path.  `source_generation_metadata.json` records provenance,
normalisation closure and scientific limitations.
