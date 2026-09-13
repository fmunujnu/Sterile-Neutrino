# Registered public-dk2nu energy--baseline input

`psi_exposure_weighted_four_flavours.csv` is a reusable, visible derived input
for the NuMI-only diagnostic.  Its energy marginal is the registered
MicroBooNE PDF-extracted flux, while its conditional baseline shapes come from
two publicly readable RHC NuMI dk2nu files.  FHC shapes use the documented
charge-conjugate RHC proxy.

This is not a collaboration release.  It is the registered conditional-
baseline input used by the active approximate BNB+NuMI analysis.  The event
kernel already contains the energy-marginal flux, so the adapter normalises
each flavour and energy row to q(L|E,flavour) and does not multiply the flux a
second time.  `source_generation_metadata.json` is the immutable generation-
time record; its historical statement that the output was not connected to the
active analysis describes the state when it was generated, before promotion.
