# Experiment module contract

Each detector/beam package owns its binning, public-input readers, archival
adapters, detector-folded templates, predictor and workflow. It exposes a named
chi-square contribution through `sterile_fit.analysis.registry`.

Shared oscillation models, covariance algorithms, likelihood mathematics,
fitting and profile scans remain outside experiment packages. The combination
layer knows experiment IDs and scalar chi-square values, not file formats or
channel definitions.

