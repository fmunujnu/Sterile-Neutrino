# Experiment data layout

Use one directory per detector and one child directory per beam or dataset:

```text
<detector>/
  shared/       files genuinely shared across that detector's datasets
  <beam>/
    inputs/     visible flux and external numerical inputs
    raw*/       immutable published source material
    derived/    normalized responses and covariance products
    reweighting/ detector-folded parameter-dependent templates
```

Do not reuse one beam's response as another beam's response. A new dataset is
registered only after its own provenance, prediction, covariance and validation
exist.
