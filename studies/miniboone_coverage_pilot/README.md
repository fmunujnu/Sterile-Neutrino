# MiniBooNE coverage pilot

This study compares the collaboration's released 90% coverage contour with
the usual fixed two-parameter likelihood threshold and a local fake-experiment
calibration.  It imports the active MiniBooNE prediction/covariance adapter and
does not modify or enter the default MicroBooNE or MiniBooNE scan.

For each selected point on the official contour, Gaussian fake datasets are
drawn by Cholesky factorisation of that point's 38-bin covariance. Every fake
dataset is fit over the complete released 190x190 grid plus the exact tested
point (which is usually between grid nodes). Outputs are visible CSV/JSON/PNG.
