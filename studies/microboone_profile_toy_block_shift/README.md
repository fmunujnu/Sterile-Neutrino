# Coarse block-shift estimate of profiled-Toy contours

This study tiles contiguous log-mass blocks along the existing fixed-Toy
CLs=0.05 contour. Each block samples three masses and five nearby amplitudes,
using 50 profiled Toys per generating hypothesis at every sample point. A
local plane in `(log mass, log amplitude)` is fitted to
`q = p4 - 0.05 p3`; its zero gives one blockwise log-amplitude shift. The
existing contour segment is translated by that shift. Discontinuities between
blocks are intentionally retained as an honest display of the coarse method.

It reuses the active MicroBooNE prediction, covariance, Toy generation and
profile functions and does not modify active scan results.
