# MicroBooNE per-Toy profile convergence

This study does not change active inference. It selects three representative
points close to the existing fixed-Toy 95% CLs contour in each formal scan
coordinate (Fig.3a appearance and Fig.3b electron disappearance). It then
reuses the active prediction, covariance, Toy generator and profile functions,
but re-runs the allowed mixing-parameter profile inside every Toy.

Nested prefixes of one deterministic 5000-Toy stream are evaluated at
N=50,100,200,500,1000,2000,5000. Thus changes with N are not confounded by
unrelated random samples. CSV files retain every test statistic and tail flag.
