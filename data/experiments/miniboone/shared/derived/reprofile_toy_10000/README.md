# MiniBooNE full-grid reprofile-Toy calibration

`point_calibration.csv` is the merged result of the completed `relics1`
batch `mb_full_10000_half` at Git commit
`8e851b989590584f83ad032869beaec84fac8210`.

- Grid: the complete public MiniBooNE 190 x 190 grid (36,100 points).
- Toys: 10,000 pseudo-experiments at every tested point.
- Generator: the public 38-bin Gaussian prediction and its point-dependent
  covariance.
- Profile: every pseudo-experiment is independently minimized over the same
  complete 190 x 190 grid.
- Seed: deterministic per-point stream derived from master seed `20260913`.

This is the primary local MiniBooNE calibration used by `python run.py
miniboone`. It remains a public-input reconstruction and is not the
collaboration's internal calibration.
