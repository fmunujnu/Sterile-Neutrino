# MicroBooNE BNB+NuMI completed reprofile-Toy calibration

The CSV files are the merged summary of relics2 batch
`mb_fig3ab_reprofile_5000`, calculated at Git commit
`3183413041746e3d6f68314cfe2cf1e5dc695bf5`.

- Fig.3a grid: 61 x 61 points, mass range 0.01--100 eV2.
- Fig.3b grid: 74 x 61 points, mass range 0.1--40 eV2.
- At every point: 5,000 Toys generated under 3nu and 5,000 under 4nu.
- Every Toy repeats the coordinate-constrained profile.
- `point_calibration.csv` is the canonical combined table; the two smaller
  files are exact figure-specific subsets.

Render through the active entry point:

```text
python run.py scan --model 3+1 --calibration reprofile-toy
```

This is a public-input reconstruction, not MicroBooNE's internal simulation.
