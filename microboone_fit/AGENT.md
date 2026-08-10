# MicroBooNE BNB analysis rules

## Active scientific scope

- Active code is a BNB-only four-channel (104 reconstructed-bin) 3+1 analysis.
- The active channels are `nue_cc_fc`, `nue_cc_pc`, `numu_cc_fc`, and `numu_cc_pc`.
- NuMI, BNB pi0/NC channels, 1+3+1 physics, and multi-experiment combinations are outside the active likelihood until their own declared inputs and validations exist.
- Never describe an output as the MicroBooNE Collaboration result, an internal-MC reproduction, or a full BNB+NuMI fit.

## Parameter contract

- Public APIs and configuration use only `delta_m2_41_eV2`, `sin2_theta14`, and `sin2_theta24`.
- `delta_m2_41_eV2` means Δm²41 in eV², never a mass, a square root, or a logarithm.
- `sin2_theta14` and `sin2_theta24` mean sin² of the named angle. Convert to radians only in `parameters.py`.
- Do not introduce bare names such as `dm41`, `theta14`, `mixing`, or `amplitude` in active code.
- The exact appearance amplitude must be derived from the mixing matrix; do not silently substitute a small-angle formula.

## Data and detector contract

- `data/raw/` is read-only. Keep the source DOI and SHA-256 in `data/manifests/sources.yaml`.
- Do not silently slice a public table. Declare every channel and bin selection in `binning.py`.
- A profile run requires an explicit 104-bin total covariance archive and explicit detector-folded true-energy templates. It must fail if either file is absent or malformed.
- A response template already contains flux, cross section, efficiency, selection and migration. Never multiply those quantities a second time. Neutrino and antineutrino templates are separate required inputs.
- Do not calibrate predictions to observed data. Reference validation is always against the published nominal prediction.

## Statistical contract

- A profile point fixes the requested scan coordinates and minimizes every other active physical parameter.
- Use `profile_three_plus_one` or `profile_grid`; do not call a fixed-grid χ² evaluation a profile likelihood.
- Record the statistical covariance prescription in every result. For the target 2025 paper, use the documented Pearson prescription; the HEPData table header's conflicting CNP wording is only an explicit cross-check mode, never a silent default.
- The active Gaussian likelihood assumes a covariance fixed at its declared reference point. Do not claim a paper-level profile if its covariance is parameter-dependent until a validated `C(parameters)` implementation exists.
- Use Cholesky solves, not explicit matrix inversion. Active likelihood code must never silently regularize a covariance; a non-positive-definite covariance is an input error to diagnose.

## Frozen material

- `frozen/baseline_v1/` and `frozen/current_system_backup/` are evidence only. Do not import, execute, modify, or use them in active predictions.
- If legacy behavior is examined, do it in a separate diagnostic script and label it as historical; never add a compatibility adapter to `src/`.

## Required checks after an edit

1. Run `python -m pytest -q`.
2. Run `python -m sterile_fit.cli inspect-published` with `PYTHONPATH=src`.
3. Run `python -m sterile_fit.cli validate-3plus1` with `PYTHONPATH=src`.
4. If templates or covariance readers change, add a negative test that proves invalid shapes, names, or reference mismatches are rejected.
5. Separate confirmed facts, implementation assumptions, and unresolved uncertainties in the final report.
