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
- Read the BNB baseline from `configs/bnb_3nu_anchor.yaml`. The active value is `0.4685 km`; never copy the frozen `0.541 km` value into active MicroBooNE code.
- The active 3+1 probability is the paper's short-baseline limit: the first three mass states are degenerate and only `delta_m2_41_eV2` drives a phase.

## Data and detector contract

- `data/raw/` is read-only. Keep the source DOI and SHA-256 in `data/manifests/sources.yaml`.
- Do not silently slice a public table. Declare every channel and bin selection in `binning.py`.
- A profile run requires an explicit 104-bin total covariance CSV plus JSON metadata and explicit CSV detector-folded true-energy templates. It must fail if any visible file is absent or malformed.
- Active numerical inputs and scan outputs use CSV/JSON/YAML only. NPZ, pickle, opaque binary arrays, or hidden serialization conventions are forbidden.
- A response template already contains flux, cross section, efficiency, selection and migration. Never multiply those quantities a second time. Neutrino and antineutrino templates are separate required inputs.
- Do not calibrate predictions to observed data. Reference validation is always against the published nominal prediction.

## Statistical contract

- A profile point fixes the requested scan coordinates and minimizes every other active physical parameter.
- Use `profile_three_plus_one` or `profile_grid`; do not call a fixed-grid χ² evaluation a profile likelihood.
- Record the statistical covariance prescription in every result. For the target 2025 paper, use the documented Pearson prescription; the HEPData table header's conflicting CNP wording is only an explicit cross-check mode, never a silent default.
- The stored total covariance is a reference audit artifact. Active scans must use `PredictionScaledGaussianLikelihood`: rescale the released systematic covariance with the current/reference prediction ratio and add current Pearson statistics. Never silently revert to the fixed reference matrix.
- Profile the full unitary `sin2_theta14` domain. At fixed appearance amplitude, the large-`sin2_theta14` branch is not generally redundant because `|U_mu4|^2=(1-sin2_theta14)sin2_theta24` changes. Prefits must explicitly test zero-appearance boundary surfaces.
- The released aggregate Background is frozen only because its oscillatable and non-oscillatable components are unavailable. Preserve this limitation in metadata and never call the resulting curve a collaboration-exact exclusion.
- Use Cholesky solves, not explicit matrix inversion. Active likelihood code must never silently regularize a covariance; a non-positive-definite covariance is an input error to diagnose.

## Frozen material

- `frozen/baseline_v1/` and `frozen/current_system_backup/` are evidence only. Do not import, execute or modify them. The user-selected BNB flux vectors have been syntax-parsed once into `data/inputs/bnb_flux.csv`; active predictions read only that visible CSV and its provenance JSON.
- If legacy behavior is examined, do it in a separate diagnostic script and label it as historical; never add a compatibility adapter to `src/`.

## Required checks after an edit

1. Run `python -m pytest -q`.
2. Run `python scripts/check.py` from the repository root.
3. If templates or covariance readers change, add a negative test that proves invalid shapes, names, or reference mismatches are rejected.
4. Separate confirmed facts, implementation assumptions, and unresolved uncertainties in the final report.
