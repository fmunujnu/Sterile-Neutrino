# MicroBooNE BNB analysis rules

## Active scientific scope

- Active code is a BNB-only four-channel (104 reconstructed-bin) 3+1 analysis.
- The active channels are `nue_cc_fc`, `nue_cc_pc`, `numu_cc_fc`, and `numu_cc_pc`.
- NuMI, BNB pi0/NC channels, 1+3+1 physics, and multi-experiment combinations are outside the active likelihood until their own declared inputs and validations exist.
- Never describe an output as the MicroBooNE Collaboration result, an internal-MC reproduction, or a full BNB+NuMI fit.
- Experiment-specific readers, adapters, response templates, and plots live under `experiments/<detector>/<beam>/`; shared physics and statistics must not import beam-specific file layouts.
- `configs/analyses/*.yaml` is the only authority for selecting contributions to a combined scan. An entry marked `inputs_unavailable` must fail if enabled.

## Parameter contract

- Public APIs and configuration use only `delta_m2_41_eV2`, `sin2_theta14`, and `sin2_theta24`.
- `delta_m2_41_eV2` means Δm²41 in eV², never a mass, a square root, or a logarithm.
- `sin2_theta14` and `sin2_theta24` mean sin² of the named angle. Convert to radians only in `parameters.py`.
- Do not introduce bare names such as `dm41`, `theta14`, `mixing`, or `amplitude` in active code.
- The exact appearance amplitude must be derived from the mixing matrix; do not silently substitute a small-angle formula.
- Read the BNB baseline from `configs/experiments/microboone/bnb/analysis.yaml`. The active value is `0.4685 km`; never copy the frozen `0.541 km` value into active MicroBooNE code.
- The active 3+1 probability is the paper's short-baseline limit: the first three mass states are degenerate and only `delta_m2_41_eV2` drives a phase.

## Data and detector contract

- Raw files under `data/experiments/<detector>/<beam>/` or the detector's `shared/` directory are read-only. Keep source DOI and SHA-256 in the detector-level provenance file.
- Do not silently slice a public table. Declare every channel and bin selection in `binning.py`.
- A profile run requires an explicit 104-bin total covariance CSV plus JSON metadata and explicit CSV detector-folded true-energy templates. It must fail if any visible file is absent or malformed.
- Active numerical inputs and scan outputs use CSV/JSON/YAML only. NPZ, pickle, opaque binary arrays, or hidden serialization conventions are forbidden.
- A response template already contains flux, cross section, efficiency, selection and migration. Never multiply those quantities a second time. Neutrino and antineutrino templates are separate required inputs.
- Do not calibrate predictions to observed data. The current reference closure is against the HEPData unconstrained total only; it is an empirical algebraic anchor, not evidence that this table is the paper 3nu/null prediction.

## Canonical experiment layout: copy the BNB workflow

The active MicroBooNE BNB layout is the sole reference for every new beam or
experiment workflow.  Do not invent another directory layer, an
`intermediate_checks/` tree, or a second input/output convention.  Before adding
files, map them to these exact responsibilities:

```text
configs/experiments/<detector>/<beam>/analysis.yaml
    Declares scientific status, baseline, reference parameters and visible paths.

data/experiments/<detector>/shared/raw/
    Immutable collaboration/public release shared by beams.  Store provenance
    and hashes; readers validate the full release before selecting channels.

data/experiments/<detector>/<beam>/raw_response/
    Immutable beam-specific public response source, only when not shared.

data/experiments/<detector>/<beam>/inputs/
    Human-readable primary beam inputs such as flux CSV plus provenance JSON.

data/experiments/<detector>/<beam>/derived/
    Reproducible, reusable prepared inputs: normalized Reco matrices and declared
    covariance CSV/JSON.  No plots, fit results, test files or one-off event scans.

data/experiments/<detector>/<beam>/reweighting/
    The visible kernel contract only: true-energy grid, fixed background, all
    source-to-final response-count CSVs, metadata and reference_closure.csv.

src/sterile_fit/experiments/<detector>/<beam>/
    Stable beam adapter API.  Follow BNB names and separation: binning.py,
    published_inputs.py, templates.py, prediction.py and workflow.py.  A beam
    that is not likelihood-ready may stop before workflow.py, but must not create
    a competing interface.  Shared physics/statistics never imports this layout.

scripts/experiments/<detector>/<beam>/
    Thin reproducible orchestration only: prepare inputs, build kernel and plot a
    single experiment.  Scripts may call src interfaces; src never imports scripts.

outputs/spectra/<detector>/<beam>/
    Regenerable figures and their same-stem CSV/metadata sidecars.

outputs/scans/<analysis>/<model>/
    Regenerable prefit/profile/parameter-space results.

outputs/checks/
    Explicit audit evidence only.  It is not an input dependency.  If a checked
    artifact is promoted to an input, copy it to data with provenance and hash.
```

An experiment input or kernel must never depend on `outputs/`, `tmp/`, a plot,
or a pytest directory.  `data/` may contain only reusable scientific inputs;
`outputs/` may be deleted and regenerated without changing any prediction.
PNG files are never numerical inputs.

### Required BNB-style call boundary

```text
public/shared release + beam inputs
    -> experiment published_inputs/binning adapter
    -> prepared derived Reco/covariance
    -> experiment reweighting kernel + reference closure
    -> experiment predictor/workflow
    -> common likelihood/profile code
    -> outputs/spectra or outputs/scans
```

Every new beam must keep the same public API meanings as BNB.  Do not reuse BNB
global-bin offsets, baseline, flux, covariance slice or numeric kernel.  Reuse
only generic physics/statistics and the documented interface pattern.  Any
temporary approximation such as a borrowed Reco matrix belongs in that beam's
adapter metadata and must remain disabled in combined analysis selection.

### One shared spectrum renderer is mandatory

BNB defines the sole spectrum-plotting contract.  BNB, NuMI, another beam, and
future experiments must call one shared rendering function; they must not each
implement their own Matplotlib layout, stepping convention, overflow handling,
colours, legend order, error-bar style, axis labels or CSV/metadata sidecars.

Experiment-specific code may provide only a validated plotting payload:

```text
beam/channel identifiers
reconstructed-energy edges
Data and asymmetric errors
Background
Signal + Background
named model/reference prediction vectors
scientific labels and provenance
```

The shared renderer owns every visual and output-format decision.  Adding a new
beam means writing or changing only its data adapter/payload builder.  If two
beam plots look structurally different, treat that as an interface violation,
not as acceptable experiment-specific styling.  Never copy the BNB plotting
body into a NuMI or joint script; factor or call the existing shared renderer.

### Temporary-file policy

- `__pycache__/`, `.pytest_cache/`, `pytest_*`, `tmp/pytest-*`, and
  `outputs/testing/` are disposable and must not be retained after a task.
- Tests normally use the operating-system temporary directory.  If Windows
  permissions require `--basetemp`, use one uniquely named workspace directory,
  delete it immediately after the run, and never cite it as an analysis output.
- Do not create repeated `audit_*`, `layout_*`, or `*_final2` folders.  A retained
  audit must have a unique scientific purpose, stable name and metadata; otherwise
  delete it after its result is reported.
- Do not place check plots or test summaries under `data/experiments/...`.

## Statistical contract

- A profile point fixes the requested scan coordinates and minimizes every other active physical parameter.
- Use `profile_three_plus_one` or `profile_grid`; do not call a fixed-grid χ² evaluation a profile likelihood.
- Paper-like pointwise exclusion uses `CLs=p_4nu/p_3nu` from the right tail of
  `T=chi2_4nu-chi2_3nu`.  The production calibration is Toy MC under both
  hypotheses.  A Gaussian moment approximation may remain only as an explicitly
  selected fast diagnostic and must never be labelled Toy-calibrated.
- Every Toy MC pseudo-experiment must repeat the same physical profile as the
  observed scan point.  Holding the observed-data nuisance optimum fixed while
  evaluating toys is not a profiled Toy result.
- Generate pseudo-data with the full covariance already used by the Gaussian
  likelihood.  Because its diagonal already contains the chosen Pearson count
  variance, never add a second Poisson fluctuation.  Do not clip negative
  Gaussian pseudo-counts or silently regularize the covariance.
- Record the Toy count per hypothesis and scan point, base seed, derived point
  seed rule, tail convention, finite-ensemble p-value correction, nuisance
  generator prescription, profile method and worker count.  Store Monte Carlo
  uncertainty next to every CLs value; optimizer/thread parallelism may improve
  speed only when it leaves seeded numerical results invariant within tolerance.
- Record the statistical covariance prescription in every result. For the target 2025 paper, use the documented Pearson prescription; the HEPData table header's conflicting CNP wording is only an explicit cross-check mode, never a silent default.
- The stored total covariance is a reference audit artifact. Active scans must use `PredictionScaledGaussianLikelihood`: rescale the released systematic covariance with the current/reference prediction ratio and add current Pearson statistics. Never silently revert to the fixed reference matrix.
- Profile the full unitary `sin2_theta14` domain. At fixed appearance amplitude, the large-`sin2_theta14` branch is not generally redundant because `|U_mu4|^2=(1-sin2_theta14)sin2_theta24` changes. Prefits must explicitly test zero-appearance boundary surfaces.
- The released aggregate Background is frozen only because its oscillatable and non-oscillatable components are unavailable. Preserve this limitation in metadata and never call the resulting curve a collaboration-exact exclusion.
- Use Cholesky solves, not explicit matrix inversion. Active likelihood code must never silently regularize a covariance; a non-positive-definite covariance is an input error to diagnose.

## Frozen material

- `frozen/baseline_v1/` and `frozen/current_system_backup/` are evidence only. Do not import, execute or modify them. The user-selected BNB flux vectors have been syntax-parsed once into `data/experiments/microboone/bnb/inputs/bnb_flux.csv`; active predictions read only that visible CSV and its provenance JSON.
- If legacy behavior is examined, do it in a separate diagnostic script and label it as historical; never add a compatibility adapter to `src/`.

## Required checks after an edit

1. Run `python -m pytest -q`.
2. Run `python scripts/check.py` from the repository root.
3. If templates or covariance readers change, add a negative test that proves invalid shapes, names, or reference mismatches are rejected.
4. Separate confirmed facts, implementation assumptions, and unresolved uncertainties in the final report.
