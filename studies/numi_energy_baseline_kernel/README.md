# NuMI energy--baseline kernel pilot

This isolated study builds an auditable approximation to
`psi(E_nu, L, flavour)` without changing the active MicroBooNE prediction,
oscillation, likelihood, or scan code.

The conditional baseline shape is taken from publicly downloadable NuMI
`dk2nu` events.  Each event supplies the parent decay vertex and a precomputed
ray to the location named `MicroBooNE`.  The absolute energy marginal is not
taken from this older public beam sample.  Instead, every energy bin is
renormalised to the eight MicroBooNE flux curves already registered under
`data/experiments/microboone/numi/inputs/flux_components`:

```text
integral psi(E, L, flavour) dL = Phi_MicroBooNE(E, flavour).
```

The available public sample is a Geant4 9.2, medium-energy, negative-200-kA
(RHC) NuMI production.  It is not the Geant4 4.10.4 + updated PPFX production
used for the final Nature analysis.  RHC conditional shapes are read directly;
FHC shapes use the charge-conjugate RHC flavour as an explicit proxy.  The
result is therefore a geometry-informed sensitivity model, not an official
MicroBooNE input.

Run after installing `uproot` and `awkward` from `requirements.txt`:

```powershell
python studies/numi_energy_baseline_kernel/build_psi.py --files 4
```

Results are written to
`outputs/studies/numi_energy_baseline_kernel/<batch>/results`.  The long-form
CSV files are the numerical products; PNG files are checks only.  Increasing
`--files` improves the weighted statistics but does not remove the beam-version
or FHC-proxy limitations.

Sources:

- Public NuMI dk2nu directory:
  https://portal.nersc.gov/project/dune/data/misc/NuMI_dk2nu/newtarget-200kA_20220409/
- Dk2Nu reader and field documentation:
  https://github.com/woodtp/dk2nu-numi-flux
- MicroBooNE flux ancestry/decay-position description, Sec. 5.3:
  https://lss.fnal.gov/archive/thesis/2000/fermilab-thesis-2021-20.pdf
- MicroBooNE NuMI flux update:
  https://microboone.fnal.gov/wp-content/uploads/MICROBOONE-NOTE-1129-PUB.pdf

