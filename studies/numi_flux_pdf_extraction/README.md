# NuMI flux PDF vector-extraction study

This directory is an isolated validation study. It does not feed the active
MicroBooNE likelihood. Its first gate uses the MINERvA NuMI release because the
same release provides both a vector PDF plot and human-readable numerical flux
arrays. Only after that known-truth closure passes should the extractor be
configured for MicroBooNE Public Note 1129.

## Closed-loop benchmark

The benchmark performs five independent operations:

1. select a long stroked vector path from the PDF, excluding text, axes, ticks,
   filled heat-map cells, and legend samples;
2. detect the enclosing axes from PDF line geometry;
3. associate numeric labels with tick marks and choose a linear or base-10
   logarithmic coordinate model by residual;
4. recover one value for every horizontal histogram segment and write CSV;
5. independently project both the official TXT truth and the recovered CSV
   back into absolute PDF coordinates and overlay them on a rendered page.

Run from the repository root with the bundled PDF Python runtime:

```powershell
& "C:\Users\Fmunu\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" `
  studies\numi_flux_pdf_extraction\run_benchmark.py
```

Outputs are written to:

```text
outputs/checks/numi_flux_pdf_extraction/minerva_numu_fhc/
```

The important files are:

- `recovered_flux.csv`: PDF-derived physical bin coordinates and values;
- `validation.json`: axis fits, curve selection, truth errors, and geometry errors;
- `overlay_truth_vs_pdf.png`: official TXT array projected over the PDF;
- `overlay_recovered_vs_pdf.png`: recovered CSV projected over the PDF;
- `detected_objects.json`: auditable axes, ticks, and selected-path metadata.

Passing this benchmark proves that the program recovered the plotted MINERvA
curve. It does not prove that a later MicroBooNE extraction equals an unpublished
ROOT histogram, and MINERvA flux must never be substituted for MicroBooNE flux.

The real MicroBooNE layout/log-axis integration gate is:

```powershell
& "C:\Users\Fmunu\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" `
  studies\numi_flux_pdf_extraction\check_microboone_note.py
```

It checks all eight black total-flux paths on Figures 9-12: one unique path per
panel, 400 PDF points, 200 horizontal segments, linear energy axis, logarithmic
flux axis, TeX-flattened power-of-ten labels, and boundary clipping. It is a
layout/extraction check only because no MicroBooNE numerical truth array is
publicly available.

## Requested PDF pages 5-6

This standalone local program extracts the blue `New Flux (Geant 4.10.4)`
paths specifically from PDF pages 5 and 6, writes visible CSV files, reads those
CSV files back, and projects them over the original pages:

```powershell
& "C:\Users\Fmunu\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" `
  studies\numi_flux_pdf_extraction\extract_microboone_pages_5_6.py
```

It uses local `pdfplumber`, NumPy, Pillow, and Poppler only. No model API,
cloud API, browser automation, or network connection is involved after the
source PDF has been downloaded.

For a copied standalone checkout, install the visible requirements and run the
PowerShell wrapper with an ordinary local Python interpreter:

```powershell
python -m pip install -r studies\numi_flux_pdf_extraction\requirements.txt
powershell -ExecutionPolicy Bypass -File `
  studies\numi_flux_pdf_extraction\run_pages_5_6.ps1 `
  -Python python
```

## Complete FHC/RHC flavor arrays from PDF pages 4-7

The pages 4-7 orchestration layer leaves the validated core extractor under
`src/` unchanged. It extracts the blue New Flux paths from Figures 1-4 and
uses the relations printed in the note:

- pages 4-5: FHC/RHC `numu` and `nue` components;
- pages 6-7: FHC/RHC `numu + numubar` and `nue + nuebar` totals;
- derived antineutrino: same-mode total minus neutrino component.

Run locally with no API or network connection:

```powershell
powershell -ExecutionPolicy Bypass -File `
  studies\numi_flux_pdf_extraction\run_pages_4_7.ps1 `
  -Python python
```

The eight final CSV arrays are written under
`outputs/checks/numi_flux_pdf_extraction/microboone_pages_4_7/flux_arrays/`.
Every array uses 50 bins from 0 to 5 GeV at 0.1 GeV spacing. The displayed
histogram is natively 0.1 GeV only through 2 GeV and uses wider steps above
2 GeV. Wider displayed steps are copied piecewise-constantly to the enclosed
0.1 GeV bins because the published ordinate is already per 100 MeV; the code
does not interpolate an unshown shape.

Absolute PDF point coordinates, physical coordinates, automatic linear/log10
axis fits, inferred display bounds, and boundary-censoring flags remain visible
in `source_coordinate_audit/` and `extraction_report.json`. The final arrays
remain PDF-recovered values rather than official unpublished ROOT arrays.
