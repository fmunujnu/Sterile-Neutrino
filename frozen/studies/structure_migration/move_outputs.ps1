$ErrorActionPreference = 'Stop'
$outputRoot = [IO.Path]::GetFullPath('E:\Sterile Neutrino\outputs')
$moves = [ordered]@{
 'run1_non_toy/process24_full_fig3a_analytic' = 'microboone_bnb_numi_joint/three_plus_one/legacy_process24/scan_fig3a_analytic'
 'run1_non_toy/process24_full_fig3b_analytic' = 'microboone_bnb_numi_joint/three_plus_one/legacy_process24/scan_fig3b_analytic'
 'run2_toy_mc/process24_full_fig3a_adaptive_toy_001_03_100' = 'microboone_bnb_numi_joint/three_plus_one/legacy_process24/scan_fig3a_adaptive-toy'
 'run2_toy_mc/process24_full_fig3b_adaptive_toy_001_03_100' = 'microboone_bnb_numi_joint/three_plus_one/legacy_process24/scan_fig3b_adaptive-toy'
 'run2_toy_mc/contour_comparison' = 'microboone_bnb_numi_joint/three_plus_one/legacy_process24/contour_comparison'
 'run1_non_toy/legacy_20260817_analytic_cls_bnb_numi' = 'microboone_bnb_numi_joint/three_plus_one/legacy_20260817/analytic_results'
 'run1_non_toy/one_plus_three_plus_one/20260823T153657Z' = 'microboone_bnb/one_plus_three_plus_one/20260823T153657Z/scan_mass_pair_analytic'
 'spectra/microboone/bnb' = 'microboone_bnb/three_plus_one/legacy_spectra/spectra'
 'spectra/microboone/bnb_numi_joint' = 'microboone_bnb_numi_joint/three_plus_one/legacy_spectra/spectra'
 'spectra/microboone/numi' = 'microboone_numi/mixed/legacy_spectra/spectra_and_inputs'
 'spectra/microboone/public_inputs' = 'microboone_public/inputs/legacy_spectra/public_plots'
 'archive/legacy_runs' = 'legacy/unclassified/imported/results'
}
foreach ($study in Get-ChildItem -LiteralPath "$outputRoot\checks" -Directory) {
    $moves["checks/$($study.Name)"] = "studies/$($study.Name)/legacy/results"
}
$manifestPath = Join-Path $outputRoot 'migration_20260903.json'
if (Test-Path -LiteralPath $manifestPath) { throw 'Migration already recorded; do not rerun.' }
$records = @()
foreach ($entry in $moves.GetEnumerator()) {
    $source = [IO.Path]::GetFullPath((Join-Path $outputRoot $entry.Key))
    $target = [IO.Path]::GetFullPath((Join-Path $outputRoot $entry.Value))
    foreach ($checked in @($source, $target)) {
        if (-not $checked.StartsWith($outputRoot + '\', [StringComparison]::OrdinalIgnoreCase)) { throw "Outside outputs: $checked" }
    }
    if (-not (Test-Path -LiteralPath $source)) { throw "Missing source: $source" }
    if (Test-Path -LiteralPath $target) { throw "Target exists: $target" }
    $records += [pscustomobject]@{ old = $entry.Key; new = $entry.Value; status = 'pending'; files = @() }
}
$records | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $manifestPath -Encoding utf8
foreach ($record in $records) {
    $source = Join-Path $outputRoot $record.old
    $target = Join-Path $outputRoot $record.new
    $record.files = @(Get-ChildItem -LiteralPath $source -File -Recurse | ForEach-Object {
        [pscustomobject]@{ path = [IO.Path]::GetRelativePath($source, $_.FullName); sha256 = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash }
    })
    $records | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $manifestPath -Encoding utf8
    New-Item -ItemType Directory -Path (Split-Path $target -Parent) -Force | Out-Null
    Move-Item -LiteralPath $source -Destination $target
    foreach ($file in $record.files) {
        if ((Get-FileHash -LiteralPath (Join-Path $target $file.path) -Algorithm SHA256).Hash -ne $file.sha256) { throw 'Content mismatch' }
    }
    $record.status = 'verified'
    $records | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $manifestPath -Encoding utf8
    Write-Output "$($record.old) -> $($record.new) [$($record.files.Count) files verified]"
}
# Remove only explicitly named empty shells, never their contents.
foreach ($relative in @('run1_non_toy/one_plus_three_plus_one', 'run1_non_toy', 'run2_toy_mc', 'spectra/microboone', 'spectra', 'checks')) {
    $folder = Join-Path $outputRoot $relative
    if ((Test-Path -LiteralPath $folder) -and @(Get-ChildItem -LiteralPath $folder -Force).Count -eq 0) { Remove-Item -LiteralPath $folder }
}
