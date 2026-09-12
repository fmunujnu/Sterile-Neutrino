$ErrorActionPreference = "Stop"

$repositoryRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\.."))
$serverRepository = '$HOME/data/jobs/sterile-neutrino'

if ((git -C $repositoryRoot status --porcelain).Length -ne 0) {
    throw "Working tree is not clean. Commit the intended changes before synchronizing."
}

git -C $repositoryRoot push origin main
git -C $repositoryRoot push relics2 main
ssh relics2 "git -C $serverRepository pull --ff-only"

Write-Host "relics2 synchronized to:" -NoNewline
ssh relics2 "git -C $serverRepository rev-parse --short HEAD"
