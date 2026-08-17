param(
    [string]$Python = "python"
)

$studyRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$program = Join-Path $studyRoot "extract_microboone_pages_5_6.py"

& $Python -c "import numpy, pdfplumber, PIL"
if ($LASTEXITCODE -ne 0) {
    throw "Missing local Python packages. Install requirements.txt first."
}

if (-not (Get-Command pdftocairo -ErrorAction SilentlyContinue)) {
    throw "pdftocairo (Poppler) must be available on PATH."
}

& $Python $program
if ($LASTEXITCODE -ne 0) {
    throw "Pages 5-6 extraction failed."
}
