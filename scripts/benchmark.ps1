# scripts/benchmark.ps1
# Runs mini_batcher.exe across a sweep of request rates and records the
# one-line summary each run prints into a results CSV.
#
# Usage: .\scripts\benchmark.ps1
#        .\scripts\benchmark.ps1 -OutFile results\step2.csv -DurationSeconds 5

param(
    [string]$Binary = (Join-Path $PSScriptRoot "..\mini_batcher.exe"),
    [string]$OutFile = (Join-Path $PSScriptRoot "..\results\step1.csv"),
    [int]$DurationSeconds = 5
)

$rpsValues = 20, 50, 80, 100, 150, 200, 500, 1000

if (-not (Test-Path $Binary)) {
    throw "Binary not found at $Binary. Build it first: g++ -std=c++17 -O2 -pthread main.cpp -o mini_batcher.exe -lwinmm"
}

$resultsDir = Split-Path $OutFile -Parent
if (-not (Test-Path $resultsDir)) {
    New-Item -ItemType Directory -Path $resultsDir | Out-Null
}

$pattern = 'rps=(\d+)\s+completed=(\d+)\s+avg=([\d.]+)ms\s+p50=([\d.]+)ms\s+p99=([\d.]+)ms\s+max=([\d.]+)ms'
$rows = @()

foreach ($rps in $rpsValues) {
    Write-Host "Running at $rps rps for $DurationSeconds s..."
    $line = & $Binary $rps $DurationSeconds | Select-Object -Last 1
    if ($line -match $pattern) {
        $rows += [PSCustomObject]@{
            rps       = [int]$Matches[1]
            completed = [int]$Matches[2]
            avg       = [double]$Matches[3]
            p50       = [double]$Matches[4]
            p99       = [double]$Matches[5]
            max       = [double]$Matches[6]
        }
    } else {
        Write-Warning "Could not parse output at rps=$rps : '$line'"
    }
}

$rows | Export-Csv -Path $OutFile -NoTypeInformation
Write-Host "Wrote $($rows.Count) rows to $OutFile"
