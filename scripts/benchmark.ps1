# scripts/benchmark.ps1
# Runs mini_batcher.exe across a sweep of request rates and records the
# one-line summary each run prints into a results CSV.
#
# Usage: .\scripts\benchmark.ps1
#        .\scripts\benchmark.ps1 -OutFile results\step2.csv -DurationSeconds 5
#        .\scripts\benchmark.ps1 -OutFile results\step3_delay5000.csv -ExtraArgs 5000
#        .\scripts\benchmark.ps1 -OutFile results\step4_i4_d5000.csv -ExtraArgs 4,5000,0.7

param(
    [string]$Binary = (Join-Path $PSScriptRoot "..\mini_batcher.exe"),
    [string]$OutFile = (Join-Path $PSScriptRoot "..\results\step1.csv"),
    [int]$DurationSeconds = 5,
    # Extra positional args appended after <rps> <seconds>, e.g. delay_us,
    # or instances,delay_us,contention for Step 4.
    [string[]]$ExtraArgs = @()
)

$rpsValues = 20, 50, 80, 100, 150, 200, 500, 1000

if (-not (Test-Path $Binary)) {
    throw "Binary not found at $Binary. Build it first: g++ -std=c++17 -O2 -pthread main.cpp -o mini_batcher.exe -lwinmm"
}

$resultsDir = Split-Path $OutFile -Parent
if (-not (Test-Path $resultsDir)) {
    New-Item -ItemType Directory -Path $resultsDir | Out-Null
}

$pattern = 'rps=(\d+)\s+completed=(\d+)\s+sent=(\d+)\s+avg=([\d.]+)ms\s+p50=([\d.]+)ms\s+' +
           'p99=([\d.]+)ms\s+max=([\d.]+)ms\s+runmodel_calls=(\d+)\s+avg_batch=([\d.]+)\s+busy_pct=([\d.]+)'
$rows = @()

foreach ($rps in $rpsValues) {
    Write-Host "Running at $rps rps for $DurationSeconds s (args: $ExtraArgs)..."
    $output = & $Binary $rps $DurationSeconds @ExtraArgs
    $errorLine = $output | Where-Object { $_ -match '!!! ERROR' }
    if ($errorLine) {
        Write-Warning "rps=$rps : $errorLine"
    }
    $line = $output | Select-Object -Last 1
    if ($line -match $pattern) {
        $rows += [PSCustomObject]@{
            rps            = [int]$Matches[1]
            completed      = [int]$Matches[2]
            sent           = [int]$Matches[3]
            avg            = [double]$Matches[4]
            p50            = [double]$Matches[5]
            p99            = [double]$Matches[6]
            max            = [double]$Matches[7]
            runmodel_calls = [int]$Matches[8]
            avg_batch      = [double]$Matches[9]
            busy_pct       = [double]$Matches[10]
        }
    } else {
        Write-Warning "Could not parse output at rps=$rps : '$line'"
    }
}

$rows | Export-Csv -Path $OutFile -NoTypeInformation
Write-Host "Wrote $($rows.Count) rows to $OutFile"
