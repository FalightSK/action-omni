# Progress for depth_sweep.py (8 arms x 3 datasets x 7 taps x 3 pools).
#
#   powershell -ExecutionPolicy Bypass -File scripts\status_depth.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\status_depth.ps1 -Watch
#
# Counts completed (arm, dataset) units by the "doc" row, which each unit prints
# exactly once. ASCII only -- Windows PowerShell 5.1 reads a BOM-less file as
# cp874 here, so a curly quote or em dash breaks parsing outright.

param([switch]$Watch, [int]$Every = 30)

$log = Join-Path $PSScriptRoot "..\asset\analysis\latent_compare\depth_v3.log"
$out = Join-Path $PSScriptRoot "..\asset\analysis\latent_compare\depth_sweep.json"
$TOTAL = 24   # 8 arms x 3 datasets

function Show-Status {
    Write-Host ("depth_sweep - " + (Get-Date -Format 'HH:mm:ss')) -ForegroundColor Cyan
    if (-not (Test-Path $log)) { Write-Host "  (no log yet)"; return }

    $lines = Get-Content $log
    $done = ($lines | Select-String -Pattern "^\w+\s+doc\s").Count
    $pct = [math]::Round(100 * $done / $TOTAL)
    Write-Host ("`n[done] {0}/{1} arm-datasets ({2}%)" -f $done, $TOTAL, $pct) -ForegroundColor Yellow

    # Which dataset block we are in, and the newest row.
    $sec = $lines | Select-String -Pattern "^=== (\w+)" | Select-Object -Last 1
    if ($sec) { Write-Host ("[dataset] " + $sec.Matches[0].Groups[1].Value) }
    $last = $lines | Where-Object { $_ -match "^\w+\s+\S+\s" } | Select-Object -Last 1
    if ($last) { Write-Host ("[last row] " + $last.Trim()) }

    # Rate from the process start time -- the log has no timestamps.
    $p = Get-Process python -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($p -and $done -gt 0) {
        $mins = ((Get-Date) - $p.StartTime).TotalMinutes
        $rate = $mins / $done
        $left = [math]::Round(($TOTAL - $done) * $rate, 1)
        Write-Host ("`n[rate] {0:N2} min per arm-dataset | elapsed {1:N0} min | ~{2} min left" -f `
            $rate, $mins, $left)
    }
    elseif (-not $p) {
        Write-Host "`n[proc] no python running -- finished or died" -ForegroundColor DarkGray
    }

    if (Test-Path $out) {
        $age = [math]::Round(((Get-Date) - (Get-Item $out).LastWriteTime).TotalMinutes)
        Write-Host ("[output] depth_sweep.json last written {0} min ago" -f $age)
    }

    $err = $lines | Select-String -Pattern "Traceback|Error"
    if ($err) {
        Write-Host "`n[!] errors:" -ForegroundColor Red
        $err | Select-Object -Last 2 | ForEach-Object { Write-Host ("   " + $_.Line) }
    }
}

if ($Watch) {
    while ($true) {
        Clear-Host; Show-Status
        Write-Host ("`n(refresh " + $Every + "s - Ctrl+C to stop)") -ForegroundColor DarkGray
        Start-Sleep -Seconds $Every
    }
}
else { Show-Status }
