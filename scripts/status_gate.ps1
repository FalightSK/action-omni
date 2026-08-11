# Progress for the LIBERO instruction gate (10 tasks x 3 conditions x N episodes).
#
#   powershell -ExecutionPolicy Bypass -File scripts\status_gate.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\status_gate.ps1 -Watch
#
# The interesting number is not the success rate on its own -- it is whether the
# rate MOVES between conditions. Rows are printed as they complete so the trend
# is visible long before the run finishes.
#
# ASCII only: Windows PowerShell 5.1 decodes a BOM-less file as the system ANSI
# codepage (cp874 here), so a curly quote or em dash breaks parsing outright.

param([switch]$Watch, [int]$Every = 60, [string]$Run = "exp05_groot_2view", [int]$Conditions = 2)

$log = Join-Path $PSScriptRoot ("..\asset\runs\libero\" + $Run + "\gate_eval.log")
$TOTAL = 10 * $Conditions   # 10 tasks x the conditions this run used

function Show-Status {
    Write-Host ("libero gate - " + (Get-Date -Format 'HH:mm:ss')) -ForegroundColor Cyan
    if (-not (Test-Path $log)) { Write-Host "  (no log yet)"; return }

    $lines = Get-Content $log
    $rows = $lines | Select-String -Pattern "^\s+task\s+(\d+)\s+(\w+)\s+SR\s+([\d.]+)%\s+\((\d+)/(\d+)\)"
    Write-Host ("`n[done] {0}/{1} task-conditions" -f $rows.Count, $TOTAL) -ForegroundColor Yellow

    # running totals per condition -- the comparison IS the result
    $agg = @{}
    foreach ($r in $rows) {
        $g = $r.Matches[0].Groups
        $c = $g[2].Value
        if (-not $agg.ContainsKey($c)) { $agg[$c] = @(0, 0) }
        $agg[$c][0] += [int]$g[4].Value
        $agg[$c][1] += [int]$g[5].Value
    }
    if ($agg.Count -gt 0) {
        Write-Host "`n[running SR by condition]" -ForegroundColor Yellow
        foreach ($c in @("canonical", "swapped", "empty")) {
            if ($agg.ContainsKey($c)) {
                $w = $agg[$c][0]; $n = $agg[$c][1]
                $pct = if ($n) { 100.0 * $w / $n } else { 0 }
                Write-Host ("   {0,-11} {1,5:N1}%   ({2}/{3})" -f $c, $pct, $w, $n)
            }
        }
    }

    $last = $rows | Select-Object -Last 3
    if ($last) {
        Write-Host "`n[recent]" -ForegroundColor Yellow
        foreach ($r in $last) { Write-Host ("   " + $r.Line.Trim()) }
    }

    $p = Get-Process python -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($p -and $rows.Count -gt 0) {
        $mins = ((Get-Date) - $p.StartTime).TotalMinutes
        $rate = $mins / $rows.Count
        Write-Host ("`n[rate] {0:N1} min per task-condition | ~{1:N0} min left" -f `
            $rate, (($TOTAL - $rows.Count) * $rate))
    }
    elseif (-not $p) { Write-Host "`n[proc] no python -- finished or died" -ForegroundColor DarkGray }

    $err = $lines | Select-String -Pattern "Traceback|Error|GATE_EXIT"
    if ($err) {
        Write-Host "`n[!]" -ForegroundColor Red
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
