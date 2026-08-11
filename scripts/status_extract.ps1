# Progress for the latent_compare re-extraction (8 arms x 4 datasets).
#
#   powershell -ExecutionPolicy Bypass -File scripts\status_extract.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\status_extract.ps1 -Watch
#
# ASCII only, deliberately. Windows PowerShell 5.1 decodes a BOM-less file as
# the system ANSI codepage (cp874 here), so an em dash or curly quote becomes
# mojibake and breaks parsing -- the script then fails to load rather than
# merely printing oddly.

param([switch]$Watch, [int]$Every = 20)

$log = Join-Path $PSScriptRoot "..\asset\analysis\latent_compare\extract_v3.log"
$TOTAL = 32   # 8 arms x 4 datasets

function Show-Status {
    Write-Host ("extract - " + (Get-Date -Format 'HH:mm:ss')) -ForegroundColor Cyan

    if (-not (Test-Path $log)) { Write-Host "  (no log yet)"; return }

    $lines = Get-Content $log

    # Completed units: one "wrote latents_<arm>_<key>.h5" per arm-dataset.
    $wrote = $lines | Select-String -Pattern "wrote latents_(\w+)\.h5\s+\((\d+)s\)"
    $done = $wrote.Count
    $pct = if ($TOTAL) { [math]::Round(100 * $done / $TOTAL) } else { 0 }
    Write-Host ("`n[done] {0}/{1} arm-datasets ({2}%)" -f $done, $TOTAL, $pct) -ForegroundColor Yellow

    # Per-arm tally, so a stalled arm is obvious.
    $byArm = @{}
    foreach ($w in $wrote) {
        $name = $w.Matches[0].Groups[1].Value      # <arm>_<key>
        $arm = ($name -split '_')[0]
        if ($byArm.ContainsKey($arm)) { $byArm[$arm]++ } else { $byArm[$arm] = 1 }
    }
    foreach ($a in $byArm.Keys | Sort-Object) {
        Write-Host ("   {0,-12} {1}/4" -f $a, $byArm[$a])
    }

    # Arm currently loading, and the newest progress line.
    $loading = $lines | Select-String -Pattern "^=== loading (\w+)" | Select-Object -Last 1
    if ($loading) {
        Write-Host ("`n[current] " + $loading.Matches[0].Groups[1].Value) -ForegroundColor Yellow
    }
    $prog = $lines | Select-String -Pattern "^\s+(\w+) (\d+)/(\d+) \(\s*([\d.]+)%\) eta ([\d.]+)m" |
            Select-Object -Last 1
    if ($prog) {
        $g = $prog.Matches[0].Groups
        Write-Host ("   {0}  {1}/{2}  {3}%  eta {4}m" -f `
            $g[1].Value, $g[2].Value, $g[3].Value, $g[4].Value, $g[5].Value)
    }

    # Timing: seconds per completed arm-dataset gives a usable overall estimate.
    if ($done -gt 0) {
        $secs = ($wrote | ForEach-Object { [int]$_.Matches[0].Groups[2].Value })
        $avg = ($secs | Measure-Object -Average).Average
        $left = [math]::Round(($TOTAL - $done) * $avg / 60, 1)
        Write-Host ("`n[rate] {0:N0}s per arm-dataset  |  ~{1} min remaining" -f $avg, $left)
    }

    $err = $lines | Select-String -Pattern "Traceback|OutOfMemory|EXTRACT_EXIT=[1-9]"
    if ($err) {
        Write-Host "`n[!] errors in log:" -ForegroundColor Red
        $err | Select-Object -Last 3 | ForEach-Object { Write-Host ("   " + $_.Line) }
    }

    Write-Host "`n[gpu]" -ForegroundColor Yellow
    try { Write-Host ("   " + (nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader)) }
    catch { Write-Host "   nvidia-smi unavailable" }
}

if ($Watch) {
    while ($true) {
        Clear-Host; Show-Status
        Write-Host ("`n(refresh " + $Every + "s - Ctrl+C to stop)") -ForegroundColor DarkGray
        Start-Sleep -Seconds $Every
    }
}
else { Show-Status }
