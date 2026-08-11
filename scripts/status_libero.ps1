# Progress for the LIBERO exp01 precompute -> train pipeline.
#
#   powershell -ExecutionPolicy Bypass -File scripts\status_libero.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\status_libero.ps1 -Watch
#
# Why the regex: tqdm writes progress with carriage returns and no newline, so the
# whole bar lives on ONE physical line that grows for hours. Get-Content -Tail 1
# returns that entire line; the last regex match in it is the current state.
# Naive tailing shows either nothing or megabytes of stale bar.
#
# ASCII only, deliberately. This file is read by Windows PowerShell 5.1, which
# decodes a BOM-less file as the system ANSI codepage (cp874 here) -- an em dash
# or curly quote becomes mojibake and breaks string parsing, so the script fails
# to load rather than merely printing oddly.

param([switch]$Watch, [int]$Every = 30)

$dir = Join-Path $PSScriptRoot "..\asset\runs\libero\exp01_goal"

function Show-Status {
    Write-Host ("LIBERO exp01 - " + (Get-Date -Format 'HH:mm:ss')) -ForegroundColor Cyan

    Write-Host "`n[stages]" -ForegroundColor Yellow
    $pl = Join-Path $dir "pipeline.log"
    if (Test-Path $pl) { Get-Content $pl } else { Write-Host "  (not started)" }

    Write-Host "`n[precompute]" -ForegroundColor Yellow
    $pc = Join-Path $dir "precompute.log"
    if (Test-Path $pc) {
        $last = Get-Content $pc -Tail 1
        $m = [regex]::Matches($last, '(\d+)%\|[^|]*\|\s*(\d+)/(\d+)\s*\[([^\]]+)\]')
        if ($m.Count -gt 0) {
            $g = $m[$m.Count - 1].Groups
            Write-Host ("  {0,3}%  batch {1}/{2}   [{3}]" -f $g[1].Value, $g[2].Value, $g[3].Value, $g[4].Value)
        }
        else {
            Write-Host "  starting..."
        }
        if (Select-String -Path $pc -Pattern "Traceback|OutOfMemory" -Quiet) {
            Write-Host "  !! errors present - see precompute.log" -ForegroundColor Red
        }
    }
    else {
        Write-Host "  (pending)"
    }

    # NOTE: HDF5 preallocates the full dataset on creation, so this size reaches
    # its final value almost immediately and does NOT track progress. Use the
    # batch counter above for that; this is only a disk-footprint check.
    $h5 = Join-Path $dir "vlm_embeddings.h5"
    if (Test-Path $h5) {
        Write-Host ("  cache file: {0:N1} GB (preallocated, not a progress signal)" -f ((Get-Item $h5).Length / 1GB))
    }

    Write-Host "`n[train]" -ForegroundColor Yellow
    $tl = Join-Path $dir "train.log"
    if (Test-Path $tl) { Get-Content $tl -Tail 4 } else { Write-Host "  (waiting for precompute)" }

    Write-Host "`n[gpu]" -ForegroundColor Yellow
    try {
        Write-Host ("  " + (nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader))
    }
    catch {
        Write-Host "  nvidia-smi unavailable"
    }

    $free = (Get-PSDrive F).Free / 1GB
    Write-Host ("`n[disk] F: {0:N0} GB free" -f $free)
}

if ($Watch) {
    while ($true) {
        Clear-Host
        Show-Status
        Write-Host ("`n(refreshing every " + $Every + "s - Ctrl+C to stop)") -ForegroundColor DarkGray
        Start-Sleep -Seconds $Every
    }
}
else {
    Show-Status
}
