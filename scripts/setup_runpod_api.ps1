param([switch]$Check)

$ErrorActionPreference = 'Stop'
$credentialPath = Join-Path $env:LOCALAPPDATA 'ActionOmni\runpod-api-key.dpapi'

if ($env:OS -ne 'Windows_NT') {
    throw 'This helper uses Windows DPAPI. Run it in Windows PowerShell.'
}

if ($Check) {
    if (-not (Test-Path -LiteralPath $credentialPath)) {
        Write-Error 'No saved key. Run this script once without -Check to enter it.'
        exit 2
    }
    try {
        $secret = Get-Content -LiteralPath $credentialPath -Raw | ConvertTo-SecureString
    } catch {
        Write-Error 'Could not decrypt the saved key for this Windows account.'
        exit 2
    }
} else {
    $secret = Read-Host 'Enter RunPod API key (input is hidden)' -AsSecureString
    if ($secret.Length -eq 0) {
        Write-Error 'No key entered.'
        exit 2
    }
}

$pointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secret)
try {
    $apiKey = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($pointer)
    try {
        $null = Invoke-RestMethod -Method Get -Uri 'https://api.runpod.io/v2/pods' `
            -Headers @{ Authorization = "Bearer $apiKey" } -TimeoutSec 20
        $status = 200
    } catch {
        $status = 0
        if ($_.Exception.Response -and $_.Exception.Response.StatusCode) {
            $status = [int]$_.Exception.Response.StatusCode
        }
    }
} finally {
    [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($pointer)
    Remove-Variable apiKey -ErrorAction SilentlyContinue
}

if ($status -ge 200 -and $status -lt 300) {
    if (-not $Check) {
        New-Item -ItemType Directory -Path (Split-Path -Parent $credentialPath) -Force | Out-Null
        ConvertFrom-SecureString $secret | Set-Content -LiteralPath $credentialPath -Encoding Ascii -NoNewline
    }
    Write-Host 'RunPod API key accepted. It is stored encrypted for this Windows account.'
    exit 0
}

if ($status -eq 401) {
    Write-Error 'RunPod rejected the key (HTTP 401); it was not saved.'
} elseif ($status -eq 403) {
    Write-Error 'RunPod denied this request (HTTP 403); check key permissions. It was not saved.'
} elseif ($status -eq 0) {
    Write-Error 'Could not reach the RunPod API; the key was not saved.'
} else {
    Write-Error "RunPod API check failed (HTTP $status); the key was not saved."
}
exit 1
