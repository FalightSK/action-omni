$py="C:\Users\Admin\miniconda3\envs\libero-qwen\python.exe"
$dir="E:\work\action-omni\docs\experiments\libero_readout\scripts"
$res="E:\work\action-omni\docs\experiments\libero_readout\results"
$env:HF_HOME="E:\hf_cache"; $env:MUJOCO_GL="wgl"
Set-Location $dir
while (-not (Test-Path "$res\smix_taps.npy")) { Start-Sleep 30 }
Write-Output "=== cache done; training SmolVLM2 on mixed phrasings ==="
& $py train_smolvlm_mix.py 2>&1 | Select-String -Pattern "step|saved|DONE" | Select-Object -Last 6
Write-Output "=== closed-loop: seen vs HELD-OUT para3 ==="
$jobs=@()
foreach($v in @("orig","para3")){
  $jobs += Start-Process -FilePath $py -PassThru -NoNewWindow -WorkingDirectory $dir `
    -ArgumentList @("rollout_smolvlm.py","--tap","30","--mode","all","--ckpt-tag","ck2mix",
                    "--variant",$v,"--episodes","20","--max-steps","400") `
    -RedirectStandardOutput "$dir\smix_$v.log" -RedirectStandardError "$dir\smix_$v.err"
  Start-Sleep -Seconds 8
}
$jobs | Wait-Process
foreach($v in @("orig","para3")){ Write-Output "$v : $(Select-String -Path "$dir\smix_$v.log" -Pattern '^SR')" }
Write-Output "SMIX_ALL_DONE"
