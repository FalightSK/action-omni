# Condition B: train on a MIX of phrasings {orig,para1,para2}; hold out para3.
$py="C:\Users\Admin\miniconda3\envs\libero-qwen\python.exe"
$dir="E:\work\action-omni\docs\experiments\libero_readout\scripts"
$env:HF_HOME="E:\hf_cache"; $env:MUJOCO_GL="wgl"
Set-Location $dir
# wait for condition-A arms to release the GPU
while (Get-Process -Name python -ErrorAction SilentlyContinue |
       Where-Object { $_.CommandLine -like "*rollout_qwen.py*" }) { Start-Sleep 20 }
Write-Output "=== A done; caching mixed-phrasing training features ==="
$env:MIX_PHRASINGS="1"
& $py cache_qwen_mix.py 2>&1 | Select-String -Pattern "DONE|window|ALL" | Select-Object -Last 5
Write-Output "=== training on mixed phrasings ==="
$env:TRAIN_TAG="qmix"
& $py sweep_qwen_mix.py 2>&1 | Select-String -Pattern '^\{|QSWEEP' | Select-Object -Last 8
Write-Output "CHAIN_MIX_DONE"
