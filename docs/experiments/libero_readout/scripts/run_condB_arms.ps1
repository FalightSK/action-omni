# Condition B model (trained on orig/para1/para2): seen vs HELD-OUT phrasing.
$py="C:\Users\Admin\miniconda3\envs\libero-qwen\python.exe"
$dir="E:\work\action-omni\docs\experiments\libero_readout\scripts"
$env:HF_HOME="E:\hf_cache"; $env:MUJOCO_GL="wgl"
$arms=@(@{v="orig"},@{v="para3"})
$jobs=@()
foreach($a in $arms){
  $jobs += Start-Process -FilePath $py -PassThru -NoNewWindow -WorkingDirectory $dir `
    -ArgumentList @("rollout_qwen.py","--tap","12","--mode","all","--tag","qckmix",
                    "--variant",$a.v,"--episodes","20") `
    -RedirectStandardOutput "$dir\qroll_mix_$($a.v).log" -RedirectStandardError "$dir\qroll_mix_$($a.v).err"
  Write-Output "launched condB $($a.v) pid=$($jobs[-1].Id)"
  Start-Sleep -Seconds 8
}
$jobs | Wait-Process
Write-Output "CONDB_ARMS_DONE"
