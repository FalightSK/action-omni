$py="C:\Users\Admin\miniconda3\envs\libero-smolvla\python.exe"
$dir="E:\work\action-omni\docs\experiments\libero_readout\scripts"
$env:HF_HOME="E:\hf_cache"; $env:MUJOCO_GL="wgl"
$ranges=@(@(0,4),@(4,7),@(7,10))
$jobs=@()
foreach($r in $ranges){
  $jobs += Start-Process -FilePath $py -PassThru -NoNewWindow -WorkingDirectory $dir `
    -ArgumentList @("rollout_smolvla.py","--episodes","10","--max-steps","400",
                    "--task-start",$r[0],"--task-end",$r[1]) `
    -RedirectStandardOutput "$dir\smolvla_$($r[0])_$($r[1]).log" `
    -RedirectStandardError "$dir\smolvla_$($r[0])_$($r[1]).err"
  Write-Output "launched tasks $($r[0])-$($r[1]) pid=$($jobs[-1].Id)"
  Start-Sleep -Seconds 10
}
$jobs | Wait-Process
Write-Output "SMOLVLA_ALL_DONE"
