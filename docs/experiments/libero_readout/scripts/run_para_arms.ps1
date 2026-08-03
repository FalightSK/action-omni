$py="C:\Users\Admin\miniconda3\envs\libero-qwen\python.exe"
$dir="E:\work\action-omni\docs\experiments\libero_readout\scripts"
$env:HF_HOME="E:\hf_cache"; $env:MUJOCO_GL="wgl"
$arms=@(@{mode="all";tap=12},@{mode="instr";tap=12})
$jobs=@()
foreach($a in $arms){
  $tag="$($a.mode)_tap$($a.tap)_para1"
  $jobs += Start-Process -FilePath $py -PassThru -NoNewWindow -WorkingDirectory $dir `
    -ArgumentList @("rollout_qwen.py","--tap",$a.tap,"--mode",$a.mode,
                    "--variant","para1","--episodes","20") `
    -RedirectStandardOutput "$dir\qroll_$tag.log" -RedirectStandardError "$dir\qroll_$tag.err"
  Write-Output "launched $tag pid=$($jobs[-1].Id)"
  Start-Sleep -Seconds 8
}
$jobs | Wait-Process
Write-Output "PARA_ARMS_DONE"
