# SmolVLM2: does the Qwen paraphrase collapse replicate? orig vs para1, matched.
$py="C:\Users\Admin\miniconda3\envs\libero-qwen\python.exe"
$dir="E:\work\action-omni\docs\experiments\libero_readout\scripts"
$env:HF_HOME="E:\hf_cache"; $env:MUJOCO_GL="wgl"
$jobs=@()
foreach($v in @("orig","para1")){
  $jobs += Start-Process -FilePath $py -PassThru -NoNewWindow -WorkingDirectory $dir `
    -ArgumentList @("rollout_smolvlm.py","--tap","30","--mode","all",
                    "--variant",$v,"--episodes","20","--max-steps","400") `
    -RedirectStandardOutput "$dir\sroll_$v.log" -RedirectStandardError "$dir\sroll_$v.err"
  Write-Output "launched smolvlm $v pid=$($jobs[-1].Id)"
  Start-Sleep -Seconds 8
}
$jobs | Wait-Process
Write-Output "SMOLVLM_PARA_DONE"
