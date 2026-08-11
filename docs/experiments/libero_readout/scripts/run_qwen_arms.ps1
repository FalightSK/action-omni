# Four closed-loop Qwen arms in parallel (~2 GB VRAM each on a 12 GB card).
#   all/tap12 orig + swap  -> language dependence
#   instr/tap12 orig       -> read-out token subset (vs all/tap12 orig)
#   all/tap0   orig        -> A1: zero LM transformer layers
$py   = "C:\Users\Admin\miniconda3\envs\libero-qwen\python.exe"
$dir  = "E:\work\action-omni\docs\experiments\libero_readout\scripts"
$eps  = 20     # per task -> 200 episodes per arm
$arms = @(
  @{mode="all";   tap=12; var="orig"},
  @{mode="all";   tap=12; var="swap"},
  @{mode="instr"; tap=12; var="orig"},
  @{mode="all";   tap=0;  var="orig"}
)
$jobs = @()
foreach ($a in $arms) {
  $tag = "$($a.mode)_tap$($a.tap)_$($a.var)"
  $log = "$dir\qroll_$tag.log"
  $jobs += Start-Process -FilePath $py -PassThru -NoNewWindow `
    -WorkingDirectory $dir `
    -ArgumentList @("rollout_qwen.py","--tap",$a.tap,"--mode",$a.mode,
                    "--variant",$a.var,"--episodes",$eps) `
    -RedirectStandardOutput $log -RedirectStandardError "$dir\qroll_$tag.err"
  Write-Output "launched $tag pid=$($jobs[-1].Id)"
  Start-Sleep -Seconds 8   # stagger model loads so VRAM allocation doesn't spike
}
Write-Output "waiting on $($jobs.Count) arms..."
$jobs | Wait-Process
Write-Output "ALL_ARMS_DONE"
