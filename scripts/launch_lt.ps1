# Thin launcher for the scheduled task: runs the LT pipeline and tees all output to
# the pipeline log. Kept separate so the schtasks action has no nested quoting.
& "F:\work\capstone\action-omni\scripts\run_lt_pipeline.ps1" *>&1 |
    Out-File -FilePath "F:\work\capstone\action-omni\asset\runs\language_table\exp01_baseline\pipeline.log" -Encoding utf8
