#!/bin/bash
cd /workspace/omni
# wait for both outstanding arms
while [ ! -f vroll_all_tap0_orig.json ] || [ ! -f vroll_instr_tap30_swap.json ]; do sleep 30; done
echo "=== all arms complete ==="
python final.py > final.log 2>&1
cat final.log
echo "=== FINISH_DONE ==="
