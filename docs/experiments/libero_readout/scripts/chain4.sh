#!/bin/bash
# Strictly sequential - concurrent GPU+CPU jobs thrash this box.
cd /workspace/omni
export HF_HOME=/workspace/hf_cache HF_HUB_DISABLE_PROGRESS_BARS=1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8

while ! grep -q "^ALL DONE" cache2.log; do sleep 20; done
echo "=== cache2 done ==="

echo "=== token-order intervention (forward-only) ==="
python order_probe.py > order_probe.log 2>&1
tail -20 order_probe.log

echo "=== corrected sweep: 12 taps x {all, instr} read-out ==="
python sweep2.py > sweep2.log 2>&1
grep -E '^\{|SWEEP2_DONE' sweep2.log | tail -40

echo "=== CHAIN4_DONE ==="
