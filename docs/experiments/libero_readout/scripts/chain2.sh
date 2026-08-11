#!/bin/bash
# Run stages strictly sequentially - concurrent CPU jobs thrash this box.
cd /workspace/omni
export HF_HOME=/workspace/hf_cache

# 1. wait for the CPU probe job to finish
while [ ! -f probe_results.npz ]; do sleep 15; done
echo "=== probes done, plotting ==="
python plot1.py > plot1.log 2>&1
echo "=== plot done ==="

# 2. GPU feature cache, with thread caps so preprocessing actually gets scheduled
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
python cache_train.py > cache.log 2>&1
echo "=== cache done ==="

# 3. tap-depth sweep, 12 taps x 3 LRs
python train_sweep.py > sweep.log 2>&1
echo "=== ALL_DONE ==="
