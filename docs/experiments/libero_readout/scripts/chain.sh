#!/bin/bash
cd /workspace/omni
while ! grep -q "^DONE" extract.log 2>/dev/null; do sleep 15; done
echo "extract finished, launching probes(CPU) + cache_train(GPU)"
nohup python probes.py > probes.log 2>&1 &
nohup python cache_train.py > cache.log 2>&1 &
wait
echo CHAIN_DONE
