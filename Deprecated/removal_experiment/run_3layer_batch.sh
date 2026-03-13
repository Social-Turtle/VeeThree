#!/bin/bash
echo "=== 3-Layer Experiment Batch ==="
echo "Started: $(date)"
echo ""

count=0
total=30

for filters in 6 7 8 9 10; do
    for n_pass in $(seq $filters -1 $((filters - 5))); do
        if [ $n_pass -lt 1 ]; then continue; fi
        count=$((count + 1))
        echo "[$count/$total] 3L/${filters}F/${n_pass}pass"
        uv run python removal_experiment/run_experiment.py --layers 3 --filters $filters --n-pass $n_pass 2>&1 | tail -3
        echo ""
    done
done

echo "=== Completed: $(date) ==="
