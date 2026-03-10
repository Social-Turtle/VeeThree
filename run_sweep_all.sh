#!/usr/bin/env bash
# run_sweep_all.sh — run all three Pareto sweeps then plot.
# Logs each sweep to results/logs/. Continues even if one sweep fails.
#
# Usage:
#   ./run_sweep_all.sh          # full sweeps (slow — run overnight)
#   ./run_sweep_all.sh --quick  # smoke-test configs (~5 min)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS="$SCRIPT_DIR/results"
LOGS="$RESULTS/logs"
mkdir -p "$LOGS"

QUICK=""
if [[ "${1:-}" == "--quick" ]]; then
  QUICK="--quick"
  echo "=== QUICK mode ==="
fi

START=$(date +%s)
echo "Started: $(date)"
echo "Logs → $LOGS/"
echo ""

run_sweep() {
  local name="$1"
  local script="$2"
  local log="$LOGS/${name}_$(date +%Y%m%d_%H%M%S).log"

  echo "──────────────────────────────────────────"
  echo "▶  $name  $(date '+%H:%M:%S')"
  echo "   log: $log"

  if uv run python "$SCRIPT_DIR/$script" $QUICK 2>&1 | tee "$log"; then
    echo "✓  $name done  $(date '+%H:%M:%S')"
  else
    echo "✗  $name FAILED (exit $?) — continuing"
  fi
  echo ""
}

run_sweep "cnn_sweep"  "benchmark/cnn_sweep.py"
run_sweep "lut_sweep"  "benchmark/lut_sweep.py"
run_sweep "fe_sweep"   "benchmark/fe_sweep.py"

echo "──────────────────────────────────────────"
echo "▶  plot  $(date '+%H:%M:%S')"
uv run python "$SCRIPT_DIR/benchmark/plot_pareto.py" 2>&1 | tee "$LOGS/plot_$(date +%Y%m%d_%H%M%S).log"

ELAPSED=$(( $(date +%s) - START ))
echo ""
echo "All done in $(( ELAPSED / 60 ))m $(( ELAPSED % 60 ))s"
echo "Plot → $RESULTS/pareto_active_bits.png"
