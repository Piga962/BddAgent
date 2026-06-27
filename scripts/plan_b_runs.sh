#!/usr/bin/env bash
# Plan B multi-seed runs with a per-run external watchdog. Three provider
# queues run in parallel; within each queue, runs are sequential to respect
# per-provider rate limits. If a run's log file goes stale (no append) for
# more than $STALE_THRESHOLD seconds, the python process is SIGKILL'd and the
# queue advances — partial data for that run is discarded.
#
# Usage:
#   scripts/plan_b_runs.sh azure
#   scripts/plan_b_runs.sh anthropic
#   scripts/plan_b_runs.sh gemini

set -u

cd "$(dirname "$0")/.."

LOG_DIR="logs/plan_b"
mkdir -p "$LOG_DIR"

STALE_THRESHOLD=180  # 3 min of log silence => watchdog kills

run_one() {
  local label="$1"; shift
  local log="$LOG_DIR/${label}.log"
  : > "$log"
  echo "[$(date +%H:%M:%S)] START $label" | tee -a "$LOG_DIR/_master.log"

  python3 run_bdd_vs_cot.py "$@" > "$log" 2>&1 &
  local pid=$!

  while kill -0 "$pid" 2>/dev/null; do
    sleep 30
    local now mtime stale
    now=$(date +%s)
    mtime=$(stat -f '%m' "$log" 2>/dev/null || echo "$now")
    stale=$(( now - mtime ))
    if (( stale > STALE_THRESHOLD )); then
      echo "[$(date +%H:%M:%S)] $label: stale ${stale}s — SIGKILL pid $pid" | tee -a "$LOG_DIR/_master.log"
      kill -9 "$pid" 2>/dev/null
      wait "$pid" 2>/dev/null
      echo "[$(date +%H:%M:%S)] WATCHDOG-KILL $label (data discarded)" | tee -a "$LOG_DIR/_master.log"
      return 1
    fi
  done

  wait "$pid" 2>/dev/null
  local rc=$?
  if [[ $rc -eq 0 ]]; then
    echo "[$(date +%H:%M:%S)] DONE  $label" | tee -a "$LOG_DIR/_master.log"
  else
    echo "[$(date +%H:%M:%S)] FAIL  $label rc=$rc" | tee -a "$LOG_DIR/_master.log"
  fi
  return 0
}

case "${1:-}" in
  azure)
    run_one gpt-35-turbo_s100   --model gpt-35-turbo --provider azure --seed 100
    run_one gpt-35-turbo_s200   --model gpt-35-turbo --provider azure --seed 200
    run_one gpt-4o_s100         --model gpt-4o       --provider azure --seed 100
    run_one gpt-4.1_s100        --model gpt-4.1      --provider azure --seed 100
    run_one gpt-5.3-codex_s100  --model gpt-5.3-codex --provider azure_responses --seed 100
    ;;
  anthropic)
    run_one claude-haiku-4.5_s100  --model claude-haiku-4-5-20251001  --provider anthropic --seed 100
    run_one claude-haiku-4.5_s200  --model claude-haiku-4-5-20251001  --provider anthropic --seed 200
    run_one claude-sonnet-4_s100   --model claude-sonnet-4-20250514   --provider anthropic --seed 100
    ;;
  gemini)
    # gemini-2.5-flash seed=100 already completed (summary_20260504_143029.json) — skipped
    run_one gemini-3.1-flash-lite_s100  --model gemini-3.1-flash-lite-preview --provider gemini --seed 100
    run_one gemini-3.1-flash-lite_s200  --model gemini-3.1-flash-lite-preview --provider gemini --seed 200
    ;;
  *)
    echo "Usage: $0 {azure|anthropic|gemini}" >&2
    exit 2
    ;;
esac

echo "[$(date +%H:%M:%S)] queue $1 complete" | tee -a "$LOG_DIR/_master.log"
