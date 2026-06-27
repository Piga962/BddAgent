#!/usr/bin/env bash
# Plan B as a single sequential queue. One run at a time, cheapest first so
# results bank early. Each run uses the patched runner with subprocess-based
# execute_test (HumanEval/75 brute-force no longer hangs).
#
# Watchdog: 10 minutes of log silence => SIGKILL and advance. The fixed runner
# should never go quiet that long because subprocess timeout is 30s, so any
# real hang would surface as fast per-test timeouts well under the watchdog.

set -u

cd "$(dirname "$0")/.."

LOG_DIR="logs/plan_b_seq"
mkdir -p "$LOG_DIR"
MASTER="$LOG_DIR/_master.log"

STALE_THRESHOLD=600  # 10 min — generous; fixed runner should never need this

run_one() {
  local label="$1"; shift
  local log="$LOG_DIR/${label}.log"
  : > "$log"
  echo "[$(date +%H:%M:%S)] START $label" | tee -a "$MASTER"

  python3 run_tcgp_vs_cot.py "$@" > "$log" 2>&1 &
  local pid=$!

  while kill -0 "$pid" 2>/dev/null; do
    sleep 30
    local now mtime stale
    now=$(date +%s)
    mtime=$(stat -f '%m' "$log" 2>/dev/null || echo "$now")
    stale=$(( now - mtime ))
    if (( stale > STALE_THRESHOLD )); then
      echo "[$(date +%H:%M:%S)] $label: stale ${stale}s — SIGKILL pid $pid" | tee -a "$MASTER"
      kill -9 "$pid" 2>/dev/null
      wait "$pid" 2>/dev/null
      echo "[$(date +%H:%M:%S)] WATCHDOG-KILL $label" | tee -a "$MASTER"
      return 1
    fi
  done

  wait "$pid" 2>/dev/null
  local rc=$?
  if [[ $rc -eq 0 ]]; then
    echo "[$(date +%H:%M:%S)] DONE  $label" | tee -a "$MASTER"
  else
    echo "[$(date +%H:%M:%S)] FAIL  $label rc=$rc" | tee -a "$MASTER"
  fi
}

# Sequential queue: cheapest-first to bank quickly, expensive last.
# gemini-2.5-flash s100 is skipped — already banked at summary_20260504_143029.json.
run_one gemini-3.1-flash-lite_s100  --model gemini-3.1-flash-lite-preview --provider gemini    --seed 100
run_one gemini-3.1-flash-lite_s200  --model gemini-3.1-flash-lite-preview --provider gemini    --seed 200
run_one gpt-35-turbo_s100           --model gpt-35-turbo                  --provider azure     --seed 100
run_one gpt-35-turbo_s200           --model gpt-35-turbo                  --provider azure     --seed 200
run_one claude-haiku-4.5_s100       --model claude-haiku-4-5-20251001     --provider anthropic --seed 100
run_one claude-haiku-4.5_s200       --model claude-haiku-4-5-20251001     --provider anthropic --seed 200
run_one claude-sonnet-4_s100        --model claude-sonnet-4-20250514      --provider anthropic --seed 100
run_one gpt-4.1_s100                --model gpt-4.1                       --provider azure     --seed 100
run_one gpt-4o_s100                 --model gpt-4o                        --provider azure     --seed 100
run_one gpt-5.3-codex_s100          --model gpt-5.3-codex                 --provider azure_responses --seed 100

echo "[$(date +%H:%M:%S)] sequential queue complete" | tee -a "$MASTER"
