#!/usr/bin/env bash
# Fetch / verify the four benchmarks used in the paper.
# Run from the repo root with your venv active:  bash scripts/download_datasets.sh
#
# HumanEval ships in the repo (data/humaneval/humaneval.jsonl). ClassEval and
# LiveCodeBench are pulled from the HuggingFace Hub on first use and cached in
# ~/.cache/huggingface. HumanEval+ comes from the `evalplus` package. This
# script pre-fetches them and verifies everything is reachable before you run
# the experiments.
#
# Sources / licenses:
#   HumanEval      Chen et al. 2021 (OpenAI), MIT.            local file (committed)
#   HumanEval+     EvalPlus (Liu et al.), Apache-2.0.         pip: evalplus
#   ClassEval      FudanSELab/ClassEval (Du et al.).          HF dataset
#   LiveCodeBench  bzantium/livecodebench (Jain et al.).      HF dataset; paper uses first 50 problems
set -euo pipefail
cd "$(dirname "$0")/.."

echo "==> 1/4 HumanEval (local, committed)"
if [[ -f data/humaneval/humaneval.jsonl ]]; then
  n=$(wc -l < data/humaneval/humaneval.jsonl | tr -d ' ')
  echo "    OK: data/humaneval/humaneval.jsonl ($n problems; expected 164)"
else
  echo "    ERROR: data/humaneval/humaneval.jsonl missing." >&2; exit 1
fi

echo "==> 2/4 HumanEval+ (evalplus package)"
python -c "from evalplus.data import get_human_eval_plus; print('    OK:', len(get_human_eval_plus()), 'problems')"

echo "==> 3/4 ClassEval (FudanSELab/ClassEval, HF)"
python -c "from datasets import load_dataset; d=load_dataset('FudanSELab/ClassEval', split='test'); print('    OK:', len(d), 'tasks (paper uses all 100)')"

echo "==> 4/4 LiveCodeBench (bzantium/livecodebench, HF)"
python -c "from datasets import load_dataset; d=load_dataset('bzantium/livecodebench', split='test'); print('    OK:', len(d), 'problems total; paper uses the FIRST 50 (run with --limit 50, deterministic order)')"

echo
echo "All datasets reachable. See REPRODUCE.md for the run commands."
