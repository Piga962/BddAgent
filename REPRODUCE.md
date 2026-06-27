# Reproducing the paper

Two paths: **(A)** regenerate every figure/table from the committed answers (no
API keys, minutes), and **(B)** re-run the experiments end-to-end (API keys +
budget + hours). Most reviewers want (A).

## A. Regenerate figures/tables from committed results

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python build_figure_data.py          # results/ -> results/figure_data.json
python build_figure_data.py --check  # verify reproduction vs the paper
python generate_figures.py           # fig1..fig9 -> figures/
python cost_analysis.py              # tab:costs
```

Expected `--check` output: **all Pass@1 values across fig1/fig5/fig9 reproduce
the paper**, with two warnings (see Caveats). `generate_figures.py` writes 9
figures (PNG+PDF) to `figures/` (override with `--out DIR`).

### How verification works

`build_figure_data.py` recomputes each figure's numbers directly from the
per-problem `results/*.jsonl` (Pass@1 = passed / N) and the run summaries,
pinning the canonical run per model, then compares against the pristine paper
literals kept in `generate_figures.py` (`_PAPER_*`). `generate_figures.py`
plots the results-derived numbers, falling back to the literals only if
`results/figure_data.json` is absent.

## B. Re-run experiments (optional)

```bash
cp .env.example .env        # add API keys
bash scripts/download_datasets.sh

# HumanEval (per model); providers: azure | openai | gemini | anthropic
python run_tcgp_vs_cot.py   --model gpt-4o                     --provider openai    --seed 42
python run_tcgp_vs_cot.py   --model claude-haiku-4-5-20251001  --provider anthropic --seed 42
# ClassEval (100) and LiveCodeBench (first 50)
python run_classeval.py     --model gpt-4o --provider openai --limit 100
python run_livecodebench.py --model gpt-4o --provider openai --limit 50
# HumanEval+ robustness over committed HumanEval generations
python run_humaneval_plus.py --all
```

Results land as new timestamped files under `results/`; the committed answers
are never overwritten. Note that even with fixed seeds, providers may update
model weights/sampling over time, so exact pass rates are not guaranteed to
reproduce across dates.

## Caveats (reported by `--check` as warnings)

1. **GPT-5.3-codex token averages** differ ~2% from the paper
   (e.g. TCGP 849 vs 870). Pass@1 is unaffected; this only shifts the
   token-efficiency bars (fig4) slightly.
2. **Claude-Haiku LiveCodeBench (22/14/16)** could not be reproduced from any
   committed run or chunk combination. It is marked `UNVERIFIED` and the paper
   value is carried (not derived) into fig9. The underlying run file appears to
   be missing from the committed set.

Both are intentional, surfaced rather than hidden, so reviewers see exactly
what the committed data does and does not back.

## Notes on Anthropic runs

- Canonical Claude HumanEval runs live in `results/bdd_vs_cot_anthropic/` (not
  `results/bdd_vs_cot/`).
- Claude-Sonnet HumanEval completed 158/164 problems; Pass@1 is computed over
  the full 164 denominator (24/164 = 14.6%), matching the paper.
