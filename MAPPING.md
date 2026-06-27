# Paper artifact → script → results → dataset

`bdd` in code/filenames = **TCGP** in the paper (see README naming note).
Figures are produced by `generate_figures.py`, which reads numbers from
`results/figure_data.json` (built by `build_figure_data.py` from the committed
runs). Run `python build_figure_data.py --check` to verify reproduction.

| Paper artifact | Script(s) | Result file(s) / dir | Dataset |
|---|---|---|---|
| `tab:humaneval` (Gemini-2.5-Flash, n=164) | `run_tcgp_vs_cot.py` | `results/bdd_vs_cot/{bdd,cot,direct}_gemini-2.5-flash_*.jsonl` | HumanEval |
| `fig:multimodel` / `tab:bdd-vs-cot-multimodel` (10 models) | `run_tcgp_vs_cot.py` → `generate_figures.py` (fig1) | `results/bdd_vs_cot/` + `results/bdd_vs_cot_anthropic/` (Claude) | HumanEval |
| `fig:bdd-cot` (capability scatter) | `generate_figures.py` (fig2) | derived from HumanEval `MODELS_DATA` | HumanEval |
| `fig:cot-hurts` | `generate_figures.py` (fig3) | derived from `MODELS_DATA` | HumanEval |
| `fig:tokens` (token efficiency) | `generate_figures.py` (fig4) | derived from `MODELS_DATA` tokens | HumanEval |
| `tab:costs` | `cost_analysis.py` | `MODELS_DATA` tokens/pass-rates + fixed pricing | HumanEval |
| `fig:heatmap` | `generate_figures.py` (fig6) | derived from `MODELS_DATA` | HumanEval |
| `fig:robustness` / `tab:humanevalplus` (5 models) | `run_humaneval_plus.py` → `generate_figures.py` (fig5) | `results/humaneval_plus/humaneval_plus_summary_v2.json` | HumanEval+ |
| `fig:three-benchmarks` / `tab:task-difficulty` | `run_classeval.py`, `run_livecodebench.py` → `generate_figures.py` (fig9) | `results/classeval/`, `results/livecodebench/`, HumanEval | ClassEval, LiveCodeBench, HumanEval |
| `fig:token-exhaustion` | `generate_figures.py` (fig8) | `results/analysis_output/error_analysis_results.json` | LiveCodeBench |

## Datasets

| Dataset | How obtained | Notes |
|---|---|---|
| HumanEval | `data/humaneval/humaneval.jsonl` (committed) | 164 problems |
| HumanEval+ | `evalplus` package (`get_human_eval_plus()`) | ~80× more tests/problem |
| ClassEval | HF `FudanSELab/ClassEval` (split=test) | 100 class-level tasks |
| LiveCodeBench | HF `bzantium/livecodebench` (split=test) | paper uses the **first 50** (`--limit 50`, deterministic) |

Run `bash scripts/download_datasets.sh` to fetch/verify all four.

## Canonical runs

Several models were run multiple times. `build_figure_data.py` pins the exact
canonical run per model (the one whose pass counts reproduce the paper's
Pass@1). Claude-Haiku ClassEval is summed from its two partial chunks
(39 + 61 = 100). See `CANONICAL_RUNS` / `CLASSEVAL_RUNS` / `LIVECODEBENCH_RUNS`
in `build_figure_data.py`.
