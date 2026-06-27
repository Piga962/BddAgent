# results/ — pre-generated answers for the paper

These are the raw model outputs and per-run summaries the paper's tables and
figures are computed from. They are committed so the package is self-contained:
you can regenerate every figure **without** re-calling any LLM API. Re-running
the experiments (see `REPRODUCE.md`) will add new timestamped files alongside
these; it will not overwrite them.

## Filename schema

```
{condition}_{model}_{YYYYMMDD}_{HHMMSS}.jsonl   # per-problem outputs
summary_{model}_{YYYYMMDD}_{HHMMSS}.json         # per-run aggregate (pass@1, tokens, ...)
```

- `condition` ∈ `{bdd, cot, direct}`. **`bdd` is the on-disk label for the
  paper's method, TCGP (test-case-guided prompting)** — see the note in the
  top-level `README.md`. `cot` = Chain-of-Thought, `direct` = direct prompting.
- `model` is the provider model id (e.g. `gpt-4.1`, `gemini-2.5-flash`,
  `claude-haiku-4-5-20251001`).

## Which directory feeds which paper artifact

| Directory | Benchmark | Paper artifact(s) |
|-----------|-----------|-------------------|
| `bdd_vs_cot/` | HumanEval (n=164) | `tab:humaneval`, `fig:multimodel`/`tab:bdd-vs-cot-multimodel`, `fig:heatmap`, `tab:costs`, and the derived `fig:bdd-cot`/`fig:cot-hurts`/`fig:tokens` |
| `bdd_vs_cot_anthropic/` | HumanEval (n=164) | **Canonical Claude-Haiku / Claude-Sonnet runs** for the same artifacts (the Anthropic models' HumanEval runs live here, not in `bdd_vs_cot/`) |
| `humaneval_plus/`, `evalplus/` | HumanEval+ | `fig:robustness` / `tab:humanevalplus` |
| `classeval/` | ClassEval (n=100) | `fig:three-benchmarks` / `tab:task-difficulty` |
| `livecodebench/` | LiveCodeBench (first 50) | `fig:three-benchmarks` / `tab:task-difficulty`, `fig:token-exhaustion` |
| `humaneval_ablation/`, `ablations/` | HumanEval | Supplementary §E1–E4 (ablations) |
| `analysis_output/` | — | Intermediate analysis artifacts (error distributions, bootstrap) |
| `EXPERIMENT_SUMMARY.md` | — | Human-readable run summary |

`results/figure_data.json` is the machine-generated bridge from these runs to the
figures: `build_figure_data.py` pins the exact canonical run per model (the one
whose pass counts reproduce the paper's Pass@1) and writes the numbers that
`generate_figures.py` then plots. Run `python build_figure_data.py --check` to
verify the committed answers still reproduce the paper (all 10 HumanEval Pass@1
values match; GPT-5.3-codex average token counts differ by ~2%, which only
affects the token-efficiency figure, not any Pass@1 result).

Legacy/not-in-paper result sets (DevEval, early ablations) and the abandoned
Plan-B partial Claude reruns live under `archive/results-legacy/` and are not
shipped.
