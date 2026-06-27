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
| `humaneval_plus/`, `evalplus/` | HumanEval+ | `fig:robustness` / `tab:humanevalplus` |
| `classeval/` | ClassEval (n=100) | `fig:three-benchmarks` / `tab:task-difficulty` |
| `livecodebench/` | LiveCodeBench (first 50) | `fig:three-benchmarks` / `tab:task-difficulty`, `fig:token-exhaustion` |
| `humaneval_ablation/`, `ablations/` | HumanEval | Supplementary §E1–E4 (ablations) |
| `analysis_output/` | — | Intermediate analysis artifacts (error distributions, bootstrap) |
| `EXPERIMENT_SUMMARY.md` | — | Human-readable run summary |

Legacy/not-in-paper result sets (DevEval, early ablations, the superseded
`bdd_vs_cot_anthropic` run) live under `archive/results-legacy/` and are not
shipped.
