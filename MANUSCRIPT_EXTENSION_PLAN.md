# Manuscript Extension Plan (no new API experiments)

Goal: grow the paper from ~9 to ~12-13 pages using only logged data + prose.
Status: approved in full ("go with all"), including local HumanEval+ re-scoring.
Execution was blocked by a temporary platform outage; resume from Track 1.

## Track 1 — New analyses from logged data (scripts go in this repo)

1. **McNemar paired tests + Holm correction** (`make_paper_analyses.py`, analysis 1)
   - Data is paired (same problems under all 3 strategies on LCB stratified n=180).
   - TCGP vs Direct, TCGP vs CoT, CoT vs Direct; per model + pooled;
     Holm-Bonferroni across the 10 models.
   - Answers the "no significance testing / multiple comparisons" critique.

2. **Partial-correctness breakdown** (analysis 2)
   - LCB records store tests_passed / n_tests; report all/some/none of public
     tests passed per condition. Softens the "Pass@1 is binary" threat.

3. **Reasoning-token analysis** (analysis 3)
   - usage_step1/2.reasoning_tokens logged per call. Question: does TCGP reduce
     hidden deliberation on reasoning models (grok-4-20-reasoning, gpt-5.5)?
   - Report either way (honest micro-finding).

4. **Sample robustness** (analysis 4)
   - Pilot (first-50) vs stratified (n=180) LCB samples: pooled aggregates
     replicated (~31/34/46). Per-model rows vary (e.g. grok-reasoning 48->32);
     report honestly as sample sensitivity in Threats.

5. **Cross-seed variance on HumanEval** (analysis 5)
   - 3 seeds already run; SD per model/condition; appendix table.

6. **HumanEval+ re-scoring (local, no API)**
   - evalplus (in venv) re-scores stored seed-42 solutions against 80x tests.
   - Build samples jsonl per (model, cond) from extracted_code + prompt preamble;
     run evalplus evaluate; report solution-robustness subsection.

## Track 2 — Prose expansions

7. **Related Work**: add ~10 real citations
   - Kojima et al. 2022 (zero-shot CoT); Schulhoff et al. 2024 (The Prompt Report);
     Hou et al. 2024 TOSEM (LLMs for SE survey); Chang et al. (LLM evaluation survey);
     Beck 2003 (TDD book); George & Williams 2004 (TDD empirical);
     Solis & Wang 2011 (BDD characteristics); Sainz et al. 2023 +
     Jacovi et al. 2023 (benchmark contamination); Liu et al. 2023 (prompting survey).
8. **Discussion subsection**: implications for benchmark/evaluation design
   (saturation, per-difficulty reporting, raw-output logging as standard).
9. **Replication-package description** (~0.3 pp): harness, raw logs + telemetry,
   manifest, make_paper_figures.py, make_paper_analyses.py.

## Track 3 — Appendices

10. Per-model x per-difficulty LCB table; per-seed HumanEval table; full prompt
    templates; 1-2 more qualitative code examples (40 logged candidates).

## Execution order
Track 1 (1 -> 3 -> 2 -> 4 -> 5, then 6 in background) -> Track 2 -> Track 3
-> recompile -> anti-contradiction consistency scan -> commit + push both repos.

## Honesty rules
- Analyses 3 and 4 are reported however they come out.
- Every new number in the text must come from the committed scripts' output.
