# BDD vs No-BDD Ablation Study - Experiment Summary

**Date:** March 30, 2026
**Mode:** local_file_completion
**Dataset:** DevEval (LM_prompt_elements.jsonl)

## 1. GPT-4.1 Single-Model Ablation (50 samples)

| Metric | WITH BDD | WITHOUT BDD | Difference |
|--------|----------|-------------|------------|
| Sample Size | 50 | 50 | - |
| API Success Rate | 100.0% | 100.0% | 0.0% |
| 95% CI | [93.2%, 100%] | [93.2%, 100%] | - |
| Avg Duration | 5.60s | 2.31s | +3.29s |
| Avg Tokens | 6,462 | 3,031 | +3,431 |

**Key Finding:** BDD methodology requires approximately 2.4x more time and 2.1x more tokens than direct prompting.

## 2. Multi-Model Comparison (20-25 samples each)

### GPT-4.1

| Metric | BDD | Direct | Difference |
|--------|-----|--------|------------|
| Samples | 23 | 23 | - |
| API Success | 100.0% | 100.0% | 0.0% |
| 95% CI | [85.7%, 100%] | [85.7%, 100%] | - |
| Avg Duration | 6.14s | 2.58s | +3.56s |
| Avg Tokens | 9,161 | 4,396 | +4,765 |

### Gemini 2.5 Flash

| Metric | BDD | Direct | Difference |
|--------|-----|--------|------------|
| Samples | 25 | 25 | - |
| API Success | 100.0% | 100.0% | 0.0% |
| 95% CI | [86.7%, 100%] | [86.7%, 100%] | - |
| Avg Duration | 7.72s | 4.11s | +3.61s |
| Avg Tokens | 11,201 | 5,633 | +5,568 |

## 3. Effect Size Analysis

| Comparison | Duration Cohen's d | Interpretation |
|------------|-------------------|----------------|
| GPT-4.1 BDD vs Direct | 1.918 | Large effect |
| Gemini 2.5 Flash BDD vs Direct | 3.024 | Large effect |

The large effect sizes indicate a significant and meaningful difference in processing time between BDD and direct prompting approaches.

## 4. LaTeX Table for Publication

```latex
\begin{table}[htbp]
\centering
\caption{BDD vs Direct Prompting Ablation Study}
\label{tab:ablation}
\begin{tabular}{llcccc}
\toprule
Model & Method & n & API Success (95\% CI) & Time (s) & Tokens \\
\midrule
GPT-4.1 & BDD & 50 & 100\% [93.2\%, 100\%] & 5.60 & 6,462 \\
        & Direct & 50 & 100\% [93.2\%, 100\%] & 2.31 & 3,031 \\
\midrule
Gemini 2.5 Flash & BDD & 25 & 100\% [86.7\%, 100\%] & 7.72 & 11,201 \\
                 & Direct & 25 & 100\% [86.7\%, 100\%] & 4.11 & 5,633 \\
\bottomrule
\end{tabular}
\end{table}
```

## 5. Important Limitations

1. **API Success vs Pass@1:** These results show API success rates (successful code generation), NOT Pass@1 (functional correctness). Pass@1 requires running the generated code through the DevEval evaluation harness.

2. **Sample Size:** While we have 50 samples for GPT-4.1 and 25 for Gemini, larger sample sizes (100+) would provide tighter confidence intervals.

3. **Single Context Mode:** Results are for `local_file_completion` mode only. Additional experiments needed for:
   - `without_context`
   - `local_file_infiling`

## 6. Next Steps for Publication

1. **Run DevEval Evaluation:** Execute generated code against test cases to get Pass@1 metrics
2. **Expand Sample Size:** Run experiments with 100+ samples for statistical power
3. **Additional Context Modes:** Test all three context modes
4. **Code Quality Analysis:** Implement static analysis for cyclomatic complexity, code coverage potential

## 7. Reproducibility

- **Random Seeds Used:** 42, 100, 500, 999
- **Models:** GPT-4.1 (Azure OpenAI), Gemini 2.5 Flash (Google)
- **API Versions:** Azure OpenAI v1, Gemini v1beta
- **Temperature:** 0.2 (for reproducibility)

## File Locations

- GPT-4.1 ablation: `results/ablation/`
- Multi-model comparison: `results/multimodel_ablation/`
- Scripts: `run_ablation.py`, `run_multimodel_ablation.py`, `analyze_ablation_results.py`
