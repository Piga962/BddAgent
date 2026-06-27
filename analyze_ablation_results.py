#!/usr/bin/env python3
"""
Statistical Analysis for BDD vs No-BDD Ablation Study

Generates publication-ready statistics:
- 95% confidence intervals (Wilson score)
- Cohen's d effect sizes
- McNemar's test for paired comparisons
- LaTeX table output
"""

import json
import math
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Dict
from collections import defaultdict


@dataclass
class AblationMetrics:
    """Metrics for one condition (BDD or No-BDD)."""
    model: str
    with_bdd: bool
    n_total: int
    n_success: int
    success_rate: float
    ci_lower: float
    ci_upper: float
    avg_duration: float
    avg_tokens: int
    avg_code_length: int


def wilson_score_interval(successes: int, total: int, confidence: float = 0.95) -> Tuple[float, float]:
    """Calculate Wilson score confidence interval for a proportion."""
    if total == 0:
        return 0.0, 0.0

    z = 1.96  # 95% confidence
    p = successes / total
    n = total

    denominator = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denominator
    spread = z * math.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denominator

    return max(0, center - spread), min(1, center + spread)


def cohens_d(group1: List[float], group2: List[float]) -> Tuple[float, str]:
    """Calculate Cohen's d effect size."""
    if not group1 or not group2:
        return 0.0, "undefined"

    n1, n2 = len(group1), len(group2)
    mean1 = sum(group1) / n1
    mean2 = sum(group2) / n2

    var1 = sum((x - mean1) ** 2 for x in group1) / n1 if n1 > 0 else 0
    var2 = sum((x - mean2) ** 2 for x in group2) / n2 if n2 > 0 else 0

    pooled_std = math.sqrt((var1 + var2) / 2)

    if pooled_std == 0:
        return 0.0, "undefined"

    d = (mean1 - mean2) / pooled_std

    if abs(d) < 0.2:
        magnitude = "negligible"
    elif abs(d) < 0.5:
        magnitude = "small"
    elif abs(d) < 0.8:
        magnitude = "medium"
    else:
        magnitude = "large"

    return d, magnitude


def mcnemar_test(paired_results: List[Tuple[bool, bool]]) -> Tuple[float, float]:
    """
    McNemar's test for paired binary data.
    Returns chi-square statistic and p-value.
    """
    # Count discordant pairs
    b = sum(1 for bdd, no_bdd in paired_results if bdd and not no_bdd)  # BDD success, No-BDD fail
    c = sum(1 for bdd, no_bdd in paired_results if not bdd and no_bdd)  # BDD fail, No-BDD success

    if b + c == 0:
        return 0.0, 1.0

    # McNemar's chi-square with continuity correction
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)

    # Approximate p-value from chi-square distribution (1 df)
    # Using simple approximation for p-value
    if chi2 > 10.83:
        p_value = 0.001
    elif chi2 > 6.64:
        p_value = 0.01
    elif chi2 > 3.84:
        p_value = 0.05
    else:
        p_value = 0.5  # Not significant

    return chi2, p_value


def load_ablation_results(results_dir: str) -> Dict[str, Dict[str, List[dict]]]:
    """Load all ablation results from directory."""
    results_path = Path(results_dir)
    all_results = defaultdict(lambda: {"bdd": [], "no_bdd": []})

    for jsonl_file in results_path.glob("*.jsonl"):
        with open(jsonl_file) as f:
            for line in f:
                result = json.loads(line)
                model = result.get("model", "gpt-4.1")  # Default for older files

                if result.get("with_bdd", True):
                    all_results[model]["bdd"].append(result)
                else:
                    all_results[model]["no_bdd"].append(result)

    return dict(all_results)


def analyze_model(model: str, bdd_results: List[dict], no_bdd_results: List[dict]) -> Tuple[AblationMetrics, AblationMetrics]:
    """Analyze results for a single model."""
    # BDD metrics
    bdd_success = sum(1 for r in bdd_results if r.get("success", False))
    bdd_rate = bdd_success / len(bdd_results) if bdd_results else 0
    bdd_ci = wilson_score_interval(bdd_success, len(bdd_results))

    bdd_durations = [r.get("duration_seconds", 0) for r in bdd_results if r.get("success")]
    bdd_tokens = [r.get("tokens_used", 0) for r in bdd_results if r.get("success")]
    bdd_code_len = [len(r.get("generated_code", "")) for r in bdd_results if r.get("success")]

    bdd_metrics = AblationMetrics(
        model=model,
        with_bdd=True,
        n_total=len(bdd_results),
        n_success=bdd_success,
        success_rate=bdd_rate,
        ci_lower=bdd_ci[0],
        ci_upper=bdd_ci[1],
        avg_duration=sum(bdd_durations) / len(bdd_durations) if bdd_durations else 0,
        avg_tokens=int(sum(bdd_tokens) / len(bdd_tokens)) if bdd_tokens else 0,
        avg_code_length=int(sum(bdd_code_len) / len(bdd_code_len)) if bdd_code_len else 0
    )

    # No-BDD metrics
    no_bdd_success = sum(1 for r in no_bdd_results if r.get("success", False))
    no_bdd_rate = no_bdd_success / len(no_bdd_results) if no_bdd_results else 0
    no_bdd_ci = wilson_score_interval(no_bdd_success, len(no_bdd_results))

    no_bdd_durations = [r.get("duration_seconds", 0) for r in no_bdd_results if r.get("success")]
    no_bdd_tokens = [r.get("tokens_used", 0) for r in no_bdd_results if r.get("success")]
    no_bdd_code_len = [len(r.get("generated_code", "")) for r in no_bdd_results if r.get("success")]

    no_bdd_metrics = AblationMetrics(
        model=model,
        with_bdd=False,
        n_total=len(no_bdd_results),
        n_success=no_bdd_success,
        success_rate=no_bdd_rate,
        ci_lower=no_bdd_ci[0],
        ci_upper=no_bdd_ci[1],
        avg_duration=sum(no_bdd_durations) / len(no_bdd_durations) if no_bdd_durations else 0,
        avg_tokens=int(sum(no_bdd_tokens) / len(no_bdd_tokens)) if no_bdd_tokens else 0,
        avg_code_length=int(sum(no_bdd_code_len) / len(no_bdd_code_len)) if no_bdd_code_len else 0
    )

    return bdd_metrics, no_bdd_metrics


def generate_latex_table(all_metrics: List[Tuple[AblationMetrics, AblationMetrics]]) -> str:
    """Generate LaTeX table for publication."""
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Ablation Study: BDD vs Direct Prompting}",
        "\\label{tab:ablation}",
        "\\begin{tabular}{lcccccc}",
        "\\toprule",
        "Model & Method & n & API Success & 95\\% CI & Avg Time (s) & Avg Tokens \\\\",
        "\\midrule"
    ]

    for bdd_m, no_bdd_m in all_metrics:
        # BDD row
        bdd_ci = f"[{bdd_m.ci_lower:.1%}, {bdd_m.ci_upper:.1%}]"
        lines.append(
            f"{bdd_m.model} & BDD & {bdd_m.n_total} & {bdd_m.success_rate:.1%} & {bdd_ci} & "
            f"{bdd_m.avg_duration:.1f} & {bdd_m.avg_tokens} \\\\"
        )

        # No-BDD row
        no_bdd_ci = f"[{no_bdd_m.ci_lower:.1%}, {no_bdd_m.ci_upper:.1%}]"
        lines.append(
            f" & Direct & {no_bdd_m.n_total} & {no_bdd_m.success_rate:.1%} & {no_bdd_ci} & "
            f"{no_bdd_m.avg_duration:.1f} & {no_bdd_m.avg_tokens} \\\\"
        )
        lines.append("\\midrule")

    lines[-1] = "\\bottomrule"  # Replace last midrule
    lines.extend([
        "\\end{tabular}",
        "\\end{table}"
    ])

    return "\n".join(lines)


def generate_report(results_dir: str = "results/ablation") -> str:
    """Generate full statistical analysis report."""
    all_results = load_ablation_results(results_dir)

    if not all_results:
        # Try multimodel_ablation directory
        all_results = load_ablation_results("results/multimodel_ablation")

    if not all_results:
        return "No results found."

    report_lines = [
        "=" * 70,
        "STATISTICAL ANALYSIS: BDD vs No-BDD ABLATION STUDY",
        "=" * 70,
        ""
    ]

    all_metrics = []

    for model, results in sorted(all_results.items()):
        bdd_results = results["bdd"]
        no_bdd_results = results["no_bdd"]

        if not bdd_results and not no_bdd_results:
            continue

        bdd_m, no_bdd_m = analyze_model(model, bdd_results, no_bdd_results)
        all_metrics.append((bdd_m, no_bdd_m))

        report_lines.extend([
            f"\n{'=' * 50}",
            f"MODEL: {model}",
            f"{'=' * 50}",
            "",
            f"{'Metric':<25} {'BDD':<20} {'No-BDD':<20}",
            "-" * 65,
            f"{'Samples':<25} {bdd_m.n_total:<20} {no_bdd_m.n_total:<20}",
            f"{'API Success':<25} {bdd_m.n_success}/{bdd_m.n_total:<17} {no_bdd_m.n_success}/{no_bdd_m.n_total:<17}",
            f"{'Success Rate':<25} {bdd_m.success_rate:.1%}{'':<16} {no_bdd_m.success_rate:.1%}",
            f"{'95% CI':<25} [{bdd_m.ci_lower:.1%}, {bdd_m.ci_upper:.1%}]{'':<5} [{no_bdd_m.ci_lower:.1%}, {no_bdd_m.ci_upper:.1%}]",
            f"{'Avg Duration (s)':<25} {bdd_m.avg_duration:.2f}s{'':<15} {no_bdd_m.avg_duration:.2f}s",
            f"{'Avg Tokens':<25} {bdd_m.avg_tokens:<20} {no_bdd_m.avg_tokens:<20}",
            f"{'Avg Code Length':<25} {bdd_m.avg_code_length:<20} {no_bdd_m.avg_code_length:<20}",
        ])

        # Effect size for duration
        bdd_dur = [r.get("duration_seconds", 0) for r in bdd_results if r.get("success")]
        no_bdd_dur = [r.get("duration_seconds", 0) for r in no_bdd_results if r.get("success")]

        if bdd_dur and no_bdd_dur:
            d, magnitude = cohens_d(bdd_dur, no_bdd_dur)
            report_lines.append(f"\nDuration Effect Size: d = {d:.3f} ({magnitude})")

        # Success rate difference
        rate_diff = bdd_m.success_rate - no_bdd_m.success_rate
        report_lines.append(f"Success Rate Difference: {rate_diff:+.1%}")

    # Summary section
    report_lines.extend([
        "",
        "=" * 70,
        "SUMMARY",
        "=" * 70,
        ""
    ])

    # Average across models
    if all_metrics:
        avg_bdd_rate = sum(m[0].success_rate for m in all_metrics) / len(all_metrics)
        avg_no_bdd_rate = sum(m[1].success_rate for m in all_metrics) / len(all_metrics)

        report_lines.extend([
            f"Average BDD Success Rate: {avg_bdd_rate:.1%}",
            f"Average No-BDD Success Rate: {avg_no_bdd_rate:.1%}",
            f"Average Improvement: {(avg_bdd_rate - avg_no_bdd_rate):+.1%}",
            "",
            "Note: These are API success rates. Pass@1 requires running the",
            "DevEval evaluation harness on the generated code.",
        ])

    # LaTeX table
    if all_metrics:
        report_lines.extend([
            "",
            "=" * 70,
            "LATEX TABLE",
            "=" * 70,
            "",
            generate_latex_table(all_metrics)
        ])

    return "\n".join(report_lines)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze ablation study results")
    parser.add_argument("--dir", "-d", default="results/ablation", help="Results directory")
    parser.add_argument("--output", "-o", help="Output file (optional)")

    args = parser.parse_args()

    report = generate_report(args.dir)
    print(report)

    if args.output:
        with open(args.output, 'w') as f:
            f.write(report)
        print(f"\nReport saved to: {args.output}")
