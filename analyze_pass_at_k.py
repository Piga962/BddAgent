#!/usr/bin/env python3
"""
Analyze DevEval Pass@k evaluation results.

Parses test_output.jsonl files from DevEval evaluation and computes:
- Pass@1 rates with 95% Wilson confidence intervals
- Breakdown by status (Pass, Error, TimeOut, OOM)
- BDD vs No-BDD comparison with effect sizes
- LaTeX table for publication
"""

import json
import math
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional


@dataclass
class EvalResults:
    """Results from a single evaluation run."""
    condition: str
    total: int
    passed: int
    errors: int
    timeouts: int
    oom: int
    pass_rate: float
    ci_lower: float
    ci_upper: float


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


def cohens_h(p1: float, p2: float) -> Tuple[float, str]:
    """Calculate Cohen's h effect size for comparing proportions."""
    phi1 = 2 * math.asin(math.sqrt(p1))
    phi2 = 2 * math.asin(math.sqrt(p2))
    h = phi1 - phi2

    if abs(h) < 0.2:
        magnitude = "negligible"
    elif abs(h) < 0.5:
        magnitude = "small"
    elif abs(h) < 0.8:
        magnitude = "medium"
    else:
        magnitude = "large"

    return h, magnitude


def parse_test_output(log_file: str) -> Dict[str, int]:
    """Parse DevEval test_output.jsonl and count statuses."""
    counts = {"Pass": 0, "Error": 0, "TimeOut": 0, "OOM": 0}

    with open(log_file, 'r') as f:
        for line in f:
            try:
                result = json.loads(line.strip())
                status = result.get('status', 'Error')
                if status in counts:
                    counts[status] += 1
                else:
                    counts['Error'] += 1
            except json.JSONDecodeError:
                counts['Error'] += 1

    return counts


def analyze_condition(log_file: str, condition: str) -> Optional[EvalResults]:
    """Analyze results for a single condition."""
    if not Path(log_file).exists():
        return None

    counts = parse_test_output(log_file)
    total = sum(counts.values())

    if total == 0:
        return None

    passed = counts['Pass']
    pass_rate = passed / total
    ci_lower, ci_upper = wilson_score_interval(passed, total)

    return EvalResults(
        condition=condition,
        total=total,
        passed=passed,
        errors=counts['Error'],
        timeouts=counts['TimeOut'],
        oom=counts['OOM'],
        pass_rate=pass_rate,
        ci_lower=ci_lower,
        ci_upper=ci_upper
    )


def generate_comparison_report(bdd: EvalResults, no_bdd: EvalResults) -> str:
    """Generate comparison report text."""
    lines = [
        "=" * 70,
        "PASS@1 EVALUATION RESULTS: BDD vs No-BDD",
        "=" * 70,
        "",
        f"{'Metric':<25} {'BDD':<20} {'No-BDD':<20}",
        "-" * 65,
        f"{'Samples':<25} {bdd.total:<20} {no_bdd.total:<20}",
        f"{'Passed':<25} {bdd.passed:<20} {no_bdd.passed:<20}",
        f"{'Errors':<25} {bdd.errors:<20} {no_bdd.errors:<20}",
        f"{'Timeouts':<25} {bdd.timeouts:<20} {no_bdd.timeouts:<20}",
        f"{'OOM':<25} {bdd.oom:<20} {no_bdd.oom:<20}",
        "-" * 65,
        f"{'Pass@1 Rate':<25} {bdd.pass_rate:.1%}{'':<14} {no_bdd.pass_rate:.1%}",
        f"{'95% CI':<25} [{bdd.ci_lower:.1%}, {bdd.ci_upper:.1%}]{'':<4} [{no_bdd.ci_lower:.1%}, {no_bdd.ci_upper:.1%}]",
        "-" * 65,
    ]

    # Effect size
    diff = bdd.pass_rate - no_bdd.pass_rate
    h, magnitude = cohens_h(bdd.pass_rate, no_bdd.pass_rate)

    lines.extend([
        "",
        "STATISTICAL COMPARISON",
        "-" * 65,
        f"Pass@1 Difference: {diff:+.1%} (BDD - No-BDD)",
        f"Cohen's h: {h:.3f} ({magnitude} effect)",
        "",
    ])

    # Interpretation
    if diff > 0:
        lines.append(f"BDD shows {abs(diff):.1%} HIGHER Pass@1 than direct prompting")
    elif diff < 0:
        lines.append(f"BDD shows {abs(diff):.1%} LOWER Pass@1 than direct prompting")
    else:
        lines.append("BDD and direct prompting show EQUAL Pass@1 rates")

    return "\n".join(lines)


def generate_latex_table(bdd: EvalResults, no_bdd: EvalResults) -> str:
    """Generate LaTeX table for publication."""
    lines = [
        "",
        "LATEX TABLE",
        "=" * 70,
        "",
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Pass@1 Functional Correctness: BDD vs Direct Prompting}",
        "\\label{tab:pass1}",
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "Method & n & Pass@1 & 95\\% CI & Errors \\\\",
        "\\midrule",
        f"BDD & {bdd.total} & {bdd.pass_rate:.1%} & [{bdd.ci_lower:.1%}, {bdd.ci_upper:.1%}] & {bdd.errors} \\\\",
        f"Direct & {no_bdd.total} & {no_bdd.pass_rate:.1%} & [{no_bdd.ci_lower:.1%}, {no_bdd.ci_upper:.1%}] & {no_bdd.errors} \\\\",
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze DevEval Pass@k evaluation results"
    )
    parser.add_argument(
        "--results-dir",
        default="results/deveval_eval",
        help="Directory containing test_output.jsonl files"
    )
    parser.add_argument(
        "--output", "-o",
        help="Output file for report (default: print to stdout)"
    )

    args = parser.parse_args()

    results_dir = Path(args.results_dir)

    print("=" * 70)
    print("Analyzing DevEval Pass@k Results")
    print("=" * 70)
    print(f"Results directory: {results_dir}")

    # Analyze both conditions
    bdd_results = analyze_condition(
        str(results_dir / "bdd_test_output.jsonl"),
        "BDD"
    )
    no_bdd_results = analyze_condition(
        str(results_dir / "no_bdd_test_output.jsonl"),
        "No-BDD"
    )

    if not bdd_results and not no_bdd_results:
        print("\nNo evaluation results found!")
        print("Run run_deveval_eval.py first to generate results.")
        return

    # Single condition reports
    report_lines = []

    if bdd_results:
        report_lines.extend([
            "",
            f"BDD Results:",
            f"  Samples: {bdd_results.total}",
            f"  Pass@1: {bdd_results.pass_rate:.1%} [{bdd_results.ci_lower:.1%}, {bdd_results.ci_upper:.1%}]",
            f"  Errors: {bdd_results.errors}, Timeouts: {bdd_results.timeouts}, OOM: {bdd_results.oom}",
        ])

    if no_bdd_results:
        report_lines.extend([
            "",
            f"No-BDD Results:",
            f"  Samples: {no_bdd_results.total}",
            f"  Pass@1: {no_bdd_results.pass_rate:.1%} [{no_bdd_results.ci_lower:.1%}, {no_bdd_results.ci_upper:.1%}]",
            f"  Errors: {no_bdd_results.errors}, Timeouts: {no_bdd_results.timeouts}, OOM: {no_bdd_results.oom}",
        ])

    # Comparison report
    if bdd_results and no_bdd_results:
        comparison = generate_comparison_report(bdd_results, no_bdd_results)
        latex = generate_latex_table(bdd_results, no_bdd_results)
        report_lines.extend([
            "",
            comparison,
            latex,
        ])

    report = "\n".join(report_lines)
    print(report)

    if args.output:
        with open(args.output, 'w') as f:
            f.write(report)
        print(f"\nReport saved to: {args.output}")


if __name__ == "__main__":
    main()
