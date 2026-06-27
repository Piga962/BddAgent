#!/usr/bin/env python3
"""
Generate LaTeX Tables with Wilson Confidence Intervals

Reads experiment results from all benchmarks (HumanEval, LiveCodeBench,
ClassEval, HumanEval+) and generates publication-ready LaTeX tables
with 95% Wilson score confidence intervals.

Usage:
    python analysis/generate_ci_tables.py

Output:
    - results/analysis_output/latex_tables/table_humaneval.tex
    - results/analysis_output/latex_tables/table_livecodebench.tex
    - results/analysis_output/latex_tables/table_classeval.tex
    - results/analysis_output/latex_tables/table_humaneval_plus.tex
    - results/analysis_output/latex_tables/all_tables.tex (combined)
"""

import json
import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict


RESULTS_DIR = Path(__file__).parent.parent / "results"
OUTPUT_DIR = RESULTS_DIR / "analysis_output" / "latex_tables"


@dataclass
class BenchmarkResult:
    """Result for a single model/condition combination."""
    model: str
    condition: str
    passed: int
    total: int
    pass_rate: float
    ci_lower: float
    ci_upper: float


def wilson_score_interval(passed: int, total: int, z: float = 1.96) -> Tuple[float, float]:
    """
    Compute Wilson score confidence interval for a proportion.

    Args:
        passed: Number of successes
        total: Total number of trials
        z: Z-score for confidence level (1.96 for 95% CI)

    Returns:
        Tuple of (ci_lower, ci_upper)
    """
    if total == 0:
        return 0.0, 0.0

    p = passed / total
    denominator = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denominator
    spread = z * math.sqrt((p * (1 - p) + z**2 / (4 * total)) / total) / denominator

    return max(0.0, center - spread), min(1.0, center + spread)


def load_humaneval_results() -> List[BenchmarkResult]:
    """Load HumanEval results from summary files."""
    results = []
    summary_dir = RESULTS_DIR / "bdd_vs_cot"

    # Find the best (most recent or largest n) summary for each model
    model_summaries = defaultdict(list)

    for summary_file in summary_dir.glob("summary_*.json"):
        try:
            with open(summary_file) as f:
                data = json.load(f)
            model = data.get("model", "unknown")
            model_summaries[model].append((summary_file, data))
        except (json.JSONDecodeError, KeyError):
            continue

    # For each model, use the summary with the most problems
    for model, summaries in model_summaries.items():
        # Sort by n_problems descending, then by timestamp
        summaries.sort(key=lambda x: (x[1].get("n_problems", 0), x[0].name), reverse=True)
        best_summary = summaries[0][1]

        n_problems = best_summary.get("n_problems", 164)

        for condition in ["bdd", "cot", "direct"]:
            cond_data = best_summary.get("conditions", {}).get(condition, {})

            # Handle different data formats
            if "passed" in cond_data:
                passed = cond_data["passed"]
                pass_rate = cond_data.get("pass_rate", passed / n_problems)
            else:
                pass_rate = cond_data.get("pass_rate", 0)
                passed = int(round(pass_rate * n_problems))

            ci_lower, ci_upper = wilson_score_interval(passed, n_problems)

            results.append(BenchmarkResult(
                model=model,
                condition=condition,
                passed=passed,
                total=n_problems,
                pass_rate=pass_rate,
                ci_lower=ci_lower,
                ci_upper=ci_upper
            ))

    return results


def load_livecodebench_results() -> List[BenchmarkResult]:
    """Load LiveCodeBench results from summary files."""
    results = []
    summary_dir = RESULTS_DIR / "livecodebench"

    model_summaries = defaultdict(list)

    for summary_file in summary_dir.glob("summary_*.json"):
        try:
            with open(summary_file) as f:
                data = json.load(f)
            model = data.get("model", "unknown")
            model_summaries[model].append((summary_file, data))
        except (json.JSONDecodeError, KeyError):
            continue

    for model, summaries in model_summaries.items():
        summaries.sort(key=lambda x: (x[1].get("n_problems", 0), x[0].name), reverse=True)
        best_summary = summaries[0][1]

        n_problems = best_summary.get("n_problems", 50)

        for condition in ["bdd", "cot", "direct"]:
            cond_data = best_summary.get("conditions", {}).get(condition, {})

            passed = cond_data.get("passed", 0)
            total = cond_data.get("total", n_problems)
            pass_rate = cond_data.get("pass_rate", passed / total if total > 0 else 0)

            ci_lower, ci_upper = wilson_score_interval(passed, total)

            results.append(BenchmarkResult(
                model=model,
                condition=condition,
                passed=passed,
                total=total,
                pass_rate=pass_rate,
                ci_lower=ci_lower,
                ci_upper=ci_upper
            ))

    return results


def load_classeval_results() -> List[BenchmarkResult]:
    """Load ClassEval results from summary files."""
    results = []
    summary_dir = RESULTS_DIR / "classeval"

    model_summaries = defaultdict(list)

    for summary_file in summary_dir.glob("summary_*.json"):
        try:
            with open(summary_file) as f:
                data = json.load(f)
            model = data.get("model", "unknown")
            model_summaries[model].append((summary_file, data))
        except (json.JSONDecodeError, KeyError):
            continue

    for model, summaries in model_summaries.items():
        summaries.sort(key=lambda x: (x[1].get("n_problems", 0), x[0].name), reverse=True)
        best_summary = summaries[0][1]

        n_problems = best_summary.get("n_problems", 100)

        for condition in ["bdd", "cot", "direct"]:
            cond_data = best_summary.get("conditions", {}).get(condition, {})

            passed = cond_data.get("passed", 0)
            total = cond_data.get("total", n_problems)
            pass_rate = cond_data.get("pass_rate", passed / total if total > 0 else 0)

            ci_lower, ci_upper = wilson_score_interval(passed, total)

            results.append(BenchmarkResult(
                model=model,
                condition=condition,
                passed=passed,
                total=total,
                pass_rate=pass_rate,
                ci_lower=ci_lower,
                ci_upper=ci_upper
            ))

    return results


def load_humaneval_plus_results() -> List[BenchmarkResult]:
    """Load HumanEval+ results from summary file."""
    results = []
    summary_file = RESULTS_DIR / "humaneval_plus" / "humaneval_plus_summary_v2.json"

    if not summary_file.exists():
        summary_file = RESULTS_DIR / "humaneval_plus" / "humaneval_plus_summary.json"

    if not summary_file.exists():
        return results

    with open(summary_file) as f:
        data = json.load(f)

    for model, model_data in data.items():
        for condition in ["bdd", "cot", "direct"]:
            cond_data = model_data.get(condition, {})

            # Use plus_passed for HumanEval+ rate
            passed = cond_data.get("plus_passed", 0)
            total = cond_data.get("total", 164)
            pass_rate = cond_data.get("plus_rate", passed / total if total > 0 else 0)

            ci_lower, ci_upper = wilson_score_interval(passed, total)

            results.append(BenchmarkResult(
                model=model,
                condition=condition,
                passed=passed,
                total=total,
                pass_rate=pass_rate,
                ci_lower=ci_lower,
                ci_upper=ci_upper
            ))

    return results


def format_rate_with_ci(result: BenchmarkResult, bold_best: bool = False) -> str:
    """Format pass rate with confidence interval for LaTeX."""
    rate_pct = result.pass_rate * 100
    ci_low_pct = result.ci_lower * 100
    ci_high_pct = result.ci_upper * 100

    if bold_best:
        return f"\\textbf{{{rate_pct:.1f}}} [{ci_low_pct:.1f}, {ci_high_pct:.1f}]"
    else:
        return f"{rate_pct:.1f} [{ci_low_pct:.1f}, {ci_high_pct:.1f}]"


def generate_latex_table(
    results: List[BenchmarkResult],
    benchmark_name: str,
    caption: str,
    label: str
) -> str:
    """Generate LaTeX table from results."""

    # Organize by model
    by_model: Dict[str, Dict[str, BenchmarkResult]] = defaultdict(dict)
    for r in results:
        by_model[r.model][r.condition] = r

    # Sort models alphabetically (or by capability if you prefer)
    models = sorted(by_model.keys())

    # Filter out models with 0% across all conditions or too few samples
    valid_models = []
    for model in models:
        model_results = by_model[model]
        # Check if any condition has non-zero results and reasonable sample size
        has_valid = any(
            model_results.get(c, BenchmarkResult(model, c, 0, 0, 0, 0, 0)).pass_rate > 0
            or model_results.get(c, BenchmarkResult(model, c, 0, 0, 0, 0, 0)).total >= 50
            for c in ["bdd", "cot", "direct"]
        )
        if has_valid:
            valid_models.append(model)

    models = valid_models

    # Generate table
    lines = []
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append("\\caption{" + caption + "}")
    lines.append("\\label{" + label + "}")
    lines.append("\\begin{tabular}{lccc}")
    lines.append("\\toprule")
    lines.append("Model & BDD & CoT & Direct \\\\")
    lines.append("\\midrule")

    for model in models:
        model_results = by_model[model]

        # Get results for each condition
        bdd = model_results.get("bdd")
        cot = model_results.get("cot")
        direct = model_results.get("direct")

        # Determine which is best
        rates = []
        if bdd: rates.append(("bdd", bdd.pass_rate))
        if cot: rates.append(("cot", cot.pass_rate))
        if direct: rates.append(("direct", direct.pass_rate))

        best_condition = max(rates, key=lambda x: x[1])[0] if rates else None

        # Format cells
        def fmt(r, condition):
            if r is None:
                return "---"
            is_best = (condition == best_condition) and r.pass_rate > 0
            return format_rate_with_ci(r, bold_best=is_best)

        # Clean model name for display
        display_name = model.replace("_", "\\_")

        # Add sample size if not standard
        sample_note = ""
        sample_size = bdd.total if bdd else (cot.total if cot else (direct.total if direct else 0))
        if benchmark_name == "HumanEval" and sample_size != 164 and sample_size > 0:
            sample_note = f" (n={sample_size})"
        elif benchmark_name == "LiveCodeBench" and sample_size != 50 and sample_size > 0:
            sample_note = f" (n={sample_size})"
        elif benchmark_name == "ClassEval" and sample_size != 100 and sample_size > 0:
            sample_note = f" (n={sample_size})"

        lines.append(f"{display_name}{sample_note} & {fmt(bdd, 'bdd')} & {fmt(cot, 'cot')} & {fmt(direct, 'direct')} \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    return "\n".join(lines)


def generate_summary_stats_table(
    he_results: List[BenchmarkResult],
    lcb_results: List[BenchmarkResult],
    ce_results: List[BenchmarkResult]
) -> str:
    """Generate summary statistics table across all benchmarks."""

    def calc_stats(results: List[BenchmarkResult]) -> Dict[str, Dict[str, float]]:
        """Calculate mean/std for each condition."""
        by_condition = defaultdict(list)
        for r in results:
            if r.total >= 50:  # Only include sufficient sample sizes
                by_condition[r.condition].append(r.pass_rate)

        stats = {}
        for cond, rates in by_condition.items():
            if rates:
                mean_rate = sum(rates) / len(rates)
                if len(rates) > 1:
                    std_rate = math.sqrt(sum((r - mean_rate)**2 for r in rates) / (len(rates) - 1))
                else:
                    std_rate = 0
                stats[cond] = {"mean": mean_rate, "std": std_rate, "n": len(rates)}

        return stats

    he_stats = calc_stats(he_results)
    lcb_stats = calc_stats(lcb_results)
    ce_stats = calc_stats(ce_results)

    lines = []
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append("\\caption{Summary Statistics Across Benchmarks (Mean $\\pm$ Std)}")
    lines.append("\\label{tab:summary_stats}")
    lines.append("\\begin{tabular}{llccc}")
    lines.append("\\toprule")
    lines.append("Benchmark & N & BDD & CoT & Direct \\\\")
    lines.append("\\midrule")

    for name, stats in [("HumanEval", he_stats), ("LiveCodeBench", lcb_stats), ("ClassEval", ce_stats)]:
        n_models = stats.get("bdd", {}).get("n", 0)

        def fmt_stat(cond):
            s = stats.get(cond, {})
            if not s:
                return "---"
            return f"{s['mean']*100:.1f} $\\pm$ {s['std']*100:.1f}"

        lines.append(f"{name} & {n_models} & {fmt_stat('bdd')} & {fmt_stat('cot')} & {fmt_stat('direct')} \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    return "\n".join(lines)


def main():
    print("Generating LaTeX Tables with Wilson CIs")
    print("=" * 50)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load all results
    print("\nLoading HumanEval results...")
    he_results = load_humaneval_results()
    print(f"  Found {len(he_results)} results for {len(set(r.model for r in he_results))} models")

    print("Loading LiveCodeBench results...")
    lcb_results = load_livecodebench_results()
    print(f"  Found {len(lcb_results)} results for {len(set(r.model for r in lcb_results))} models")

    print("Loading ClassEval results...")
    ce_results = load_classeval_results()
    print(f"  Found {len(ce_results)} results for {len(set(r.model for r in ce_results))} models")

    print("Loading HumanEval+ results...")
    hep_results = load_humaneval_plus_results()
    print(f"  Found {len(hep_results)} results for {len(set(r.model for r in hep_results))} models")

    # Generate individual tables
    tables = []

    if he_results:
        he_table = generate_latex_table(
            he_results,
            "HumanEval",
            "HumanEval Pass Rates with 95\\% Wilson Confidence Intervals",
            "tab:humaneval_full"
        )
        tables.append(("HumanEval", he_table))
        with open(OUTPUT_DIR / "table_humaneval.tex", "w") as f:
            f.write(he_table)
        print(f"\n  Saved: table_humaneval.tex")

    if lcb_results:
        lcb_table = generate_latex_table(
            lcb_results,
            "LiveCodeBench",
            "LiveCodeBench Pass Rates with 95\\% Wilson Confidence Intervals (n=50 subset)",
            "tab:livecodebench_full"
        )
        tables.append(("LiveCodeBench", lcb_table))
        with open(OUTPUT_DIR / "table_livecodebench.tex", "w") as f:
            f.write(lcb_table)
        print(f"  Saved: table_livecodebench.tex")

    if ce_results:
        ce_table = generate_latex_table(
            ce_results,
            "ClassEval",
            "ClassEval Pass Rates with 95\\% Wilson Confidence Intervals",
            "tab:classeval_full"
        )
        tables.append(("ClassEval", ce_table))
        with open(OUTPUT_DIR / "table_classeval.tex", "w") as f:
            f.write(ce_table)
        print(f"  Saved: table_classeval.tex")

    if hep_results:
        hep_table = generate_latex_table(
            hep_results,
            "HumanEval+",
            "HumanEval+ Pass Rates with 95\\% Wilson Confidence Intervals",
            "tab:humaneval_plus_full"
        )
        tables.append(("HumanEval+", hep_table))
        with open(OUTPUT_DIR / "table_humaneval_plus.tex", "w") as f:
            f.write(hep_table)
        print(f"  Saved: table_humaneval_plus.tex")

    # Generate summary stats table
    if he_results and lcb_results and ce_results:
        summary_table = generate_summary_stats_table(he_results, lcb_results, ce_results)
        tables.append(("Summary", summary_table))
        with open(OUTPUT_DIR / "table_summary_stats.tex", "w") as f:
            f.write(summary_table)
        print(f"  Saved: table_summary_stats.tex")

    # Generate combined file
    with open(OUTPUT_DIR / "all_tables.tex", "w") as f:
        f.write("% Auto-generated LaTeX tables with Wilson 95% CIs\n")
        f.write(f"% Generated by generate_ci_tables.py\n\n")
        for name, table in tables:
            f.write(f"% === {name} ===\n")
            f.write(table)
            f.write("\n\n")
    print(f"  Saved: all_tables.tex (combined)")

    print(f"\nAll tables saved to: {OUTPUT_DIR}")

    # Print quick summary
    print("\n" + "=" * 50)
    print("QUICK SUMMARY")
    print("=" * 50)

    for name, results in [("HumanEval", he_results), ("LiveCodeBench", lcb_results),
                          ("ClassEval", ce_results), ("HumanEval+", hep_results)]:
        if not results:
            continue

        print(f"\n{name}:")

        # Calculate BDD advantage
        by_model = defaultdict(dict)
        for r in results:
            by_model[r.model][r.condition] = r

        bdd_wins = 0
        cot_wins = 0
        ties = 0

        for model, conds in by_model.items():
            bdd = conds.get("bdd")
            cot = conds.get("cot")
            if bdd and cot:
                if bdd.pass_rate > cot.pass_rate + 0.01:  # 1% margin
                    bdd_wins += 1
                elif cot.pass_rate > bdd.pass_rate + 0.01:
                    cot_wins += 1
                else:
                    ties += 1

        total = bdd_wins + cot_wins + ties
        if total > 0:
            print(f"  BDD > CoT: {bdd_wins}/{total} ({bdd_wins/total*100:.0f}%)")
            print(f"  CoT > BDD: {cot_wins}/{total} ({cot_wins/total*100:.0f}%)")
            print(f"  Ties (±1%): {ties}/{total}")


if __name__ == "__main__":
    main()
