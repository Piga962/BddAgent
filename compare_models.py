#!/usr/bin/env python3
"""
Multi-Model Comparison Tool for BddAgent

Generates publication-ready comparison tables and statistical analysis
for multi-model experiments.

Usage:
    python compare_models.py --results-dir results/
    python compare_models.py --results-dir results/ --output comparison_report.md
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from analysis import (
    ExperimentAnalyzer,
    compute_confidence_interval,
    compute_effect_size,
    StatisticalSummary
)


@dataclass
class ModelResults:
    """Aggregated results for a single model."""
    model_name: str
    mode: str
    total_tests: int
    successful_tests: int
    success_rate: float
    success_rate_ci: tuple  # (lower, upper)
    syntax_valid_rate: float
    avg_duration: float
    duration_std: float
    avg_tokens: float
    total_cost: float
    avg_iterations: float


class ModelComparer:
    """
    Comprehensive model comparison and report generation.
    """

    def __init__(self, results_dir: str):
        self.results_dir = Path(results_dir)
        self.analyzer = ExperimentAnalyzer(results_dir)
        self.model_results: Dict[str, ModelResults] = {}

    def load_results(self):
        """Load all experiment results."""
        self.analyzer.load_results()

        for model_mode, results in self.analyzer.loaded_results.items():
            if not results:
                continue

            # Parse model name and mode from the key
            parts = model_mode.rsplit('_', 2)
            if len(parts) >= 2:
                model_name = '_'.join(parts[:-2]) if len(parts) > 2 else parts[0]
                mode = parts[-2] if len(parts) > 2 else parts[-1]
            else:
                model_name = model_mode
                mode = "unknown"

            # Compute statistics
            total = len(results)
            successful = sum(1 for r in results if r.get('success', False))
            syntax_valid = sum(
                1 for r in results
                if r.get('validation', {}).get('syntax_valid', False)
            )

            # Duration stats
            durations = [
                r.get('timing', {}).get('duration_seconds', 0)
                for r in results
            ]
            mean_dur, ci_low_dur, ci_high_dur = compute_confidence_interval(durations)

            # Token stats
            tokens = [
                r.get('tokens', {}).get('total_tokens', 0)
                for r in results
            ]

            # Cost stats
            costs = [
                r.get('tokens', {}).get('estimated_cost_usd', 0)
                for r in results
            ]

            # Iteration stats
            iterations = [r.get('iterations', 0) for r in results]

            # Success rate CI
            p = successful / total if total > 0 else 0
            z = 1.96
            if total > 0:
                denominator = 1 + z**2 / total
                center = (p + z**2 / (2 * total)) / denominator
                spread = z * ((p * (1 - p) + z**2 / (4 * total)) / total) ** 0.5 / denominator
                ci_low = max(0, center - spread)
                ci_high = min(1, center + spread)
            else:
                ci_low, ci_high = 0, 0

            self.model_results[model_mode] = ModelResults(
                model_name=model_name,
                mode=mode,
                total_tests=total,
                successful_tests=successful,
                success_rate=p,
                success_rate_ci=(ci_low, ci_high),
                syntax_valid_rate=syntax_valid / total if total > 0 else 0,
                avg_duration=mean_dur,
                duration_std=compute_confidence_interval(durations)[0] if durations else 0,
                avg_tokens=sum(tokens) / len(tokens) if tokens else 0,
                total_cost=sum(costs),
                avg_iterations=sum(iterations) / len(iterations) if iterations else 0
            )

    def generate_comparison_table(self) -> str:
        """Generate a markdown comparison table."""
        if not self.model_results:
            return "No results to compare."

        lines = [
            "# Model Comparison Results",
            "",
            "## Summary Table",
            "",
            "| Model | Mode | Tests | Success Rate (95% CI) | Syntax Valid | Avg Duration | Avg Tokens | Total Cost |",
            "|-------|------|-------|----------------------|--------------|--------------|------------|------------|"
        ]

        for key, r in sorted(self.model_results.items()):
            ci_str = f"{r.success_rate:.1%} [{r.success_rate_ci[0]:.1%}-{r.success_rate_ci[1]:.1%}]"
            lines.append(
                f"| {r.model_name} | {r.mode} | {r.total_tests} | {ci_str} | "
                f"{r.syntax_valid_rate:.1%} | {r.avg_duration:.1f}s | "
                f"{r.avg_tokens:.0f} | ${r.total_cost:.4f} |"
            )

        return "\n".join(lines)

    def generate_effect_size_analysis(self) -> str:
        """Generate effect size analysis between models."""
        if len(self.model_results) < 2:
            return "\n## Effect Size Analysis\n\nNeed at least 2 models for comparison.\n"

        lines = [
            "",
            "## Effect Size Analysis (Cohen's d)",
            "",
            "| Comparison | Success Rate Diff | Effect Size | Magnitude |",
            "|------------|-------------------|-------------|-----------|"
        ]

        keys = list(self.model_results.keys())
        for i, key_a in enumerate(keys):
            for key_b in keys[i + 1:]:
                results_a = self.analyzer.loaded_results.get(key_a, [])
                results_b = self.analyzer.loaded_results.get(key_b, [])

                if not results_a or not results_b:
                    continue

                success_a = [1 if r.get('success', False) else 0 for r in results_a]
                success_b = [1 if r.get('success', False) else 0 for r in results_b]

                d, magnitude = compute_effect_size(success_a, success_b)
                diff = sum(success_a) / len(success_a) - sum(success_b) / len(success_b)

                r_a = self.model_results[key_a]
                r_b = self.model_results[key_b]

                lines.append(
                    f"| {r_a.model_name} vs {r_b.model_name} | "
                    f"{diff:+.1%} | {d:.3f} | {magnitude} |"
                )

        return "\n".join(lines)

    def generate_latex_table(self) -> str:
        """Generate LaTeX table for publication."""
        if not self.model_results:
            return ""

        lines = [
            "",
            "## LaTeX Table (for publication)",
            "",
            "```latex",
            "\\begin{table}[htbp]",
            "\\centering",
            "\\caption{Model Comparison on DevEval Benchmark}",
            "\\label{tab:model-comparison}",
            "\\begin{tabular}{lcccc}",
            "\\toprule",
            "Model & Success Rate & 95\\% CI & Syntax Valid & Avg. Time (s) \\\\",
            "\\midrule"
        ]

        for key, r in sorted(self.model_results.items()):
            ci_str = f"[{r.success_rate_ci[0]:.1%}, {r.success_rate_ci[1]:.1%}]"
            lines.append(
                f"{r.model_name} & {r.success_rate:.1%} & {ci_str} & "
                f"{r.syntax_valid_rate:.1%} & {r.avg_duration:.1f} \\\\"
            )

        lines.extend([
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
            "```"
        ])

        return "\n".join(lines)

    def generate_full_report(self, output_path: Optional[str] = None) -> str:
        """Generate complete comparison report."""
        report_parts = [
            self.generate_comparison_table(),
            self.generate_effect_size_analysis(),
            "",
            "## Detailed Statistics",
            ""
        ]

        # Add detailed stats for each model
        for key, r in sorted(self.model_results.items()):
            report_parts.extend([
                f"### {r.model_name} ({r.mode})",
                f"- Total Tests: {r.total_tests}",
                f"- Successful: {r.successful_tests} ({r.success_rate:.1%})",
                f"- Syntax Valid: {r.syntax_valid_rate:.1%}",
                f"- Average Duration: {r.avg_duration:.2f}s",
                f"- Average Tokens: {r.avg_tokens:.0f}",
                f"- Total Cost: ${r.total_cost:.4f}",
                f"- Average Iterations: {r.avg_iterations:.1f}",
                ""
            ])

        report_parts.append(self.generate_latex_table())

        report = "\n".join(report_parts)

        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"Report saved to: {output_path}")

        return report


def main():
    parser = argparse.ArgumentParser(
        description="Generate multi-model comparison reports"
    )
    parser.add_argument(
        "--results-dir", "-r",
        type=str,
        default="results",
        help="Directory containing experiment results"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output path for the report (default: print to stdout)"
    )
    parser.add_argument(
        "--format",
        choices=["markdown", "json", "latex"],
        default="markdown",
        help="Output format (default: markdown)"
    )

    args = parser.parse_args()

    comparer = ModelComparer(args.results_dir)
    comparer.load_results()

    if not comparer.model_results:
        print(f"No results found in {args.results_dir}")
        return

    if args.format == "json":
        output = json.dumps(
            {k: v.__dict__ for k, v in comparer.model_results.items()},
            indent=2,
            default=str
        )
    elif args.format == "latex":
        output = comparer.generate_latex_table()
    else:
        output = comparer.generate_full_report(args.output)

    if not args.output:
        print(output)
    elif args.format != "markdown":  # markdown already saved in generate_full_report
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(output)
        print(f"Report saved to: {args.output}")


if __name__ == "__main__":
    main()
