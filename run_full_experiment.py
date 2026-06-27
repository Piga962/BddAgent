#!/usr/bin/env python3
"""
Full HumanEval Experiment: BDD vs No-BDD Code Generation

Runs multiple seeds for statistical robustness and aggregates results.
"""

import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime
import math


def wilson_ci(successes: int, total: int, confidence: float = 0.95) -> tuple:
    """Compute Wilson score confidence interval."""
    if total == 0:
        return 0.0, 0.0
    z = 1.96 if confidence == 0.95 else 2.576
    p = successes / total
    denominator = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denominator
    margin = z * math.sqrt((p * (1 - p) + z**2 / (4 * total)) / total) / denominator
    return max(0, center - margin), min(1, center + margin)


def cohens_d(n1: int, pass1: int, n2: int, pass2: int) -> float:
    """Compute Cohen's d effect size."""
    p1 = pass1 / n1 if n1 > 0 else 0
    p2 = pass2 / n2 if n2 > 0 else 0
    pooled_var = (p1 * (1 - p1) + p2 * (1 - p2)) / 2
    pooled_std = math.sqrt(pooled_var) if pooled_var > 0 else 1
    return (p1 - p2) / pooled_std if pooled_std > 0 else 0


def aggregate_results(results_dir: str = "results/humaneval_ablation"):
    """Aggregate results from multiple experiment runs."""
    results_path = Path(results_dir)

    # Find all summary files
    summary_files = list(results_path.glob("summary_*.json"))

    if not summary_files:
        print("No summary files found!")
        return None

    # Aggregate
    all_summaries = []
    for sf in summary_files:
        with open(sf) as f:
            all_summaries.append(json.load(f))

    # Compute aggregated stats
    total_bdd_passed = sum(s['bdd']['passed'] for s in all_summaries)
    total_no_bdd_passed = sum(s['no_bdd']['passed'] for s in all_summaries)
    total_problems = sum(s['n_problems'] for s in all_summaries)

    bdd_rate = total_bdd_passed / total_problems if total_problems > 0 else 0
    no_bdd_rate = total_no_bdd_passed / total_problems if total_problems > 0 else 0

    bdd_ci = wilson_ci(total_bdd_passed, total_problems)
    no_bdd_ci = wilson_ci(total_no_bdd_passed, total_problems)

    effect_size = cohens_d(total_problems, total_bdd_passed, total_problems, total_no_bdd_passed)

    # Per-seed breakdown
    per_seed = []
    for s in all_summaries:
        per_seed.append({
            "seed": s.get('seed', 'unknown'),
            "bdd_pass_rate": s['bdd']['pass_rate'],
            "no_bdd_pass_rate": s['no_bdd']['pass_rate'],
            "difference": s['difference']
        })

    aggregated = {
        "experiment": "HumanEval BDD vs No-BDD Ablation",
        "n_runs": len(all_summaries),
        "total_problems": total_problems,
        "aggregated": {
            "bdd": {
                "total_passed": total_bdd_passed,
                "pass_rate": bdd_rate,
                "ci_95_lower": bdd_ci[0],
                "ci_95_upper": bdd_ci[1]
            },
            "no_bdd": {
                "total_passed": total_no_bdd_passed,
                "pass_rate": no_bdd_rate,
                "ci_95_lower": no_bdd_ci[0],
                "ci_95_upper": no_bdd_ci[1]
            },
            "difference": bdd_rate - no_bdd_rate,
            "cohens_d": effect_size
        },
        "per_seed": per_seed,
        "timestamp": datetime.now().isoformat()
    }

    # Save aggregated results
    agg_file = results_path / "aggregated_results.json"
    with open(agg_file, 'w') as f:
        json.dump(aggregated, f, indent=2)

    # Print report
    print("\n" + "=" * 70)
    print("AGGREGATED HUMANEVAL RESULTS: BDD vs No-BDD")
    print("=" * 70)
    print(f"\nTotal runs: {len(all_summaries)}")
    print(f"Total problems evaluated: {total_problems}")

    print(f"\n{'Condition':<15} {'Pass Rate':<12} {'95% CI':<20} {'Passed':<10}")
    print("-" * 60)
    print(f"{'BDD':<15} {bdd_rate*100:>6.1f}%      [{bdd_ci[0]*100:.1f}%, {bdd_ci[1]*100:.1f}%]       {total_bdd_passed}/{total_problems}")
    print(f"{'No-BDD':<15} {no_bdd_rate*100:>6.1f}%      [{no_bdd_ci[0]*100:.1f}%, {no_bdd_ci[1]*100:.1f}%]       {total_no_bdd_passed}/{total_problems}")
    print("-" * 60)

    diff_pct = (bdd_rate - no_bdd_rate) * 100
    print(f"\nAbsolute Difference: {diff_pct:+.1f}%")

    if no_bdd_rate > 0:
        relative_improvement = ((bdd_rate - no_bdd_rate) / no_bdd_rate) * 100
        print(f"Relative Improvement: {relative_improvement:+.1f}%")

    print(f"Cohen's d: {effect_size:.3f}", end="")
    if abs(effect_size) < 0.2:
        print(" (negligible)")
    elif abs(effect_size) < 0.5:
        print(" (small)")
    elif abs(effect_size) < 0.8:
        print(" (medium)")
    else:
        print(" (large)")

    print("\nPer-Seed Breakdown:")
    print(f"{'Seed':<10} {'BDD':<12} {'No-BDD':<12} {'Diff':<10}")
    print("-" * 45)
    for ps in per_seed:
        print(f"{ps['seed']:<10} {ps['bdd_pass_rate']*100:>6.1f}%      {ps['no_bdd_pass_rate']*100:>6.1f}%      {ps['difference']*100:+.1f}%")

    print(f"\nResults saved to: {agg_file}")
    print("=" * 70)

    return aggregated


def run_experiment(seed: int, model: str = "gemini-2.5-flash", provider: str = "gemini"):
    """Run a single experiment with given seed."""
    print(f"\n{'='*70}")
    print(f"Running experiment: model={model}, seed={seed}")
    print(f"{'='*70}\n")

    cmd = [
        sys.executable, "run_humaneval_ablation.py",
        "--model", model,
        "--provider", provider,
        "--seed", str(seed)
    ]

    result = subprocess.run(cmd, capture_output=False)
    return result.returncode == 0


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run full HumanEval BDD experiment")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 100, 200],
                        help="Seeds to run (default: 42 100 200)")
    parser.add_argument("--model", default="gemini-2.5-flash")
    parser.add_argument("--provider", default="gemini")
    parser.add_argument("--aggregate-only", action="store_true",
                        help="Only aggregate existing results")

    args = parser.parse_args()

    if args.aggregate_only:
        aggregate_results()
        return

    # Run experiments with each seed
    print("=" * 70)
    print("FULL HUMANEVAL EXPERIMENT: BDD vs No-BDD")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Provider: {args.provider}")
    print(f"Seeds: {args.seeds}")
    print("=" * 70)

    success_count = 0
    for seed in args.seeds:
        if run_experiment(seed, args.model, args.provider):
            success_count += 1
        else:
            print(f"WARNING: Experiment with seed {seed} failed!")

    print(f"\nCompleted {success_count}/{len(args.seeds)} experiments")

    # Aggregate results
    if success_count > 0:
        aggregate_results()


if __name__ == "__main__":
    main()
