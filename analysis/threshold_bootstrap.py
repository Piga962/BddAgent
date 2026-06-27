#!/usr/bin/env python3
"""
Bootstrap Analysis for Model Capability Threshold

Validates the 75% threshold claim using bootstrap confidence intervals.
Tests whether the threshold is statistically meaningful or cherry-picked.

Usage:
    python analysis/threshold_bootstrap.py
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict
from scipy import stats
import matplotlib.pyplot as plt


# HumanEval results from the experiments (Direct prompting = baseline capability)
# Format: (model_name, direct_pass_rate, bdd_pass_rate, cot_pass_rate)
HUMANEVAL_RESULTS = [
    ("GPT-4o", 0.860, 0.817, 0.890),
    ("GPT-5.3-codex", 0.817, 0.866, 0.902),
    ("Llama-3.3-70B", 0.768, 0.768, 0.805),
    ("GPT-4.1", 0.707, 0.689, 0.665),
    ("gpt-35-turbo", 0.701, 0.866, 0.726),
    ("Gemini-3.1-Flash-Preview", 0.280, 0.293, 0.262),
    ("Claude-Haiku-4.5", 0.195, 0.189, 0.238),
    ("Claude-Sonnet-4", 0.067, 0.146, 0.159),
    ("Gemini-2.5-Flash", 0.165, 0.232, 0.104),
    ("codex-mini", 0.024, 0.146, 0.122),
]


def compute_bdd_advantage(results: List[Tuple]) -> np.ndarray:
    """Compute BDD - CoT difference for each model."""
    return np.array([r[2] - r[3] for r in results])  # BDD - CoT


def compute_correlation(results: List[Tuple]) -> float:
    """Compute correlation between baseline capability and BDD advantage."""
    baselines = np.array([r[1] for r in results])
    advantages = compute_bdd_advantage(results)
    return np.corrcoef(baselines, advantages)[0, 1]


def find_optimal_threshold(results: List[Tuple]) -> Tuple[float, float]:
    """Find threshold that best separates CoT-winning from BDD-winning models.

    Returns: (threshold, accuracy)
    """
    baselines = np.array([r[1] for r in results])
    advantages = compute_bdd_advantage(results)

    # CoT wins when advantage < 0 (i.e., CoT > BDD)
    cot_wins = advantages < 0

    best_threshold = 0
    best_accuracy = 0

    # Try all possible thresholds
    for threshold in np.linspace(0.05, 0.95, 100):
        # Predict: above threshold -> CoT wins, below -> BDD wins
        predicted_cot_wins = baselines > threshold
        accuracy = np.mean(predicted_cot_wins == cot_wins)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = threshold

    return best_threshold, best_accuracy


def bootstrap_threshold(results: List[Tuple], n_bootstrap: int = 10000) -> Dict:
    """Bootstrap confidence interval for the optimal threshold."""
    n = len(results)
    thresholds = []
    accuracies = []
    correlations = []

    for _ in range(n_bootstrap):
        # Resample with replacement
        indices = np.random.choice(n, n, replace=True)
        sample = [results[i] for i in indices]

        # Compute statistics on sample
        threshold, accuracy = find_optimal_threshold(sample)
        correlation = compute_correlation(sample)

        thresholds.append(threshold)
        accuracies.append(accuracy)
        correlations.append(correlation)

    thresholds = np.array(thresholds)
    accuracies = np.array(accuracies)
    correlations = np.array(correlations)

    return {
        'threshold_mean': np.mean(thresholds),
        'threshold_ci_lower': np.percentile(thresholds, 2.5),
        'threshold_ci_upper': np.percentile(thresholds, 97.5),
        'threshold_std': np.std(thresholds),
        'accuracy_mean': np.mean(accuracies),
        'accuracy_ci_lower': np.percentile(accuracies, 2.5),
        'accuracy_ci_upper': np.percentile(accuracies, 97.5),
        'correlation_mean': np.mean(correlations),
        'correlation_ci_lower': np.percentile(correlations, 2.5),
        'correlation_ci_upper': np.percentile(correlations, 97.5),
        'correlation_p_value': 2 * min(
            np.mean(correlations > 0),
            np.mean(correlations < 0)
        ),  # Two-tailed p-value
        'thresholds': thresholds,
        'correlations': correlations,
    }


def test_specific_threshold(results: List[Tuple], threshold: float) -> Dict:
    """Test a specific threshold value."""
    baselines = np.array([r[1] for r in results])
    advantages = compute_bdd_advantage(results)
    cot_wins = advantages < 0

    predicted_cot_wins = baselines > threshold
    accuracy = np.mean(predicted_cot_wins == cot_wins)

    # Which models are correctly/incorrectly classified?
    correct = []
    incorrect = []
    for i, (name, baseline, bdd, cot) in enumerate(results):
        pred = "CoT" if baseline > threshold else "BDD"
        actual = "CoT" if cot > bdd else "BDD"
        if pred == actual:
            correct.append(name)
        else:
            incorrect.append(f"{name} (pred={pred}, actual={actual})")

    return {
        'threshold': threshold,
        'accuracy': accuracy,
        'n_correct': len(correct),
        'n_incorrect': len(incorrect),
        'correct_models': correct,
        'incorrect_models': incorrect,
    }


def generate_visualizations(bootstrap_results: Dict, output_dir: Path):
    """Generate visualization figures."""
    output_dir.mkdir(exist_ok=True)

    # Figure 1: Threshold distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Threshold histogram
    axes[0].hist(bootstrap_results['thresholds'], bins=50, edgecolor='black', alpha=0.7)
    axes[0].axvline(0.75, color='red', linestyle='--', linewidth=2, label='Paper claim (75%)')
    axes[0].axvline(bootstrap_results['threshold_mean'], color='blue', linestyle='-', linewidth=2, label=f'Bootstrap mean ({bootstrap_results["threshold_mean"]:.1%})')
    axes[0].axvline(bootstrap_results['threshold_ci_lower'], color='blue', linestyle=':', linewidth=1)
    axes[0].axvline(bootstrap_results['threshold_ci_upper'], color='blue', linestyle=':', linewidth=1)
    axes[0].set_xlabel('Optimal Threshold')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Bootstrap Distribution of Optimal Threshold')
    axes[0].legend()

    # Correlation histogram
    axes[1].hist(bootstrap_results['correlations'], bins=50, edgecolor='black', alpha=0.7)
    axes[1].axvline(0, color='gray', linestyle='-', linewidth=1)
    axes[1].axvline(0.35, color='red', linestyle='--', linewidth=2, label='Paper claim (r=0.35)')
    axes[1].axvline(bootstrap_results['correlation_mean'], color='blue', linestyle='-', linewidth=2, label=f'Bootstrap mean (r={bootstrap_results["correlation_mean"]:.2f})')
    axes[1].set_xlabel('Correlation (r)')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Bootstrap Distribution of Correlation')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(output_dir / 'threshold_bootstrap.png', dpi=150)
    plt.savefig(output_dir / 'threshold_bootstrap.pdf')
    plt.close()

    # Figure 2: Scatter plot with threshold
    fig, ax = plt.subplots(figsize=(10, 7))

    baselines = [r[1] for r in HUMANEVAL_RESULTS]
    advantages = [r[2] - r[3] for r in HUMANEVAL_RESULTS]
    names = [r[0] for r in HUMANEVAL_RESULTS]

    colors = ['blue' if adv > 0 else 'red' for adv in advantages]
    ax.scatter(baselines, advantages, c=colors, s=100, alpha=0.7)

    for i, name in enumerate(names):
        ax.annotate(name, (baselines[i], advantages[i]), fontsize=8,
                   xytext=(5, 5), textcoords='offset points')

    # Add threshold line
    ax.axvline(0.75, color='purple', linestyle='--', linewidth=2, alpha=0.5,
               label=f'Threshold = 75% (paper)')
    ax.axvline(bootstrap_results['threshold_mean'], color='green', linestyle='-', linewidth=2, alpha=0.5,
               label=f'Bootstrap mean = {bootstrap_results["threshold_mean"]:.1%}')
    ax.axhline(0, color='gray', linestyle='-', linewidth=1)

    # Shade regions
    ax.fill_betweenx([-0.2, 0.2], 0, 0.75, alpha=0.1, color='blue', label='BDD region')
    ax.fill_betweenx([-0.2, 0.2], 0.75, 1.0, alpha=0.1, color='red', label='CoT region')

    ax.set_xlabel('Baseline Capability (Direct Pass@1)')
    ax.set_ylabel('BDD Advantage (BDD - CoT)')
    ax.set_title('Model Capability vs BDD Advantage')
    ax.legend(loc='upper left')
    ax.set_xlim(-0.05, 1.0)
    ax.set_ylim(-0.2, 0.2)

    plt.tight_layout()
    plt.savefig(output_dir / 'capability_vs_advantage.png', dpi=150)
    plt.savefig(output_dir / 'capability_vs_advantage.pdf')
    plt.close()


def main():
    print("=" * 60)
    print("Bootstrap Analysis for Model Capability Threshold")
    print("=" * 60)

    # Set seed for reproducibility
    np.random.seed(42)

    # Compute point estimates
    print("\n1. Point Estimates (n=10 models)")
    print("-" * 40)

    correlation = compute_correlation(HUMANEVAL_RESULTS)
    print(f"   Correlation (baseline vs BDD advantage): r = {correlation:.3f}")

    optimal_threshold, accuracy = find_optimal_threshold(HUMANEVAL_RESULTS)
    print(f"   Optimal threshold: {optimal_threshold:.1%}")
    print(f"   Classification accuracy at optimal: {accuracy:.1%}")

    # Test the paper's claimed threshold (75%)
    print("\n2. Testing Paper's Claimed Threshold (75%)")
    print("-" * 40)
    test_75 = test_specific_threshold(HUMANEVAL_RESULTS, 0.75)
    print(f"   Accuracy at 75%: {test_75['accuracy']:.1%}")
    print(f"   Correctly classified: {test_75['correct_models']}")
    print(f"   Incorrectly classified: {test_75['incorrect_models']}")

    # Bootstrap analysis
    print("\n3. Bootstrap Analysis (n=10,000 resamples)")
    print("-" * 40)
    bootstrap = bootstrap_threshold(HUMANEVAL_RESULTS, n_bootstrap=10000)

    print(f"   Threshold: {bootstrap['threshold_mean']:.1%} [{bootstrap['threshold_ci_lower']:.1%}, {bootstrap['threshold_ci_upper']:.1%}]")
    print(f"   Threshold std: {bootstrap['threshold_std']:.1%}")
    print(f"   Classification accuracy: {bootstrap['accuracy_mean']:.1%} [{bootstrap['accuracy_ci_lower']:.1%}, {bootstrap['accuracy_ci_upper']:.1%}]")
    print(f"   Correlation: r = {bootstrap['correlation_mean']:.3f} [{bootstrap['correlation_ci_lower']:.3f}, {bootstrap['correlation_ci_upper']:.3f}]")
    print(f"   Correlation p-value: {bootstrap['correlation_p_value']:.3f}")

    # Statistical significance
    print("\n4. Statistical Significance")
    print("-" * 40)
    if bootstrap['correlation_p_value'] < 0.05:
        print("   ✓ Correlation IS statistically significant (p < 0.05)")
    else:
        print("   ✗ Correlation is NOT statistically significant (p >= 0.05)")
        print("     The threshold should be interpreted as an observed pattern,")
        print("     not a validated predictive threshold.")

    # Check if 75% is within CI
    if bootstrap['threshold_ci_lower'] <= 0.75 <= bootstrap['threshold_ci_upper']:
        print(f"   ✓ Paper's 75% threshold is within 95% CI")
    else:
        print(f"   ✗ Paper's 75% threshold is OUTSIDE 95% CI")
        print(f"     Consider revising to: {bootstrap['threshold_mean']:.1%}")

    # Generate figures
    output_dir = Path(__file__).parent.parent / "results" / "analysis_output"
    generate_visualizations(bootstrap, output_dir)
    print(f"\n5. Figures saved to: {output_dir}")

    # Save results
    results_file = output_dir / 'threshold_bootstrap_results.json'
    with open(results_file, 'w') as f:
        save_results = {
            'point_estimates': {
                'correlation': correlation,
                'optimal_threshold': optimal_threshold,
                'accuracy_at_optimal': accuracy,
            },
            'test_75_percent': {
                'accuracy': test_75['accuracy'],
                'correct_models': test_75['correct_models'],
                'incorrect_models': test_75['incorrect_models'],
            },
            'bootstrap': {
                'n_bootstrap': 10000,
                'threshold_mean': bootstrap['threshold_mean'],
                'threshold_ci': [bootstrap['threshold_ci_lower'], bootstrap['threshold_ci_upper']],
                'threshold_std': bootstrap['threshold_std'],
                'correlation_mean': bootstrap['correlation_mean'],
                'correlation_ci': [bootstrap['correlation_ci_lower'], bootstrap['correlation_ci_upper']],
                'correlation_p_value': bootstrap['correlation_p_value'],
                'accuracy_mean': bootstrap['accuracy_mean'],
                'accuracy_ci': [bootstrap['accuracy_ci_lower'], bootstrap['accuracy_ci_upper']],
            },
            'recommendation': (
                "The 75% threshold is within the bootstrap 95% CI, but the correlation "
                "is not statistically significant. Present as an 'observed pattern' "
                "rather than a 'validated threshold'."
            ) if bootstrap['correlation_p_value'] >= 0.05 else (
                "The correlation is statistically significant. The threshold can be "
                "presented as a meaningful predictive boundary."
            )
        }
        json.dump(save_results, f, indent=2)

    print(f"   Results saved to: {results_file}")


if __name__ == "__main__":
    main()
