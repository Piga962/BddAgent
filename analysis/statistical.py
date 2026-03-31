"""
Statistical Analysis Utilities for BddAgent Experiments

Provides confidence intervals, significance testing, effect sizes,
and model comparison analysis.
"""

import json
import math
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path


@dataclass
class StatisticalSummary:
    """Statistical summary of a metric."""
    metric_name: str
    n: int
    mean: float
    std: float
    min_val: float
    max_val: float
    median: float
    ci_lower: float  # 95% CI lower bound
    ci_upper: float  # 95% CI upper bound

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ModelComparison:
    """Comparison between two models."""
    model_a: str
    model_b: str
    metric: str

    # Descriptive stats
    mean_a: float
    mean_b: float
    std_a: float
    std_b: float

    # Effect size
    cohens_d: float
    effect_magnitude: str  # "negligible", "small", "medium", "large"

    # Statistical test (simplified without scipy)
    mean_difference: float
    pooled_std: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def compute_mean(values: List[float]) -> float:
    """Compute mean of values."""
    if not values:
        return 0.0
    return sum(values) / len(values)


def compute_std(values: List[float], mean: Optional[float] = None) -> float:
    """Compute standard deviation of values."""
    if len(values) < 2:
        return 0.0
    if mean is None:
        mean = compute_mean(values)
    variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
    return math.sqrt(variance)


def compute_median(values: List[float]) -> float:
    """Compute median of values."""
    if not values:
        return 0.0
    sorted_values = sorted(values)
    n = len(sorted_values)
    mid = n // 2
    if n % 2 == 0:
        return (sorted_values[mid - 1] + sorted_values[mid]) / 2
    return sorted_values[mid]


def compute_confidence_interval(
    values: List[float],
    confidence: float = 0.95
) -> Tuple[float, float, float]:
    """
    Compute confidence interval for the mean.

    Uses t-distribution approximation for small samples.

    Args:
        values: List of values
        confidence: Confidence level (default 0.95)

    Returns:
        Tuple of (mean, ci_lower, ci_upper)
    """
    if len(values) < 2:
        mean = values[0] if values else 0.0
        return mean, mean, mean

    n = len(values)
    mean = compute_mean(values)
    std = compute_std(values, mean)
    se = std / math.sqrt(n)

    # t-critical value approximation for 95% CI
    # For n > 30, z ≈ 1.96; for smaller n, use larger values
    if n >= 30:
        t_critical = 1.96
    elif n >= 15:
        t_critical = 2.13
    elif n >= 10:
        t_critical = 2.26
    else:
        t_critical = 2.57  # Conservative for small samples

    margin = t_critical * se
    return mean, mean - margin, mean + margin


def compute_effect_size(
    group_a: List[float],
    group_b: List[float]
) -> Tuple[float, str]:
    """
    Compute Cohen's d effect size between two groups.

    Args:
        group_a: Values from first group
        group_b: Values from second group

    Returns:
        Tuple of (cohens_d, magnitude_string)
    """
    if len(group_a) < 2 or len(group_b) < 2:
        return 0.0, "insufficient_data"

    mean_a = compute_mean(group_a)
    mean_b = compute_mean(group_b)
    std_a = compute_std(group_a, mean_a)
    std_b = compute_std(group_b, mean_b)

    # Pooled standard deviation
    n_a = len(group_a)
    n_b = len(group_b)
    pooled_std = math.sqrt(
        ((n_a - 1) * std_a ** 2 + (n_b - 1) * std_b ** 2) /
        (n_a + n_b - 2)
    )

    if pooled_std == 0:
        return 0.0, "no_variance"

    cohens_d = (mean_a - mean_b) / pooled_std

    # Interpret magnitude
    abs_d = abs(cohens_d)
    if abs_d < 0.2:
        magnitude = "negligible"
    elif abs_d < 0.5:
        magnitude = "small"
    elif abs_d < 0.8:
        magnitude = "medium"
    else:
        magnitude = "large"

    return cohens_d, magnitude


def compare_models(
    results_a: List[Dict],
    results_b: List[Dict],
    model_a_name: str,
    model_b_name: str,
    metric_key: str
) -> ModelComparison:
    """
    Compare two models on a specific metric.

    Args:
        results_a: Results from model A
        results_b: Results from model B
        model_a_name: Name of model A
        model_b_name: Name of model B
        metric_key: Key to extract metric from results

    Returns:
        ModelComparison object
    """
    values_a = [r.get(metric_key, 0) for r in results_a if metric_key in r]
    values_b = [r.get(metric_key, 0) for r in results_b if metric_key in r]

    mean_a = compute_mean(values_a)
    mean_b = compute_mean(values_b)
    std_a = compute_std(values_a, mean_a)
    std_b = compute_std(values_b, mean_b)

    cohens_d, magnitude = compute_effect_size(values_a, values_b)

    # Pooled std for reference
    n_a, n_b = len(values_a), len(values_b)
    if n_a + n_b > 2:
        pooled_std = math.sqrt(
            ((n_a - 1) * std_a ** 2 + (n_b - 1) * std_b ** 2) /
            (n_a + n_b - 2)
        )
    else:
        pooled_std = 0.0

    return ModelComparison(
        model_a=model_a_name,
        model_b=model_b_name,
        metric=metric_key,
        mean_a=mean_a,
        mean_b=mean_b,
        std_a=std_a,
        std_b=std_b,
        cohens_d=cohens_d,
        effect_magnitude=magnitude,
        mean_difference=mean_a - mean_b,
        pooled_std=pooled_std
    )


class ExperimentAnalyzer:
    """
    Comprehensive analyzer for BddAgent experiment results.
    """

    def __init__(self, results_dir: str = "results"):
        self.results_dir = Path(results_dir)
        self.loaded_results: Dict[str, List[Dict]] = {}

    def load_results(self, pattern: str = "*_metrics.jsonl") -> Dict[str, List[Dict]]:
        """Load all result files matching pattern."""
        self.loaded_results = {}

        for path in self.results_dir.glob(pattern):
            model_mode = path.stem.replace("_metrics", "")
            results = []

            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        results.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

            self.loaded_results[model_mode] = results

        return self.loaded_results

    def compute_summary(
        self,
        results: List[Dict],
        metric_key: str
    ) -> StatisticalSummary:
        """Compute statistical summary for a metric."""
        values = [r.get(metric_key, 0) for r in results if metric_key in r]

        if not values:
            return StatisticalSummary(
                metric_name=metric_key,
                n=0, mean=0, std=0, min_val=0, max_val=0,
                median=0, ci_lower=0, ci_upper=0
            )

        mean, ci_lower, ci_upper = compute_confidence_interval(values)

        return StatisticalSummary(
            metric_name=metric_key,
            n=len(values),
            mean=mean,
            std=compute_std(values, mean),
            min_val=min(values),
            max_val=max(values),
            median=compute_median(values),
            ci_lower=ci_lower,
            ci_upper=ci_upper
        )

    def analyze_success_rates(self) -> Dict[str, Dict[str, float]]:
        """Analyze success rates across all loaded results."""
        analysis = {}

        for model_mode, results in self.loaded_results.items():
            if not results:
                continue

            total = len(results)
            successful = sum(1 for r in results if r.get('success', False))
            syntax_valid = sum(
                1 for r in results
                if r.get('validation', {}).get('syntax_valid', False)
            )

            analysis[model_mode] = {
                "total": total,
                "successful": successful,
                "success_rate": successful / total if total > 0 else 0,
                "syntax_valid": syntax_valid,
                "syntax_valid_rate": syntax_valid / total if total > 0 else 0
            }

            # Compute confidence interval for success rate
            if total > 0:
                # Wilson score interval for proportions
                p = successful / total
                z = 1.96  # 95% CI
                denominator = 1 + z**2 / total
                center = (p + z**2 / (2 * total)) / denominator
                spread = z * math.sqrt((p * (1 - p) + z**2 / (4 * total)) / total) / denominator
                analysis[model_mode]["success_rate_ci_lower"] = max(0, center - spread)
                analysis[model_mode]["success_rate_ci_upper"] = min(1, center + spread)

        return analysis

    def compare_all_models(
        self,
        metric_key: str = "success"
    ) -> List[ModelComparison]:
        """Compare all loaded models pairwise."""
        comparisons = []
        model_modes = list(self.loaded_results.keys())

        for i, model_a in enumerate(model_modes):
            for model_b in model_modes[i + 1:]:
                comparison = compare_models(
                    self.loaded_results[model_a],
                    self.loaded_results[model_b],
                    model_a,
                    model_b,
                    metric_key
                )
                comparisons.append(comparison)

        return comparisons

    def generate_report(self, output_path: Optional[str] = None) -> Dict[str, Any]:
        """Generate comprehensive analysis report."""
        report = {
            "generated_at": str(Path(__file__).parent),
            "models_analyzed": list(self.loaded_results.keys()),
            "success_rates": self.analyze_success_rates(),
            "summaries": {},
            "comparisons": {}
        }

        # Compute summaries for key metrics
        metrics_to_analyze = [
            "timing.duration_seconds",
            "tokens.total_tokens",
            "tokens.estimated_cost_usd",
            "iterations"
        ]

        for model_mode, results in self.loaded_results.items():
            report["summaries"][model_mode] = {}

            # Flatten nested metrics
            flattened = []
            for r in results:
                flat = {
                    "success": r.get("success", False),
                    "iterations": r.get("iterations", 0)
                }
                if "timing" in r:
                    flat["duration_seconds"] = r["timing"].get("duration_seconds", 0)
                if "tokens" in r:
                    flat["total_tokens"] = r["tokens"].get("total_tokens", 0)
                    flat["estimated_cost_usd"] = r["tokens"].get("estimated_cost_usd", 0)
                flattened.append(flat)

            for metric in ["duration_seconds", "total_tokens", "estimated_cost_usd", "iterations"]:
                summary = self.compute_summary(flattened, metric)
                report["summaries"][model_mode][metric] = summary.to_dict()

        # Model comparisons
        if len(self.loaded_results) >= 2:
            comparisons = self.compare_all_models("success")
            report["comparisons"]["success"] = [c.to_dict() for c in comparisons]

        # Save report if path provided
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, default=str)

        return report

    def print_summary(self):
        """Print human-readable summary."""
        print("\n" + "=" * 70)
        print("EXPERIMENT ANALYSIS SUMMARY")
        print("=" * 70)

        success_rates = self.analyze_success_rates()

        for model_mode, stats in success_rates.items():
            print(f"\n{model_mode}")
            print("-" * 50)
            print(f"  Total Tests: {stats['total']}")
            print(f"  Success Rate: {stats['success_rate']:.1%} "
                  f"(95% CI: [{stats.get('success_rate_ci_lower', 0):.1%}, "
                  f"{stats.get('success_rate_ci_upper', 0):.1%}])")
            print(f"  Syntax Valid: {stats['syntax_valid_rate']:.1%}")

        # Print comparisons if multiple models
        if len(self.loaded_results) >= 2:
            print("\n" + "-" * 70)
            print("MODEL COMPARISONS (Success Rate)")
            print("-" * 70)

            comparisons = self.compare_all_models("success")
            for comp in comparisons:
                print(f"\n{comp.model_a} vs {comp.model_b}:")
                print(f"  Mean difference: {comp.mean_difference:+.3f}")
                print(f"  Cohen's d: {comp.cohens_d:.3f} ({comp.effect_magnitude})")

        print("\n" + "=" * 70)
