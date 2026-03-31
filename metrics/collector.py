"""
Comprehensive Metrics Collection for BddAgent Experiments

Collects timing, token usage, cost, validation results, and code quality metrics
for rigorous experimental analysis.
"""

import json
import time
import os
import csv
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any
from pathlib import Path
from contextlib import contextmanager
import threading


@dataclass
class TimingMetrics:
    """Timing metrics for a single test case."""
    start_time: float = 0.0
    end_time: float = 0.0
    duration_seconds: float = 0.0
    llm_time_seconds: float = 0.0  # Time spent in LLM calls
    processing_time_seconds: float = 0.0  # Time spent in code extraction/processing


@dataclass
class TokenMetrics:
    """Token usage metrics."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0


@dataclass
class ValidationResult:
    """Validation results for generated code."""
    syntax_valid: bool = False
    syntax_error: Optional[str] = None
    has_function_body: bool = False
    meets_indentation: bool = False
    has_return_statement: bool = False
    complexity_score: Optional[float] = None
    lines_of_code: int = 0
    warnings: List[str] = field(default_factory=list)


@dataclass
class TestCaseMetrics:
    """Complete metrics for a single test case."""
    # Identification
    namespace: str = ""
    test_index: int = 0
    experiment_id: str = ""
    model_name: str = ""
    mode: str = ""

    # Timing
    timing: TimingMetrics = field(default_factory=TimingMetrics)

    # Tokens & Cost
    tokens: TokenMetrics = field(default_factory=TokenMetrics)

    # Execution
    iterations: int = 0
    retries: int = 0
    success: bool = False
    error_type: Optional[str] = None
    error_message: Optional[str] = None

    # Validation
    validation: ValidationResult = field(default_factory=ValidationResult)

    # Output
    generated_code: str = ""
    code_length: int = 0

    # Timestamps
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = asdict(self)
        return result


@dataclass
class ExperimentMetrics:
    """Aggregate metrics for an entire experiment."""
    experiment_id: str
    experiment_name: str
    model_name: str
    mode: str
    start_time: str
    end_time: Optional[str] = None

    # Aggregate counts
    total_tests: int = 0
    successful_tests: int = 0
    failed_tests: int = 0
    syntax_valid_count: int = 0

    # Aggregate timing
    total_duration_seconds: float = 0.0
    avg_duration_seconds: float = 0.0
    min_duration_seconds: float = float('inf')
    max_duration_seconds: float = 0.0

    # Aggregate tokens
    total_tokens: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    avg_tokens_per_test: float = 0.0
    total_cost_usd: float = 0.0

    # Aggregate iterations
    total_iterations: int = 0
    avg_iterations: float = 0.0

    # Success rates
    success_rate: float = 0.0
    syntax_valid_rate: float = 0.0

    # Error breakdown
    error_counts: Dict[str, int] = field(default_factory=dict)

    def compute_aggregates(self, test_metrics: List[TestCaseMetrics]):
        """Compute aggregate statistics from individual test metrics."""
        self.total_tests = len(test_metrics)
        if self.total_tests == 0:
            return

        durations = []
        for m in test_metrics:
            if m.success:
                self.successful_tests += 1
            else:
                self.failed_tests += 1
                error_type = m.error_type or "unknown"
                self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1

            if m.validation.syntax_valid:
                self.syntax_valid_count += 1

            durations.append(m.timing.duration_seconds)
            self.total_duration_seconds += m.timing.duration_seconds
            self.total_tokens += m.tokens.total_tokens
            self.total_prompt_tokens += m.tokens.prompt_tokens
            self.total_completion_tokens += m.tokens.completion_tokens
            self.total_cost_usd += m.tokens.estimated_cost_usd
            self.total_iterations += m.iterations

        # Compute averages
        self.avg_duration_seconds = self.total_duration_seconds / self.total_tests
        self.avg_tokens_per_test = self.total_tokens / self.total_tests
        self.avg_iterations = self.total_iterations / self.total_tests

        # Min/max
        if durations:
            self.min_duration_seconds = min(durations)
            self.max_duration_seconds = max(durations)

        # Rates
        self.success_rate = self.successful_tests / self.total_tests
        self.syntax_valid_rate = self.syntax_valid_count / self.total_tests

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


class MetricsCollector:
    """
    Thread-safe metrics collector for BddAgent experiments.

    Provides:
    - Real-time metrics collection during experiment execution
    - Automatic persistence to JSON and CSV
    - Aggregate statistics computation
    - Progress tracking
    """

    def __init__(
        self,
        experiment_id: str,
        experiment_name: str,
        model_name: str,
        mode: str,
        output_dir: str = "results"
    ):
        self.experiment_id = experiment_id
        self.experiment_name = experiment_name
        self.model_name = model_name
        self.mode = mode
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.test_metrics: List[TestCaseMetrics] = []
        self._lock = threading.Lock()
        self._current_test: Optional[TestCaseMetrics] = None
        self._test_start_time: float = 0.0

        # Initialize experiment metrics
        self.experiment_metrics = ExperimentMetrics(
            experiment_id=experiment_id,
            experiment_name=experiment_name,
            model_name=model_name,
            mode=mode,
            start_time=datetime.now().isoformat()
        )

    @contextmanager
    def track_test(self, namespace: str, test_index: int):
        """Context manager to track a single test case."""
        metrics = TestCaseMetrics(
            namespace=namespace,
            test_index=test_index,
            experiment_id=self.experiment_id,
            model_name=self.model_name,
            mode=self.mode
        )
        metrics.timing.start_time = time.time()

        self._current_test = metrics
        self._test_start_time = metrics.timing.start_time

        try:
            yield metrics
            metrics.success = True
        except Exception as e:
            metrics.success = False
            metrics.error_type = type(e).__name__
            metrics.error_message = str(e)
            raise
        finally:
            metrics.timing.end_time = time.time()
            metrics.timing.duration_seconds = (
                metrics.timing.end_time - metrics.timing.start_time
            )
            with self._lock:
                self.test_metrics.append(metrics)
            self._current_test = None

    def record_tokens(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        cost_per_input_token: float = 0.0,
        cost_per_output_token: float = 0.0
    ):
        """Record token usage for current test."""
        if self._current_test:
            self._current_test.tokens.prompt_tokens += prompt_tokens
            self._current_test.tokens.completion_tokens += completion_tokens
            self._current_test.tokens.total_tokens += prompt_tokens + completion_tokens
            self._current_test.tokens.estimated_cost_usd += (
                prompt_tokens * cost_per_input_token +
                completion_tokens * cost_per_output_token
            )

    def record_iterations(self, iterations: int):
        """Record number of agent iterations."""
        if self._current_test:
            self._current_test.iterations = iterations

    def record_validation(self, validation: ValidationResult):
        """Record validation results."""
        if self._current_test:
            self._current_test.validation = validation

    def record_generated_code(self, code: str):
        """Record the generated code."""
        if self._current_test:
            self._current_test.generated_code = code
            self._current_test.code_length = len(code)

    def record_error(self, error_type: str, error_message: str):
        """Record an error for the current test."""
        if self._current_test:
            self._current_test.success = False
            self._current_test.error_type = error_type
            self._current_test.error_message = error_message

    def get_progress(self) -> Dict[str, Any]:
        """Get current progress statistics."""
        with self._lock:
            completed = len(self.test_metrics)
            successful = sum(1 for m in self.test_metrics if m.success)
            return {
                "completed": completed,
                "successful": successful,
                "failed": completed - successful,
                "success_rate": successful / completed if completed > 0 else 0.0
            }

    def finalize(self) -> ExperimentMetrics:
        """Finalize experiment and compute aggregate metrics."""
        self.experiment_metrics.end_time = datetime.now().isoformat()
        self.experiment_metrics.compute_aggregates(self.test_metrics)
        return self.experiment_metrics

    def save_results(self, prefix: str = ""):
        """Save all results to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"{prefix}_{self.model_name}_{self.mode}_{timestamp}" if prefix else \
                    f"{self.model_name}_{self.mode}_{timestamp}"

        # Clean filename
        base_name = base_name.replace("/", "_").replace(":", "_")

        # Save individual test metrics as JSONL
        jsonl_path = self.output_dir / f"{base_name}_metrics.jsonl"
        with open(jsonl_path, 'w', encoding='utf-8') as f:
            for metrics in self.test_metrics:
                f.write(json.dumps(metrics.to_dict(), default=str) + '\n')

        # Save experiment summary as JSON
        summary_path = self.output_dir / f"{base_name}_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(self.experiment_metrics.to_dict(), f, indent=2, default=str)

        # Save as CSV for easy analysis
        csv_path = self.output_dir / f"{base_name}_results.csv"
        self._save_csv(csv_path)

        # Save failed cases separately
        failed_path = self.output_dir / f"{base_name}_failed.jsonl"
        failed_cases = [m for m in self.test_metrics if not m.success]
        if failed_cases:
            with open(failed_path, 'w', encoding='utf-8') as f:
                for metrics in failed_cases:
                    f.write(json.dumps(metrics.to_dict(), default=str) + '\n')

        return {
            "metrics_jsonl": str(jsonl_path),
            "summary_json": str(summary_path),
            "results_csv": str(csv_path),
            "failed_jsonl": str(failed_path) if failed_cases else None
        }

    def _save_csv(self, path: Path):
        """Save metrics to CSV format."""
        if not self.test_metrics:
            return

        fieldnames = [
            'namespace', 'test_index', 'model_name', 'mode', 'success',
            'duration_seconds', 'prompt_tokens', 'completion_tokens',
            'total_tokens', 'estimated_cost_usd', 'iterations',
            'syntax_valid', 'lines_of_code', 'error_type', 'error_message'
        ]

        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for m in self.test_metrics:
                writer.writerow({
                    'namespace': m.namespace,
                    'test_index': m.test_index,
                    'model_name': m.model_name,
                    'mode': m.mode,
                    'success': m.success,
                    'duration_seconds': f"{m.timing.duration_seconds:.3f}",
                    'prompt_tokens': m.tokens.prompt_tokens,
                    'completion_tokens': m.tokens.completion_tokens,
                    'total_tokens': m.tokens.total_tokens,
                    'estimated_cost_usd': f"{m.tokens.estimated_cost_usd:.6f}",
                    'iterations': m.iterations,
                    'syntax_valid': m.validation.syntax_valid,
                    'lines_of_code': m.validation.lines_of_code,
                    'error_type': m.error_type or '',
                    'error_message': m.error_message or ''
                })

    def print_summary(self):
        """Print a summary of the experiment results."""
        m = self.experiment_metrics
        print("\n" + "=" * 60)
        print(f"EXPERIMENT SUMMARY: {m.experiment_name}")
        print("=" * 60)
        print(f"Experiment ID: {m.experiment_id}")
        print(f"Model: {m.model_name}")
        print(f"Mode: {m.mode}")
        print(f"Duration: {m.start_time} to {m.end_time}")
        print("-" * 60)
        print(f"Total Tests: {m.total_tests}")
        print(f"Successful: {m.successful_tests} ({m.success_rate:.1%})")
        print(f"Failed: {m.failed_tests}")
        print(f"Syntax Valid: {m.syntax_valid_count} ({m.syntax_valid_rate:.1%})")
        print("-" * 60)
        print(f"Total Duration: {m.total_duration_seconds:.2f}s")
        print(f"Avg per Test: {m.avg_duration_seconds:.2f}s")
        print(f"Range: {m.min_duration_seconds:.2f}s - {m.max_duration_seconds:.2f}s")
        print("-" * 60)
        print(f"Total Tokens: {m.total_tokens:,}")
        print(f"Avg Tokens/Test: {m.avg_tokens_per_test:.0f}")
        print(f"Total Cost: ${m.total_cost_usd:.4f}")
        print("-" * 60)
        print(f"Avg Iterations: {m.avg_iterations:.1f}")
        if m.error_counts:
            print("\nError Breakdown:")
            for error_type, count in sorted(m.error_counts.items()):
                print(f"  {error_type}: {count}")
        print("=" * 60)
