"""
Experiment Configuration System for BddAgent

Provides YAML-based experiment configuration with validation,
versioning, and reproducibility controls.
"""

import os
import yaml
import hashlib
import subprocess
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any
from pathlib import Path


@dataclass
class ModelConfig:
    """Configuration for a single model."""
    name: str
    provider: str  # azure, openai, gemini, anthropic
    model_id: str
    temperature: float = 0.2
    max_tokens: int = 4096
    seed: Optional[int] = None

    def get_litellm_id(self) -> str:
        """Get the litellm-compatible model identifier."""
        return f"{self.provider}/{self.model_id}"


@dataclass
class DatasetConfig:
    """Configuration for experiment dataset."""
    path: str
    mode: str  # without_context, local_file_completion, local_file_infiling
    sample_size: Optional[int] = None  # None = all samples
    sample_seed: Optional[int] = 42  # For reproducible sampling
    stratify_by: Optional[str] = None  # Field to stratify sampling by


@dataclass
class MetricsConfig:
    """Configuration for metrics collection."""
    collect_timing: bool = True
    collect_tokens: bool = True
    collect_cost: bool = True
    collect_iterations: bool = True
    validate_syntax: bool = True
    validate_runtime: bool = False  # Requires test execution
    code_quality_metrics: bool = True


@dataclass
class OutputConfig:
    """Configuration for experiment outputs."""
    base_dir: str = "results"
    save_raw_outputs: bool = True
    save_metrics_json: bool = True
    save_summary_csv: bool = True
    save_failed_cases: bool = True


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""
    experiment_id: str
    experiment_name: str
    description: str
    models: List[ModelConfig]
    dataset: DatasetConfig
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    # Reproducibility
    random_seed: int = 42
    max_retries: int = 3
    timeout_seconds: int = 300

    # Metadata (auto-populated)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    git_hash: Optional[str] = None
    python_version: Optional[str] = None

    def __post_init__(self):
        """Populate metadata fields."""
        if self.git_hash is None:
            self.git_hash = self._get_git_hash()
        if self.python_version is None:
            import sys
            self.python_version = sys.version

    @staticmethod
    def _get_git_hash() -> Optional[str]:
        """Get current git commit hash."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True, text=True, timeout=5
            )
            return result.stdout.strip() if result.returncode == 0 else None
        except Exception:
            return None

    def get_config_hash(self) -> str:
        """Generate a hash of the configuration for tracking."""
        config_str = yaml.dump(asdict(self), sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()[:12]

    def get_output_dir(self) -> Path:
        """Get the output directory for this experiment."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dir_name = f"{self.experiment_id}_{timestamp}"
        return Path(self.output.base_dir) / dir_name

    def to_yaml(self, path: str):
        """Save configuration to YAML file."""
        with open(path, 'w') as f:
            yaml.dump(asdict(self), f, default_flow_style=False, sort_keys=False)

    @classmethod
    def from_yaml(cls, path: str) -> 'ExperimentConfig':
        """Load configuration from YAML file."""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)

        # Convert nested dicts to dataclasses
        data['models'] = [ModelConfig(**m) for m in data.get('models', [])]
        data['dataset'] = DatasetConfig(**data.get('dataset', {}))
        data['metrics'] = MetricsConfig(**data.get('metrics', {}))
        data['output'] = OutputConfig(**data.get('output', {}))

        return cls(**data)

    def validate(self) -> List[str]:
        """Validate configuration and return list of errors."""
        errors = []

        # Check dataset path
        if not os.path.exists(self.dataset.path):
            errors.append(f"Dataset path does not exist: {self.dataset.path}")

        # Check mode validity
        valid_modes = ['without_context', 'local_file_completion', 'local_file_infiling']
        if self.dataset.mode not in valid_modes:
            errors.append(f"Invalid mode '{self.dataset.mode}'. Must be one of: {valid_modes}")

        # Check models
        if not self.models:
            errors.append("At least one model must be specified")

        for model in self.models:
            if model.temperature < 0 or model.temperature > 2:
                errors.append(f"Model {model.name}: temperature must be between 0 and 2")
            if model.max_tokens < 1:
                errors.append(f"Model {model.name}: max_tokens must be positive")

        return errors


def create_default_config(
    dataset_path: str,
    mode: str = 'local_file_completion',
    experiment_name: str = "BddAgent Experiment"
) -> ExperimentConfig:
    """Create a default experiment configuration."""

    experiment_id = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    models = [
        ModelConfig(
            name="GPT-4.1-mini",
            provider="azure",
            model_id="gpt-4.1-mini",
            temperature=0.2,
            seed=42
        ),
        ModelConfig(
            name="GPT-4o",
            provider="openai",
            model_id="gpt-4o",
            temperature=0.2,
            seed=42
        ),
        ModelConfig(
            name="Gemini-2.5-Pro",
            provider="gemini",
            model_id="gemini-2.5-pro",
            temperature=0.2
        ),
    ]

    dataset = DatasetConfig(
        path=dataset_path,
        mode=mode,
        sample_size=None,  # Use all samples
        sample_seed=42
    )

    return ExperimentConfig(
        experiment_id=experiment_id,
        experiment_name=experiment_name,
        description="Multi-model BDD-driven code generation experiment",
        models=models,
        dataset=dataset
    )


# Example usage and CLI
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate experiment configuration")
    parser.add_argument("--dataset", required=True, help="Path to dataset JSONL file")
    parser.add_argument("--mode", default="local_file_completion",
                       choices=['without_context', 'local_file_completion', 'local_file_infiling'])
    parser.add_argument("--output", default="experiment_config.yaml", help="Output config file")

    args = parser.parse_args()

    config = create_default_config(args.dataset, args.mode)
    config.to_yaml(args.output)
    print(f"Configuration saved to {args.output}")
