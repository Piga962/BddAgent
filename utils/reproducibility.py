"""
Reproducibility Controls for BddAgent Experiments

Provides seed management, environment capture, and experiment manifests
for ensuring reproducible experimental results.
"""

import os
import sys
import json
import random
import hashlib
import platform
import subprocess
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any


def set_seeds(seed: int = 42):
    """
    Set random seeds for reproducibility.

    Args:
        seed: Random seed value
    """
    random.seed(seed)

    # Try to set numpy seed if available
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass

    # Set environment variable for any subprocess
    os.environ['PYTHONHASHSEED'] = str(seed)

    return seed


def get_git_info() -> Dict[str, Optional[str]]:
    """Get git repository information."""
    info = {
        "commit_hash": None,
        "branch": None,
        "remote_url": None,
        "is_dirty": None,
        "commit_date": None
    }

    try:
        # Get commit hash
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            info["commit_hash"] = result.stdout.strip()

        # Get branch name
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            info["branch"] = result.stdout.strip()

        # Get remote URL
        result = subprocess.run(
            ["git", "config", "--get", "remote.origin.url"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            info["remote_url"] = result.stdout.strip()

        # Check if dirty
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            info["is_dirty"] = len(result.stdout.strip()) > 0

        # Get commit date
        result = subprocess.run(
            ["git", "log", "-1", "--format=%ci"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            info["commit_date"] = result.stdout.strip()

    except Exception:
        pass

    return info


def get_environment_info() -> Dict[str, Any]:
    """
    Capture comprehensive environment information.

    Returns:
        Dictionary with environment details
    """
    info = {
        "timestamp": datetime.now().isoformat(),
        "python": {
            "version": sys.version,
            "executable": sys.executable,
            "platform": sys.platform
        },
        "system": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "node": platform.node()
        },
        "cwd": os.getcwd(),
        "git": get_git_info()
    }

    # Get installed packages
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode == 0:
            packages = {}
            for line in result.stdout.strip().split('\n'):
                if '==' in line:
                    name, version = line.split('==', 1)
                    packages[name] = version
            info["packages"] = packages
    except Exception:
        info["packages"] = None

    # Get relevant environment variables (sanitized)
    env_vars = {}
    relevant_vars = [
        'PYTHONPATH', 'VIRTUAL_ENV', 'CONDA_DEFAULT_ENV',
        'CUDA_VISIBLE_DEVICES', 'OMP_NUM_THREADS'
    ]
    for var in relevant_vars:
        if var in os.environ:
            env_vars[var] = os.environ[var]

    # Check for API keys (presence only, not values)
    api_keys = ['OPENAI_API_KEY', 'AZURE_OPENAI_KEY', 'GEMINI_API_KEY', 'ANTHROPIC_API_KEY']
    info["api_keys_present"] = {key: key in os.environ for key in api_keys}

    info["environment_variables"] = env_vars

    return info


@dataclass
class ExperimentManifest:
    """
    Complete experiment manifest for reproducibility.

    Records all information needed to reproduce an experiment.
    """
    experiment_id: str
    experiment_name: str
    description: str

    # Configuration
    config_hash: str
    config: Dict[str, Any]

    # Environment
    environment: Dict[str, Any] = field(default_factory=dict)

    # Timing
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    started_at: Optional[str] = None
    completed_at: Optional[str] = None

    # Results summary
    results_summary: Optional[Dict[str, Any]] = None

    # File paths
    output_dir: Optional[str] = None
    config_file: Optional[str] = None
    results_files: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.environment:
            self.environment = get_environment_info()

    def start(self):
        """Mark experiment as started."""
        self.started_at = datetime.now().isoformat()

    def complete(self, results_summary: Optional[Dict[str, Any]] = None):
        """Mark experiment as completed."""
        self.completed_at = datetime.now().isoformat()
        if results_summary:
            self.results_summary = results_summary

    def add_result_file(self, path: str):
        """Add a result file to the manifest."""
        self.results_files.append(path)

    def get_manifest_hash(self) -> str:
        """Generate a unique hash for this manifest."""
        content = json.dumps(asdict(self), sort_keys=True, default=str)
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def save(self, path: Optional[str] = None):
        """Save manifest to JSON file."""
        if path is None:
            if self.output_dir:
                path = os.path.join(self.output_dir, f"manifest_{self.experiment_id}.json")
            else:
                path = f"manifest_{self.experiment_id}.json"

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(asdict(self), f, indent=2, default=str)

        return path

    @classmethod
    def load(cls, path: str) -> 'ExperimentManifest':
        """Load manifest from JSON file."""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls(**data)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    def print_summary(self):
        """Print a human-readable summary."""
        print("\n" + "=" * 60)
        print("EXPERIMENT MANIFEST")
        print("=" * 60)
        print(f"ID: {self.experiment_id}")
        print(f"Name: {self.experiment_name}")
        print(f"Description: {self.description}")
        print(f"Config Hash: {self.config_hash}")
        print("-" * 60)
        print(f"Created: {self.created_at}")
        print(f"Started: {self.started_at or 'Not started'}")
        print(f"Completed: {self.completed_at or 'Not completed'}")
        print("-" * 60)
        git_info = self.environment.get('git', {})
        print(f"Git Commit: {git_info.get('commit_hash', 'N/A')}")
        print(f"Git Branch: {git_info.get('branch', 'N/A')}")
        print(f"Git Dirty: {git_info.get('is_dirty', 'N/A')}")
        print("-" * 60)
        python_info = self.environment.get('python', {})
        print(f"Python: {python_info.get('version', 'N/A').split()[0]}")
        print(f"Platform: {self.environment.get('system', {}).get('platform', 'N/A')}")
        print("-" * 60)
        if self.results_summary:
            print("Results Summary:")
            for key, value in self.results_summary.items():
                print(f"  {key}: {value}")
        print("=" * 60)


def create_manifest(
    experiment_id: str,
    experiment_name: str,
    description: str,
    config: Dict[str, Any],
    output_dir: Optional[str] = None
) -> ExperimentManifest:
    """
    Create a new experiment manifest.

    Args:
        experiment_id: Unique experiment identifier
        experiment_name: Human-readable name
        description: Experiment description
        config: Experiment configuration
        output_dir: Output directory for results

    Returns:
        ExperimentManifest instance
    """
    # Compute config hash
    config_str = json.dumps(config, sort_keys=True, default=str)
    config_hash = hashlib.sha256(config_str.encode()).hexdigest()[:12]

    manifest = ExperimentManifest(
        experiment_id=experiment_id,
        experiment_name=experiment_name,
        description=description,
        config_hash=config_hash,
        config=config,
        output_dir=output_dir
    )

    return manifest
