#!/usr/bin/env python3
"""
Run DevEval Pass@k evaluation on BddAgent results.

This script wraps the official DevEval pass_k.py evaluation harness
to evaluate generated code against test cases.

Prerequisites:
1. Download Source_Code.tar.gz from HuggingFace and extract
2. Setup DevEval conda environment
3. Convert results using convert_to_deveval.py
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path


def check_prerequisites(source_code_root: str, data_file: str) -> bool:
    """Check that all prerequisites are met."""
    errors = []

    # Check Source_Code exists
    if not os.path.isdir(source_code_root):
        errors.append(f"Source_Code not found at: {source_code_root}")
        errors.append("  Download from: https://huggingface.co/datasets/LJ0815/DevEval/resolve/main/Source_Code.tar.gz")

    # Check data.jsonl exists
    if not os.path.isfile(data_file):
        errors.append(f"data.jsonl not found at: {data_file}")

    if errors:
        print("Prerequisites check FAILED:")
        for e in errors:
            print(f"  - {e}")
        return False

    print("Prerequisites check PASSED")
    return True


def run_pass_k(
    completion_file: str,
    log_file: str,
    source_code_root: str,
    data_file: str,
    pass_k_script: str,
    n: int = 1,
    k: int = 1
) -> int:
    """Run the DevEval pass_k.py evaluation.

    Returns:
        Return code from pass_k.py (0 = success)
    """
    cmd = [
        sys.executable,  # Use current Python interpreter
        pass_k_script,
        "--output_file", completion_file,
        "--log_file", log_file,
        "--source_code_root", source_code_root,
        "--data_file", data_file,
        "--n", str(n),
        "--k", str(k)
    ]

    print(f"\nRunning: {' '.join(cmd)}")
    print("-" * 60)

    result = subprocess.run(cmd, cwd=os.path.dirname(pass_k_script))
    return result.returncode


def main():
    parser = argparse.ArgumentParser(
        description="Run DevEval Pass@k evaluation on BddAgent results"
    )
    parser.add_argument(
        "--deveval-root",
        default="data/DevEval-main",
        help="Path to DevEval directory"
    )
    parser.add_argument(
        "--results-dir",
        default="results/deveval_eval",
        help="Directory containing converted completion files"
    )
    parser.add_argument(
        "--condition", "-c",
        choices=["bdd", "no_bdd", "both"],
        default="both",
        help="Which condition to evaluate"
    )
    parser.add_argument(
        "--n", type=int, default=1,
        help="Number of completions per task"
    )
    parser.add_argument(
        "--k", type=int, default=1,
        help="K value for Pass@k"
    )

    args = parser.parse_args()

    # Setup paths
    deveval_root = Path(args.deveval_root)
    results_dir = Path(args.results_dir)

    source_code_root = str(deveval_root / "Source_Code")
    data_file = str(deveval_root / "data.jsonl")
    pass_k_script = str(deveval_root / "pass_k.py")

    print("=" * 60)
    print("DevEval Pass@k Evaluation")
    print("=" * 60)
    print(f"DevEval root: {deveval_root}")
    print(f"Results dir: {results_dir}")
    print(f"Evaluating: {args.condition}")
    print(f"Pass@{args.k} with n={args.n}")

    # Check prerequisites
    if not check_prerequisites(source_code_root, data_file):
        sys.exit(1)

    # Check pass_k.py exists
    if not os.path.isfile(pass_k_script):
        print(f"ERROR: pass_k.py not found at {pass_k_script}")
        sys.exit(1)

    results_dir.mkdir(parents=True, exist_ok=True)

    # Run evaluations
    conditions = []
    if args.condition in ["bdd", "both"]:
        conditions.append(("bdd", "bdd_completions.jsonl", "bdd_test_output.jsonl"))
    if args.condition in ["no_bdd", "both"]:
        conditions.append(("no_bdd", "no_bdd_completions.jsonl", "no_bdd_test_output.jsonl"))

    for name, completion_file, log_file in conditions:
        completion_path = results_dir / completion_file
        log_path = results_dir / log_file

        if not completion_path.exists():
            print(f"\nWARNING: {completion_path} not found, skipping {name}")
            continue

        print(f"\n{'='*60}")
        print(f"Evaluating: {name.upper()}")
        print(f"{'='*60}")
        print(f"Completions: {completion_path}")
        print(f"Log output: {log_path}")

        # Count samples
        with open(completion_path) as f:
            n_samples = sum(1 for _ in f)
        print(f"Samples: {n_samples}")

        returncode = run_pass_k(
            str(completion_path),
            str(log_path),
            source_code_root,
            data_file,
            pass_k_script,
            args.n,
            args.k
        )

        if returncode != 0:
            print(f"WARNING: pass_k.py returned non-zero exit code: {returncode}")

    print("\n" + "=" * 60)
    print("Evaluation complete!")
    print("=" * 60)
    print(f"Results saved to: {results_dir}")
    print("\nNext: Run analyze_pass_at_k.py to compute statistics")


if __name__ == "__main__":
    main()
