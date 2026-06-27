#!/usr/bin/env python3
"""
Convert BDD vs CoT results to EvalPlus format and run HumanEval+ evaluation.

EvalPlus provides 80x more test cases per problem, testing solution robustness.
"""

import json
import os
import subprocess
import tempfile
from pathlib import Path
from evalplus.data import get_human_eval_plus

RESULTS_DIR = Path(__file__).parent / "results" / "bdd_vs_cot"
EVALPLUS_DIR = Path(__file__).parent / "results" / "evalplus"


def load_humaneval_plus():
    """Load HumanEval+ problems."""
    return get_human_eval_plus()


def convert_to_evalplus_format(jsonl_path: Path, problems: dict) -> list:
    """
    Convert our JSONL results to EvalPlus format.

    EvalPlus expects: {"task_id": "HumanEval/0", "completion": "code..."}
    Our format has: {"task_id": "HumanEval/0", "generated_code": "code..."}

    The completion should be the code that follows the prompt (function body).
    """
    results = []
    with open(jsonl_path) as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            task_id = data["task_id"]

            # Our generated_code is just the function body (indented)
            # EvalPlus expects the completion to follow the prompt
            completion = data.get("generated_code", "")

            if completion:
                results.append({
                    "task_id": task_id,
                    "completion": completion
                })

    return results


def run_evalplus_evaluation(samples_path: Path, output_path: Path) -> dict:
    """Run EvalPlus evaluation on samples."""
    cmd = [
        "python3", "-m", "evalplus.evaluate",
        "--dataset", "humaneval",
        "--samples", str(samples_path),
        "--i-just-wanna-run",  # Skip sanitization
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    # Parse results from output
    output = result.stdout + result.stderr
    return output


def extract_model_name(filename: str) -> str:
    """Extract model name from filename like 'bdd_gpt-4o_20260401_154931.jsonl'."""
    import re
    # Remove condition prefix and .jsonl suffix
    stem = filename.replace('.jsonl', '')
    # Pattern: {condition}_{model}_{YYYYMMDD}_{HHMMSS}
    # Timestamp is always last two underscore-separated parts (8 digits, 6 digits)
    match = re.match(r'^(bdd|cot|direct)_(.+)_(\d{8})_(\d{6})$', stem)
    if match:
        return match.group(2)
    return None


def get_latest_results_for_model(model_name: str) -> dict:
    """Get the latest result files for a model (bdd, cot, direct)."""
    files = {}
    for condition in ["bdd", "cot", "direct"]:
        # Find all files for this condition
        matches = []
        for f in RESULTS_DIR.glob(f"{condition}_*.jsonl"):
            extracted = extract_model_name(f.name)
            if extracted == model_name:
                matches.append(f)
        # Sort by modification time, newest first
        matches = sorted(matches, key=lambda p: p.stat().st_mtime, reverse=True)
        if matches:
            files[condition] = matches[0]
    return files


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run EvalPlus (HumanEval+) evaluation")
    parser.add_argument("--model", type=str, help="Model name to evaluate (e.g., gpt-4o)")
    parser.add_argument("--list-models", action="store_true", help="List available models")
    parser.add_argument("--all", action="store_true", help="Evaluate all models")
    args = parser.parse_args()

    # Create output directory
    EVALPLUS_DIR.mkdir(parents=True, exist_ok=True)

    # Load HumanEval+ problems
    print("Loading HumanEval+ problems...")
    problems = load_humaneval_plus()
    print(f"Loaded {len(problems)} problems")

    if args.list_models:
        # Find all unique models
        models = set()
        for f in RESULTS_DIR.glob("bdd_*.jsonl"):
            model = extract_model_name(f.name)
            if model:
                models.add(model)
        print("Available models:")
        for m in sorted(models):
            files = get_latest_results_for_model(m)
            conditions = ", ".join(files.keys())
            print(f"  - {m} ({conditions})")
        return

    # Determine which models to evaluate
    if args.all:
        models = set()
        for f in RESULTS_DIR.glob("bdd_*.jsonl"):
            model = extract_model_name(f.name)
            if model:
                models.add(model)
        models = sorted(models)
    elif args.model:
        models = [args.model]
    else:
        parser.print_help()
        return

    # Process each model
    all_results = {}
    for model in models:
        print(f"\n{'='*60}")
        print(f"Evaluating model: {model}")
        print('='*60)

        files = get_latest_results_for_model(model)
        if not files:
            print(f"  No result files found for {model}")
            continue

        model_results = {}
        for condition, filepath in files.items():
            print(f"\n  Condition: {condition}")
            print(f"  File: {filepath.name}")

            # Convert to EvalPlus format
            samples = convert_to_evalplus_format(filepath, problems)
            print(f"  Samples: {len(samples)}")

            if len(samples) == 0:
                print("  Skipping - no valid samples")
                continue

            # Write to temp file
            samples_file = EVALPLUS_DIR / f"{model}_{condition}_samples.jsonl"
            with open(samples_file, 'w') as f:
                for s in samples:
                    f.write(json.dumps(s) + '\n')

            print(f"  Wrote samples to: {samples_file}")

            # Run EvalPlus
            print(f"  Running EvalPlus evaluation...")
            output = run_evalplus_evaluation(samples_file, EVALPLUS_DIR)
            print(output)

            # Try to parse pass rates from output
            # EvalPlus outputs: "pass@1: 0.XXX"
            import re
            he_match = re.search(r'Base.*?pass@1:\s*([\d.]+)', output)
            hep_match = re.search(r'Plus.*?pass@1:\s*([\d.]+)', output)

            model_results[condition] = {
                "humaneval_pass1": float(he_match.group(1)) if he_match else None,
                "humanevalplus_pass1": float(hep_match.group(1)) if hep_match else None,
                "n_samples": len(samples)
            }

        all_results[model] = model_results

    # Summary
    print("\n" + "="*80)
    print("SUMMARY: HumanEval vs HumanEval+ Results")
    print("="*80)
    print(f"{'Model':<25} {'Cond':<8} {'HumanEval':<12} {'HumanEval+':<12} {'Drop':<8}")
    print("-"*80)

    for model, conditions in all_results.items():
        for cond, data in conditions.items():
            he = data.get("humaneval_pass1")
            hep = data.get("humanevalplus_pass1")
            if he and hep:
                drop = he - hep
                print(f"{model:<25} {cond:<8} {he*100:>10.1f}% {hep*100:>10.1f}% {drop*100:>6.1f}%")
            else:
                print(f"{model:<25} {cond:<8} {'N/A':<12} {'N/A':<12}")

    # Save summary
    summary_file = EVALPLUS_DIR / "evalplus_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSummary saved to: {summary_file}")


if __name__ == "__main__":
    main()
