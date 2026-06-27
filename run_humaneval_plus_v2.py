#!/usr/bin/env python3
"""
HumanEval+ evaluator that uses original pass results and tests plus_inputs
for robustness on solutions that already passed base tests.

This properly respects the original evaluation while adding HumanEval+ robustness testing.
"""

import json
import signal
from pathlib import Path
from contextlib import contextmanager
from evalplus.data import get_human_eval_plus

RESULTS_DIR = Path(__file__).parent / "results" / "bdd_vs_cot"
OUTPUT_DIR = Path(__file__).parent / "results" / "humaneval_plus"


class TimeoutError(Exception):
    pass


@contextmanager
def time_limit(seconds):
    """Simple timeout using signal (works on Unix)."""
    def signal_handler(signum, frame):
        raise TimeoutError("Timed out!")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


def extract_model_name(filename: str) -> str:
    """Extract model name from filename."""
    import re
    stem = filename.replace('.jsonl', '')
    match = re.match(r'^(bdd|cot|direct)_(.+)_(\d{8})_(\d{6})$', stem)
    if match:
        return match.group(2)
    return None


def get_latest_results_for_model(model_name: str) -> dict:
    """Get the latest result files for a model."""
    files = {}
    for condition in ["bdd", "cot", "direct"]:
        matches = []
        for f in RESULTS_DIR.glob(f"{condition}_*.jsonl"):
            extracted = extract_model_name(f.name)
            if extracted == model_name:
                matches.append(f)
        matches = sorted(matches, key=lambda p: p.stat().st_mtime, reverse=True)
        if matches:
            files[condition] = matches[0]
    return files


def test_plus_inputs(full_code: str, entry_point: str, plus_inputs: list,
                     canonical_solution: str, prompt: str, timeout: int = 10) -> tuple:
    """
    Test if a solution handles plus_inputs correctly.
    Compares outputs with the canonical solution.

    Returns (passed: bool, error: str or None)
    """
    if not plus_inputs:
        return True, None  # No plus inputs to test

    # Get the canonical function output for comparison
    canonical_code = prompt + canonical_solution

    try:
        with time_limit(timeout):
            # Execute canonical solution
            canonical_globals = {}
            exec(canonical_code, canonical_globals)
            canonical_func = canonical_globals.get(entry_point)

            # Execute our solution
            solution_globals = {}
            exec(full_code, solution_globals)
            solution_func = solution_globals.get(entry_point)

            if not canonical_func or not solution_func:
                return False, "function not found"

            # Test each plus input (limit to 20 for speed)
            for inp in plus_inputs[:20]:
                try:
                    if isinstance(inp, (list, tuple)):
                        expected = canonical_func(*inp)
                        actual = solution_func(*inp)
                    else:
                        expected = canonical_func(inp)
                        actual = solution_func(inp)

                    if expected != actual:
                        return False, f"output mismatch on plus input"
                except Exception as e:
                    return False, f"plus input error: {type(e).__name__}"

            return True, None

    except TimeoutError:
        return False, "timeout"
    except Exception as e:
        return False, f"{type(e).__name__}: {str(e)[:50]}"


def evaluate_file(jsonl_path: Path, problems: dict) -> dict:
    """
    Evaluate solutions using:
    - Original 'passed' field for HumanEval base
    - Plus inputs testing for HumanEval+ (only for originally passed solutions)
    """
    results = {
        "base_passed": 0,  # From original evaluation
        "plus_passed": 0,  # Original passed + survives plus inputs
        "total": 0,
        "details": []
    }

    with open(jsonl_path) as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            task_id = data["task_id"]

            if task_id not in problems:
                continue

            problem = problems[task_id]
            original_passed = data.get("passed", False)
            completion = data.get("generated_code", "")

            results["total"] += 1

            # Base result from original evaluation
            base_passed = original_passed
            if base_passed:
                results["base_passed"] += 1

            # Plus testing: only test if base passed
            plus_passed = False
            plus_error = None

            if base_passed and completion:
                prompt = problem["prompt"]
                full_code = prompt + completion
                entry_point = problem["entry_point"]
                plus_inputs = problem.get("plus_input", [])
                canonical = problem["canonical_solution"]

                plus_passed, plus_error = test_plus_inputs(
                    full_code, entry_point, plus_inputs, canonical, prompt
                )

                if plus_passed:
                    results["plus_passed"] += 1

            results["details"].append({
                "task_id": task_id,
                "base_passed": base_passed,
                "plus_passed": plus_passed,
                "plus_error": plus_error
            })

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run HumanEval+ evaluation (v2)")
    parser.add_argument("--model", type=str, help="Model name to evaluate")
    parser.add_argument("--list-models", action="store_true", help="List available models")
    parser.add_argument("--all", action="store_true", help="Evaluate all models")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading HumanEval+ problems...")
    problems = get_human_eval_plus()
    print(f"Loaded {len(problems)} problems")

    if args.list_models:
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

            results = evaluate_file(filepath, problems)

            base_rate = results["base_passed"] / results["total"] if results["total"] > 0 else 0
            plus_rate = results["plus_passed"] / results["total"] if results["total"] > 0 else 0

            # Robustness = % of base-passed that also pass plus
            robustness = results["plus_passed"] / results["base_passed"] if results["base_passed"] > 0 else 0

            print(f"  Total: {results['total']}")
            print(f"  HumanEval (base):  {results['base_passed']}/{results['total']} = {base_rate*100:.1f}%")
            print(f"  HumanEval+ (plus): {results['plus_passed']}/{results['total']} = {plus_rate*100:.1f}%")
            print(f"  Robustness:        {results['plus_passed']}/{results['base_passed']} = {robustness*100:.1f}%")

            model_results[condition] = {
                "base_passed": results["base_passed"],
                "plus_passed": results["plus_passed"],
                "total": results["total"],
                "base_rate": base_rate,
                "plus_rate": plus_rate,
                "robustness": robustness,
                "drop": base_rate - plus_rate
            }

        all_results[model] = model_results

    # Summary
    print("\n" + "="*90)
    print("SUMMARY: HumanEval vs HumanEval+ Results")
    print("="*90)
    print(f"{'Model':<30} {'Cond':<8} {'HumanEval':<12} {'HumanEval+':<12} {'Drop':<8} {'Robust':<8}")
    print("-"*90)

    for model, conditions in all_results.items():
        for cond, data in conditions.items():
            base = data["base_rate"]
            plus = data["plus_rate"]
            drop = data["drop"]
            robust = data["robustness"]
            print(f"{model:<30} {cond:<8} {base*100:>10.1f}% {plus*100:>10.1f}% {drop*100:>6.1f}% {robust*100:>6.1f}%")

    # Save summary
    summary_file = OUTPUT_DIR / "humaneval_plus_summary_v2.json"
    with open(summary_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSummary saved to: {summary_file}")


if __name__ == "__main__":
    main()
