#!/usr/bin/env python3
"""
Convert BddAgent ablation results to DevEval completion format.

BddAgent format:
    {"namespace": "...", "generated_code": "...", "with_bdd": true, ...}

DevEval format:
    {"namespace": "...", "completion": "..."}
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List


def convert_single_file(input_file: str, output_file: str) -> Dict[str, int]:
    """Convert a single JSONL file to DevEval format.

    Returns:
        Stats dict with counts of converted, skipped, errors
    """
    stats = {"converted": 0, "skipped": 0, "errors": 0}

    with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
        for line_num, line in enumerate(f_in, 1):
            try:
                result = json.loads(line.strip())

                # Skip failed generations
                if not result.get('success', False):
                    stats["skipped"] += 1
                    continue

                # Convert to DevEval format
                deveval_record = {
                    "namespace": result['namespace'],
                    "completion": result['generated_code']
                }

                f_out.write(json.dumps(deveval_record) + '\n')
                stats["converted"] += 1

            except json.JSONDecodeError as e:
                print(f"Warning: Invalid JSON on line {line_num}: {e}")
                stats["errors"] += 1
            except KeyError as e:
                print(f"Warning: Missing field {e} on line {line_num}")
                stats["errors"] += 1

    return stats


def convert_ablation_results(results_dir: str, output_dir: str, condition: str = None):
    """Convert all ablation results to DevEval format.

    Args:
        results_dir: Directory containing ablation JSONL files
        output_dir: Directory to write DevEval-format files
        condition: Optional filter for 'bdd' or 'no_bdd' only
    """
    results_path = Path(results_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Find all JSONL files
    jsonl_files = list(results_path.glob("*.jsonl"))

    if not jsonl_files:
        print(f"No JSONL files found in {results_dir}")
        return

    print(f"Found {len(jsonl_files)} JSONL files")

    # Group files by condition (bdd vs no_bdd)
    bdd_results = []
    no_bdd_results = []

    for f in jsonl_files:
        if 'no_bdd' in f.name.lower():
            no_bdd_results.append(f)
        elif 'bdd' in f.name.lower():
            bdd_results.append(f)

    # Merge and convert BDD results
    if condition is None or condition == 'bdd':
        if bdd_results:
            print(f"\nProcessing {len(bdd_results)} BDD result files...")
            merged_output = output_path / "bdd_completions.jsonl"
            total_stats = {"converted": 0, "skipped": 0, "errors": 0}

            with open(merged_output, 'w') as f_out:
                seen_namespaces = set()
                for result_file in sorted(bdd_results, reverse=True):  # Most recent first
                    with open(result_file) as f_in:
                        for line in f_in:
                            try:
                                result = json.loads(line.strip())
                                namespace = result.get('namespace')

                                # Skip duplicates (keep first occurrence = most recent)
                                if namespace in seen_namespaces:
                                    continue
                                seen_namespaces.add(namespace)

                                if not result.get('success', False):
                                    total_stats["skipped"] += 1
                                    continue

                                deveval_record = {
                                    "namespace": namespace,
                                    "completion": result['generated_code']
                                }
                                f_out.write(json.dumps(deveval_record) + '\n')
                                total_stats["converted"] += 1

                            except (json.JSONDecodeError, KeyError) as e:
                                total_stats["errors"] += 1

            print(f"  BDD: {total_stats['converted']} converted, "
                  f"{total_stats['skipped']} skipped, {total_stats['errors']} errors")
            print(f"  Output: {merged_output}")

    # Merge and convert No-BDD results
    if condition is None or condition == 'no_bdd':
        if no_bdd_results:
            print(f"\nProcessing {len(no_bdd_results)} No-BDD result files...")
            merged_output = output_path / "no_bdd_completions.jsonl"
            total_stats = {"converted": 0, "skipped": 0, "errors": 0}

            with open(merged_output, 'w') as f_out:
                seen_namespaces = set()
                for result_file in sorted(no_bdd_results, reverse=True):
                    with open(result_file) as f_in:
                        for line in f_in:
                            try:
                                result = json.loads(line.strip())
                                namespace = result.get('namespace')

                                if namespace in seen_namespaces:
                                    continue
                                seen_namespaces.add(namespace)

                                if not result.get('success', False):
                                    total_stats["skipped"] += 1
                                    continue

                                deveval_record = {
                                    "namespace": namespace,
                                    "completion": result['generated_code']
                                }
                                f_out.write(json.dumps(deveval_record) + '\n')
                                total_stats["converted"] += 1

                            except (json.JSONDecodeError, KeyError) as e:
                                total_stats["errors"] += 1

            print(f"  No-BDD: {total_stats['converted']} converted, "
                  f"{total_stats['skipped']} skipped, {total_stats['errors']} errors")
            print(f"  Output: {merged_output}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert BddAgent results to DevEval completion format"
    )
    parser.add_argument(
        "--input", "-i",
        default="results/ablation",
        help="Input directory containing ablation JSONL files"
    )
    parser.add_argument(
        "--output", "-o",
        default="results/deveval_eval",
        help="Output directory for DevEval-format files"
    )
    parser.add_argument(
        "--condition", "-c",
        choices=["bdd", "no_bdd"],
        default=None,
        help="Only convert specific condition (default: both)"
    )

    args = parser.parse_args()

    print("="*60)
    print("Converting BddAgent Results to DevEval Format")
    print("="*60)
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")

    convert_ablation_results(args.input, args.output, args.condition)

    print("\n" + "="*60)
    print("Conversion complete!")
    print("="*60)


if __name__ == "__main__":
    main()
