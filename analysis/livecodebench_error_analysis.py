#!/usr/bin/env python3
"""
LiveCodeBench Error Analysis Script

Analyzes why CoT catastrophically fails on some models (e.g., GPT-4o: 2% vs 16% Direct).
Categorizes failure modes and generates visualizations.

Usage:
    python analysis/livecodebench_error_analysis.py
"""

import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np


RESULTS_DIR = Path(__file__).parent.parent / "results" / "livecodebench"


def load_jsonl(filepath: Path) -> List[dict]:
    """Load JSONL file."""
    results = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    return results


def categorize_error(entry: dict) -> str:
    """Categorize the type of error."""
    if entry.get('passed', False):
        return 'PASSED'

    error = entry.get('error', '') or ''
    code = entry.get('generated_code', '') or ''

    # Check for empty/no output
    if 'Got: ' in error and error.split('Got: ')[-1].strip() == '':
        return 'NO_OUTPUT'

    # Check for syntax errors
    if 'SyntaxError' in error or 'IndentationError' in error:
        return 'SYNTAX_ERROR'

    # Check for runtime errors
    if 'NameError' in error or 'TypeError' in error or 'AttributeError' in error:
        return 'RUNTIME_ERROR'

    # Check for timeout
    if 'timeout' in error.lower() or 'time limit' in error.lower():
        return 'TIMEOUT'

    # Check for wrong answer (has output but incorrect)
    if 'Expected:' in error and 'Got:' in error:
        expected = error.split('Expected:')[1].split('Got:')[0].strip()
        got = error.split('Got:')[-1].strip()
        if got and expected != got:
            return 'WRONG_ANSWER'

    # Check if code is truncated (reasoning took too many tokens)
    if code and (code.endswith('...') or len(code.strip().split('\n')) < 3):
        return 'TRUNCATED'

    # Default
    return 'OTHER_ERROR'


def analyze_token_distribution(entries: List[dict]) -> dict:
    """Analyze token usage patterns."""
    tokens = [e.get('tokens_used', 0) for e in entries]
    passed_tokens = [e.get('tokens_used', 0) for e in entries if e.get('passed')]
    failed_tokens = [e.get('tokens_used', 0) for e in entries if not e.get('passed')]

    return {
        'mean': np.mean(tokens) if tokens else 0,
        'std': np.std(tokens) if tokens else 0,
        'max': max(tokens) if tokens else 0,
        'min': min(tokens) if tokens else 0,
        'passed_mean': np.mean(passed_tokens) if passed_tokens else 0,
        'failed_mean': np.mean(failed_tokens) if failed_tokens else 0,
    }


def estimate_reasoning_ratio(code: str) -> float:
    """Estimate the ratio of reasoning/comments to actual code."""
    if not code:
        return 0.0

    lines = code.split('\n')
    comment_lines = sum(1 for l in lines if l.strip().startswith('#') or l.strip().startswith('"""') or l.strip().startswith("'''"))
    total_lines = len([l for l in lines if l.strip()])

    if total_lines == 0:
        return 0.0
    return comment_lines / total_lines


def find_cot_catastrophe_examples(cot_entries: List[dict], direct_entries: List[dict]) -> List[dict]:
    """Find problems where CoT failed but Direct succeeded."""
    cot_by_id = {e['question_id']: e for e in cot_entries}
    direct_by_id = {e['question_id']: e for e in direct_entries}

    examples = []
    for qid in cot_by_id:
        cot = cot_by_id[qid]
        direct = direct_by_id.get(qid)

        if direct and direct.get('passed') and not cot.get('passed'):
            examples.append({
                'question_id': qid,
                'title': cot.get('title', ''),
                'difficulty': cot.get('difficulty', ''),
                'cot_error': categorize_error(cot),
                'cot_tokens': cot.get('tokens_used', 0),
                'direct_tokens': direct.get('tokens_used', 0),
                'cot_code_preview': cot.get('generated_code', '')[:500],
                'direct_code_preview': direct.get('generated_code', '')[:500],
            })

    return examples


def analyze_model(model_name: str, timestamp: str = None) -> dict:
    """Analyze a single model's LiveCodeBench results."""
    # Find the latest files for this model
    bdd_files = list(RESULTS_DIR.glob(f"bdd_{model_name}_*.jsonl"))
    cot_files = list(RESULTS_DIR.glob(f"cot_{model_name}_*.jsonl"))
    direct_files = list(RESULTS_DIR.glob(f"direct_{model_name}_*.jsonl"))

    if not cot_files or not direct_files:
        return None

    # Use latest or specified timestamp
    if timestamp:
        bdd_file = RESULTS_DIR / f"bdd_{model_name}_{timestamp}.jsonl"
        cot_file = RESULTS_DIR / f"cot_{model_name}_{timestamp}.jsonl"
        direct_file = RESULTS_DIR / f"direct_{model_name}_{timestamp}.jsonl"
    else:
        bdd_file = sorted(bdd_files)[-1] if bdd_files else None
        cot_file = sorted(cot_files)[-1]
        direct_file = sorted(direct_files)[-1]

    bdd_entries = load_jsonl(bdd_file) if bdd_file and bdd_file.exists() else []
    cot_entries = load_jsonl(cot_file)
    direct_entries = load_jsonl(direct_file)

    # Categorize errors
    cot_errors = defaultdict(int)
    direct_errors = defaultdict(int)
    bdd_errors = defaultdict(int)

    for e in cot_entries:
        cot_errors[categorize_error(e)] += 1
    for e in direct_entries:
        direct_errors[categorize_error(e)] += 1
    for e in bdd_entries:
        bdd_errors[categorize_error(e)] += 1

    # Token analysis
    cot_tokens = analyze_token_distribution(cot_entries)
    direct_tokens = analyze_token_distribution(direct_entries)

    # Find catastrophe examples
    examples = find_cot_catastrophe_examples(cot_entries, direct_entries)

    return {
        'model': model_name,
        'n_problems': len(cot_entries),
        'cot_pass_rate': cot_errors['PASSED'] / len(cot_entries) if cot_entries else 0,
        'direct_pass_rate': direct_errors['PASSED'] / len(direct_entries) if direct_entries else 0,
        'bdd_pass_rate': bdd_errors['PASSED'] / len(bdd_entries) if bdd_entries else 0,
        'cot_error_distribution': dict(cot_errors),
        'direct_error_distribution': dict(direct_errors),
        'bdd_error_distribution': dict(bdd_errors),
        'cot_tokens': cot_tokens,
        'direct_tokens': direct_tokens,
        'catastrophe_examples': examples[:5],  # Top 5 examples
    }


def generate_report(results: List[dict], output_dir: Path):
    """Generate analysis report and figures."""
    output_dir.mkdir(exist_ok=True)

    # Figure 1: Error distribution comparison for GPT-4o
    gpt4o_results = next((r for r in results if 'gpt-4o' in r['model']), None)
    if gpt4o_results:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        for idx, (condition, errors) in enumerate([
            ('TCGP', gpt4o_results['bdd_error_distribution']),
            ('CoT', gpt4o_results['cot_error_distribution']),
            ('Direct', gpt4o_results['direct_error_distribution'])
        ]):
            categories = list(errors.keys())
            values = list(errors.values())
            colors = ['green' if c == 'PASSED' else 'red' if c == 'NO_OUTPUT' else 'orange' for c in categories]

            axes[idx].bar(categories, values, color=colors)
            axes[idx].set_title(f'{condition} Error Distribution')
            axes[idx].set_xlabel('Error Type')
            axes[idx].set_ylabel('Count')
            axes[idx].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig(output_dir / 'gpt4o_error_distribution.png', dpi=150)
        plt.savefig(output_dir / 'gpt4o_error_distribution.pdf')
        plt.close()

    # Figure 2: Token usage comparison
    fig, ax = plt.subplots(figsize=(10, 6))

    models = [r['model'] for r in results]
    cot_tokens = [r['cot_tokens']['mean'] for r in results]
    direct_tokens = [r['direct_tokens']['mean'] for r in results]

    x = np.arange(len(models))
    width = 0.35

    ax.bar(x - width/2, cot_tokens, width, label='CoT', color='blue', alpha=0.7)
    ax.bar(x + width/2, direct_tokens, width, label='Direct', color='green', alpha=0.7)

    ax.set_xlabel('Model')
    ax.set_ylabel('Average Tokens')
    ax.set_title('Token Usage: CoT vs Direct on LiveCodeBench')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_dir / 'token_comparison.png', dpi=150)
    plt.savefig(output_dir / 'token_comparison.pdf')
    plt.close()

    # Write text report
    with open(output_dir / 'error_analysis_report.md', 'w') as f:
        f.write("# LiveCodeBench Error Analysis Report\n\n")

        for r in results:
            f.write(f"## {r['model']}\n\n")
            f.write(f"- **Problems**: {r['n_problems']}\n")
            f.write(f"- **Pass Rates**: TCGP={r['bdd_pass_rate']:.1%}, CoT={r['cot_pass_rate']:.1%}, Direct={r['direct_pass_rate']:.1%}\n\n")

            f.write("### Error Distribution (CoT)\n\n")
            for error_type, count in sorted(r['cot_error_distribution'].items(), key=lambda x: -x[1]):
                pct = count / r['n_problems'] * 100
                f.write(f"- {error_type}: {count} ({pct:.1f}%)\n")

            f.write("\n### CoT Catastrophe Examples\n\n")
            if r['catastrophe_examples']:
                f.write("Problems where CoT failed but Direct succeeded:\n\n")
                for ex in r['catastrophe_examples'][:3]:
                    f.write(f"**{ex['title']}** ({ex['difficulty']})\n")
                    f.write(f"- CoT Error: {ex['cot_error']}\n")
                    f.write(f"- CoT Tokens: {ex['cot_tokens']}, Direct Tokens: {ex['direct_tokens']}\n\n")

            f.write("\n---\n\n")

    print(f"Report saved to {output_dir / 'error_analysis_report.md'}")


def main():
    print("LiveCodeBench Error Analysis")
    print("=" * 50)

    # Models to analyze (focus on those with CoT catastrophe)
    models_to_analyze = [
        ('gpt-4o', '20260402_164225'),
        ('gpt-5.3-codex', '20260402_215204'),
        ('gpt-35-turbo', '20260402_180436'),
        ('gpt-4.1', '20260403_114332'),
        ('gemini-3.1-flash-lite-preview', '20260402_184844'),
    ]

    results = []
    for model, timestamp in models_to_analyze:
        print(f"\nAnalyzing {model}...")
        result = analyze_model(model, timestamp)
        if result:
            results.append(result)
            print(f"  CoT: {result['cot_pass_rate']:.1%}, Direct: {result['direct_pass_rate']:.1%}")
            print(f"  CoT avg tokens: {result['cot_tokens']['mean']:.0f}")
            print(f"  Top CoT error: {max(result['cot_error_distribution'].items(), key=lambda x: x[1] if x[0] != 'PASSED' else 0)}")

    # Generate report
    output_dir = RESULTS_DIR.parent / "analysis_output"
    generate_report(results, output_dir)

    # Save raw results as JSON
    with open(output_dir / 'error_analysis_results.json', 'w') as f:
        # Convert to JSON-serializable
        serializable = []
        for r in results:
            r_copy = r.copy()
            r_copy['cot_tokens'] = dict(r['cot_tokens'])
            r_copy['direct_tokens'] = dict(r['direct_tokens'])
            serializable.append(r_copy)
        json.dump(serializable, f, indent=2)

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
