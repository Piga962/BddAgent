#!/usr/bin/env python3
"""
Ablation Study Runner for BDD vs No-BDD Comparison

This script runs the critical ablation study needed for publication:
- Same model (GPT-4.1)
- Same architecture (multi-agent)
- WITH vs WITHOUT BDD integration
"""

import os
import sys
import json
import time
import random
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional
from dotenv import load_dotenv

load_dotenv()

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from openai import OpenAI


@dataclass
class TestResult:
    """Result for a single test case."""
    namespace: str
    mode: str
    with_bdd: bool
    success: bool
    generated_code: str
    duration_seconds: float
    tokens_used: int
    error: Optional[str] = None


class SimpleCodeGenerator:
    """
    Simplified code generator for ablation study.
    Tests BDD vs No-BDD with identical model and prompting.
    """

    def __init__(self, model: str = "gpt-4.1"):
        self.model = model
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        api_key = os.getenv("AZURE_OPENAI_KEY")

        self.client = OpenAI(base_url=endpoint, api_key=api_key)

    def generate_with_bdd(self, requirements: str, context: str = "") -> tuple:
        """Generate code WITH BDD methodology."""
        start = time.time()

        # Step 1: Generate BDD scenarios
        bdd_prompt = f"""Analyze these requirements and generate BDD test scenarios.

Requirements: {requirements}
{f'Context: {context}' if context else ''}

Generate 2-3 Given-When-Then scenarios that define the expected behavior.
Format:
Scenario 1: [name]
Given [precondition]
When [action]
Then [expected result]
"""

        bdd_response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": bdd_prompt}],
            max_tokens=500,
            temperature=0.2
        )
        bdd_scenarios = bdd_response.choices[0].message.content

        # Step 2: Generate code using BDD scenarios
        code_prompt = f"""Generate a Python function body based on these requirements and BDD scenarios.

Requirements: {requirements}
{f'Context: {context}' if context else ''}

BDD Test Scenarios (your code must satisfy these):
{bdd_scenarios}

IMPORTANT:
- Return ONLY the function body (no 'def' line)
- Use 4-space indentation for the base level
- CRITICAL: elif/else must be at the SAME indentation as their matching if
- CRITICAL: except/finally must be at the SAME indentation as their matching try
- Example of CORRECT structure:
  if condition:
      return x
  elif other:      # <- same level as if, NOT indented under if
      return y
- The code must pass all BDD scenarios above
"""

        code_response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": code_prompt}],
            max_tokens=1000,
            temperature=0.2
        )

        code = code_response.choices[0].message.content
        duration = time.time() - start
        tokens = (bdd_response.usage.total_tokens + code_response.usage.total_tokens)

        return code, duration, tokens

    def generate_without_bdd(self, requirements: str, context: str = "") -> tuple:
        """Generate code WITHOUT BDD methodology (direct prompting)."""
        start = time.time()

        prompt = f"""Generate a Python function body based on these requirements.

Requirements: {requirements}
{f'Context: {context}' if context else ''}

IMPORTANT:
- Return ONLY the function body (no 'def' line)
- Use 4-space indentation for the base level
- CRITICAL: elif/else must be at the SAME indentation as their matching if
- CRITICAL: except/finally must be at the SAME indentation as their matching try
- Example of CORRECT structure:
  if condition:
      return x
  elif other:      # <- same level as if, NOT indented under if
      return y
- Write clean, working Python code
"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000,
            temperature=0.2
        )

        code = response.choices[0].message.content
        duration = time.time() - start
        tokens = response.usage.total_tokens

        return code, duration, tokens


def extract_function_body(code: str) -> str:
    """Clean up generated code to extract function body with proper indentation.

    Handles cases where the LLM generates code with misaligned control structures
    (e.g., elif/else indented under if instead of at the same level).
    """
    import re
    import ast
    import textwrap

    # Remove markdown code blocks
    code = re.sub(r'```python\s*\n?', '', code)
    code = re.sub(r'```\s*\n?', '', code)
    code = code.strip()

    if not code:
        return "    pass"

    lines = code.split('\n')

    # Skip any def line if present
    start_idx = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith('def ') and '(' in stripped:
            start_idx = i + 1
            break

    lines = lines[start_idx:]

    if not lines:
        return "    pass"

    # Dedent completely first
    dedented = textwrap.dedent('\n'.join(lines))
    lines = dedented.split('\n')

    # Fix control structure alignment: elif/else must match if, except/finally must match try
    control_keywords = {'elif', 'else', 'except', 'finally'}
    block_starters = {'if', 'try', 'for', 'while', 'with', 'match'}

    fixed_lines = []
    indent_stack = []  # Track (keyword, indent_level) for block matching

    for line in lines:
        stripped = line.strip()
        if not stripped:
            fixed_lines.append('')
            continue

        current_indent = len(line) - len(line.lstrip())
        first_word = stripped.split()[0].rstrip(':') if stripped.split() else ''

        # Check if this is a continuation control keyword (elif, else, except, finally)
        if first_word in control_keywords:
            # Find the matching opener's indentation
            # Pop until we find a compatible opener at same or lower level
            while indent_stack:
                opener, opener_indent = indent_stack[-1]
                if opener_indent <= current_indent:
                    # Check if this is a valid pairing
                    valid_pairs = {
                        'elif': {'if', 'elif'},
                        'else': {'if', 'elif', 'for', 'while', 'try', 'except'},
                        'except': {'try', 'except'},
                        'finally': {'try', 'except', 'else'},
                    }
                    if opener in valid_pairs.get(first_word, set()):
                        # Use the opener's indentation
                        current_indent = opener_indent
                        break
                indent_stack.pop()

        # Track block starters
        if first_word in block_starters or first_word in {'elif', 'except'}:
            indent_stack.append((first_word, current_indent))
        elif first_word == 'else' and ':' in stripped:
            indent_stack.append(('else', current_indent))

        # Reconstruct line with 4-space base indentation
        fixed_lines.append(' ' * (4 + current_indent) + stripped)

    result = '\n'.join(fixed_lines)

    # Validate Python syntax
    try:
        test_code = "def _test_func():\n" + result
        ast.parse(test_code)
    except SyntaxError:
        # If still invalid, try simpler approach: just use 4-space base indent
        result = '\n'.join('    ' + line.strip() if line.strip() else '' for line in lines)

    if not result.strip():
        return "    pass"

    return result


def run_ablation_study(
    dataset_path: str,
    mode: str = "local_file_completion",
    sample_size: int = 50,
    seed: int = 42
):
    """Run the ablation study."""
    print("="*70)
    print("ABLATION STUDY: BDD vs No-BDD")
    print("="*70)
    print(f"Model: gpt-4.1")
    print(f"Mode: {mode}")
    print(f"Sample size: {sample_size}")
    print(f"Seed: {seed}")
    print("="*70)

    # Load dataset
    random.seed(seed)
    tests = []
    with open(dataset_path, 'r') as f:
        for line in f:
            test = json.loads(line)
            processed = {
                "namespace": test['namespace'],
                "requirements": test['input_code'],
            }
            if mode == 'local_file_completion':
                processed['context'] = test.get('contexts_above', '')
            elif mode == 'local_file_infiling':
                processed['context'] = test.get('contexts_above', '') + '\n' + test.get('contexts_below', '')
            else:
                processed['context'] = ''
            tests.append(processed)

    # Sample
    if len(tests) > sample_size:
        tests = random.sample(tests, sample_size)

    print(f"Loaded {len(tests)} test cases")

    # Initialize generator
    generator = SimpleCodeGenerator()

    # Results
    results_with_bdd = []
    results_without_bdd = []

    # Output directory
    output_dir = Path("results/ablation")
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for i, test in enumerate(tests):
        namespace = test['namespace']
        requirements = test['requirements']
        context = test['context']

        print(f"\n[{i+1}/{len(tests)}] {namespace}")

        # WITH BDD
        try:
            code_bdd, dur_bdd, tok_bdd = generator.generate_with_bdd(requirements, context)
            code_bdd = extract_function_body(code_bdd)
            results_with_bdd.append(TestResult(
                namespace=namespace,
                mode=mode,
                with_bdd=True,
                success=True,
                generated_code=code_bdd,
                duration_seconds=dur_bdd,
                tokens_used=tok_bdd
            ))
            print(f"  BDD: {dur_bdd:.1f}s, {tok_bdd} tokens")
        except Exception as e:
            results_with_bdd.append(TestResult(
                namespace=namespace,
                mode=mode,
                with_bdd=True,
                success=False,
                generated_code="",
                duration_seconds=0,
                tokens_used=0,
                error=str(e)
            ))
            print(f"  BDD: ERROR - {str(e)[:50]}")

        # WITHOUT BDD
        try:
            code_no_bdd, dur_no_bdd, tok_no_bdd = generator.generate_without_bdd(requirements, context)
            code_no_bdd = extract_function_body(code_no_bdd)
            results_without_bdd.append(TestResult(
                namespace=namespace,
                mode=mode,
                with_bdd=False,
                success=True,
                generated_code=code_no_bdd,
                duration_seconds=dur_no_bdd,
                tokens_used=tok_no_bdd
            ))
            print(f"  No-BDD: {dur_no_bdd:.1f}s, {tok_no_bdd} tokens")
        except Exception as e:
            results_without_bdd.append(TestResult(
                namespace=namespace,
                mode=mode,
                with_bdd=False,
                success=False,
                generated_code="",
                duration_seconds=0,
                tokens_used=0,
                error=str(e)
            ))
            print(f"  No-BDD: ERROR - {str(e)[:50]}")

        # Save incremental results
        if (i + 1) % 10 == 0:
            save_results(results_with_bdd, output_dir / f"bdd_{mode}_{timestamp}.jsonl")
            save_results(results_without_bdd, output_dir / f"no_bdd_{mode}_{timestamp}.jsonl")

    # Final save
    save_results(results_with_bdd, output_dir / f"bdd_{mode}_{timestamp}.jsonl")
    save_results(results_without_bdd, output_dir / f"no_bdd_{mode}_{timestamp}.jsonl")

    # Print summary
    print_summary(results_with_bdd, results_without_bdd, mode)

    return results_with_bdd, results_without_bdd


def save_results(results: List[TestResult], path: Path):
    """Save results to JSONL."""
    with open(path, 'w') as f:
        for r in results:
            f.write(json.dumps(asdict(r)) + '\n')


def print_summary(with_bdd: List[TestResult], without_bdd: List[TestResult], mode: str):
    """Print comparison summary."""
    print("\n" + "="*70)
    print("ABLATION STUDY RESULTS")
    print("="*70)

    n = len(with_bdd)

    bdd_success = sum(1 for r in with_bdd if r.success)
    no_bdd_success = sum(1 for r in without_bdd if r.success)

    bdd_time = sum(r.duration_seconds for r in with_bdd if r.success)
    no_bdd_time = sum(r.duration_seconds for r in without_bdd if r.success)

    bdd_tokens = sum(r.tokens_used for r in with_bdd if r.success)
    no_bdd_tokens = sum(r.tokens_used for r in without_bdd if r.success)

    print(f"\nMode: {mode}")
    print(f"Total tests: {n}")
    print("-"*50)
    print(f"{'Metric':<25} {'WITH BDD':<15} {'WITHOUT BDD':<15}")
    print("-"*50)
    bdd_rate = f"{bdd_success/n:.1%}"
    no_bdd_rate = f"{no_bdd_success/n:.1%}"
    print(f"{'API Success Rate':<25} {bdd_rate:<15} {no_bdd_rate:<15}")
    bdd_avg_time = f"{bdd_time/max(bdd_success,1):.2f}s"
    no_bdd_avg_time = f"{no_bdd_time/max(no_bdd_success,1):.2f}s"
    print(f"{'Avg Time (s)':<25} {bdd_avg_time:<15} {no_bdd_avg_time:<15}")
    bdd_avg_tok = f"{bdd_tokens/max(bdd_success,1):.0f}"
    no_bdd_avg_tok = f"{no_bdd_tokens/max(no_bdd_success,1):.0f}"
    print(f"{'Avg Tokens':<25} {bdd_avg_tok:<15} {no_bdd_avg_tok:<15}")
    print("-"*50)

    # Note about Pass@1 evaluation
    print("\nNOTE: These are API success rates, not Pass@1.")
    print("Pass@1 requires running the DevEval evaluation harness.")
    print(f"Results saved to: results/ablation/")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run BDD vs No-BDD ablation study")
    parser.add_argument("--dataset", "-d", default="data/DevEval-main/Experiments/prompt/LM_prompt_elements.jsonl")
    parser.add_argument("--mode", "-m", default="local_file_completion",
                       choices=["without_context", "local_file_completion", "local_file_infiling"])
    parser.add_argument("--samples", "-n", type=int, default=50)
    parser.add_argument("--seed", "-s", type=int, default=42)

    args = parser.parse_args()

    run_ablation_study(
        dataset_path=args.dataset,
        mode=args.mode,
        sample_size=args.samples,
        seed=args.seed
    )
