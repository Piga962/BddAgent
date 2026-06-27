#!/usr/bin/env python3
"""
Multi-Model Ablation Study Runner

Runs BDD vs No-BDD comparison across multiple models:
- GPT-4.1 (Azure OpenAI)
- Gemini 2.5 Flash (Google)
- Gemini 2.5 Pro (Google)
"""

import os
import sys
import json
import time
import random
import requests
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent))


@dataclass
class TestResult:
    """Result for a single test case."""
    namespace: str
    model: str
    mode: str
    with_bdd: bool
    success: bool
    generated_code: str
    duration_seconds: float
    tokens_used: int
    error: Optional[str] = None


class GeminiGenerator:
    """Code generator using Gemini models."""

    def __init__(self, model: str = "gemini-2.5-flash"):
        self.model = model
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.base_url = "https://generativelanguage.googleapis.com/v1beta"

    def _call_api(self, prompt: str, max_tokens: int = 1000) -> Tuple[str, int]:
        """Call Gemini API."""
        url = f"{self.base_url}/models/{self.model}:generateContent?key={self.api_key}"

        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "maxOutputTokens": max_tokens,
                "temperature": 0.2
            }
        }

        response = requests.post(url, json=payload, timeout=60)

        if response.status_code != 200:
            raise Exception(f"Gemini API error: {response.status_code} - {response.text[:200]}")

        result = response.json()
        text = result.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', '')

        # Estimate tokens (Gemini doesn't always return token count)
        usage = result.get('usageMetadata', {})
        tokens = usage.get('totalTokenCount', len(prompt.split()) + len(text.split()))

        return text, tokens

    def generate_with_bdd(self, requirements: str, context: str = "") -> Tuple[str, float, int]:
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

        bdd_scenarios, bdd_tokens = self._call_api(bdd_prompt, 500)

        # Step 2: Generate code using BDD scenarios
        code_prompt = f"""Generate a Python function body based on these requirements and BDD scenarios.

Requirements: {requirements}
{f'Context: {context}' if context else ''}

BDD Test Scenarios (your code must satisfy these):
{bdd_scenarios}

IMPORTANT:
- Return ONLY the function body (no 'def' line)
- Use 4-space indentation
- The code must pass all BDD scenarios above
"""

        code, code_tokens = self._call_api(code_prompt, 1000)

        duration = time.time() - start
        total_tokens = bdd_tokens + code_tokens

        return code, duration, total_tokens

    def generate_without_bdd(self, requirements: str, context: str = "") -> Tuple[str, float, int]:
        """Generate code WITHOUT BDD methodology."""
        start = time.time()

        prompt = f"""Generate a Python function body based on these requirements.

Requirements: {requirements}
{f'Context: {context}' if context else ''}

IMPORTANT:
- Return ONLY the function body (no 'def' line)
- Use 4-space indentation
- Write clean, working Python code
"""

        code, tokens = self._call_api(prompt, 1000)
        duration = time.time() - start

        return code, duration, tokens


class AzureGPTGenerator:
    """Code generator using Azure OpenAI GPT-4.1."""

    def __init__(self, model: str = "gpt-4.1"):
        self.model = model
        from openai import OpenAI

        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        api_key = os.getenv("AZURE_OPENAI_KEY")

        self.client = OpenAI(base_url=endpoint, api_key=api_key)

    def generate_with_bdd(self, requirements: str, context: str = "") -> Tuple[str, float, int]:
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
- Use 4-space indentation
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
        tokens = bdd_response.usage.total_tokens + code_response.usage.total_tokens

        return code, duration, tokens

    def generate_without_bdd(self, requirements: str, context: str = "") -> Tuple[str, float, int]:
        """Generate code WITHOUT BDD methodology."""
        start = time.time()

        prompt = f"""Generate a Python function body based on these requirements.

Requirements: {requirements}
{f'Context: {context}' if context else ''}

IMPORTANT:
- Return ONLY the function body (no 'def' line)
- Use 4-space indentation
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
    """Clean up generated code to extract function body."""
    import re
    code = re.sub(r'```python\s*\n?', '', code)
    code = re.sub(r'```\s*\n?', '', code)

    lines = code.strip().split('\n')
    result_lines = []
    in_function = False

    for line in lines:
        stripped = line.strip()

        if stripped.startswith('def ') and '(' in stripped:
            in_function = True
            continue

        if in_function or not stripped.startswith('def '):
            if stripped and not line.startswith(' '):
                result_lines.append('    ' + stripped)
            else:
                result_lines.append(line)

    result = '\n'.join(result_lines).strip()
    if not result:
        return "    pass"
    return result


def run_multimodel_ablation(
    dataset_path: str,
    models: List[str] = None,
    mode: str = "local_file_completion",
    sample_size: int = 30,
    seed: int = 42
):
    """Run ablation study across multiple models."""
    if models is None:
        models = ["gpt-4.1", "gemini-2.5-flash", "gemini-2.5-pro"]

    print("="*70)
    print("MULTI-MODEL ABLATION STUDY: BDD vs No-BDD")
    print("="*70)
    print(f"Models: {models}")
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

    if len(tests) > sample_size:
        tests = random.sample(tests, sample_size)

    print(f"Loaded {len(tests)} test cases")

    # Initialize generators
    generators = {}
    for model in models:
        if model.startswith("gpt"):
            generators[model] = AzureGPTGenerator(model)
        elif model.startswith("gemini"):
            generators[model] = GeminiGenerator(model)
        else:
            print(f"Unknown model type: {model}")
            continue

    # Results storage
    all_results = {model: {"bdd": [], "no_bdd": []} for model in generators}

    # Output directory
    output_dir = Path("results/multimodel_ablation")
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for i, test in enumerate(tests):
        namespace = test['namespace']
        requirements = test['requirements']
        context = test['context']

        print(f"\n[{i+1}/{len(tests)}] {namespace}")

        for model_name, generator in generators.items():
            # WITH BDD
            try:
                code_bdd, dur_bdd, tok_bdd = generator.generate_with_bdd(requirements, context)
                code_bdd = extract_function_body(code_bdd)
                all_results[model_name]["bdd"].append(TestResult(
                    namespace=namespace,
                    model=model_name,
                    mode=mode,
                    with_bdd=True,
                    success=True,
                    generated_code=code_bdd,
                    duration_seconds=dur_bdd,
                    tokens_used=tok_bdd
                ))
                print(f"  {model_name} BDD: {dur_bdd:.1f}s")
            except Exception as e:
                all_results[model_name]["bdd"].append(TestResult(
                    namespace=namespace,
                    model=model_name,
                    mode=mode,
                    with_bdd=True,
                    success=False,
                    generated_code="",
                    duration_seconds=0,
                    tokens_used=0,
                    error=str(e)[:200]
                ))
                print(f"  {model_name} BDD: ERROR - {str(e)[:50]}")

            # WITHOUT BDD
            try:
                code_no_bdd, dur_no_bdd, tok_no_bdd = generator.generate_without_bdd(requirements, context)
                code_no_bdd = extract_function_body(code_no_bdd)
                all_results[model_name]["no_bdd"].append(TestResult(
                    namespace=namespace,
                    model=model_name,
                    mode=mode,
                    with_bdd=False,
                    success=True,
                    generated_code=code_no_bdd,
                    duration_seconds=dur_no_bdd,
                    tokens_used=tok_no_bdd
                ))
                print(f"  {model_name} No-BDD: {dur_no_bdd:.1f}s")
            except Exception as e:
                all_results[model_name]["no_bdd"].append(TestResult(
                    namespace=namespace,
                    model=model_name,
                    mode=mode,
                    with_bdd=False,
                    success=False,
                    generated_code="",
                    duration_seconds=0,
                    tokens_used=0,
                    error=str(e)[:200]
                ))
                print(f"  {model_name} No-BDD: ERROR - {str(e)[:50]}")

        # Save incremental results
        if (i + 1) % 5 == 0:
            save_all_results(all_results, output_dir, mode, timestamp)

    # Final save
    save_all_results(all_results, output_dir, mode, timestamp)

    # Print summary
    print_multimodel_summary(all_results, mode)

    return all_results


def save_all_results(all_results: dict, output_dir: Path, mode: str, timestamp: str):
    """Save results for all models."""
    for model_name, results in all_results.items():
        model_safe = model_name.replace(".", "_").replace("-", "_")

        # BDD results
        bdd_path = output_dir / f"{model_safe}_bdd_{mode}_{timestamp}.jsonl"
        with open(bdd_path, 'w') as f:
            for r in results["bdd"]:
                f.write(json.dumps(asdict(r)) + '\n')

        # No-BDD results
        no_bdd_path = output_dir / f"{model_safe}_no_bdd_{mode}_{timestamp}.jsonl"
        with open(no_bdd_path, 'w') as f:
            for r in results["no_bdd"]:
                f.write(json.dumps(asdict(r)) + '\n')


def print_multimodel_summary(all_results: dict, mode: str):
    """Print comparison summary for all models."""
    print("\n" + "="*80)
    print("MULTI-MODEL ABLATION RESULTS")
    print("="*80)
    print(f"\nMode: {mode}")

    print("\n" + "-"*80)
    print(f"{'Model':<20} {'BDD Success':<15} {'No-BDD Success':<15} {'BDD Avg Time':<15} {'No-BDD Time':<15}")
    print("-"*80)

    for model_name, results in all_results.items():
        bdd_results = results["bdd"]
        no_bdd_results = results["no_bdd"]

        n_bdd = len(bdd_results)
        n_no_bdd = len(no_bdd_results)

        bdd_success = sum(1 for r in bdd_results if r.success)
        no_bdd_success = sum(1 for r in no_bdd_results if r.success)

        bdd_time = sum(r.duration_seconds for r in bdd_results if r.success)
        no_bdd_time = sum(r.duration_seconds for r in no_bdd_results if r.success)

        bdd_rate = f"{bdd_success}/{n_bdd}" if n_bdd else "N/A"
        no_bdd_rate = f"{no_bdd_success}/{n_no_bdd}" if n_no_bdd else "N/A"

        bdd_avg = f"{bdd_time/max(bdd_success,1):.2f}s" if bdd_success else "N/A"
        no_bdd_avg = f"{no_bdd_time/max(no_bdd_success,1):.2f}s" if no_bdd_success else "N/A"

        print(f"{model_name:<20} {bdd_rate:<15} {no_bdd_rate:<15} {bdd_avg:<15} {no_bdd_avg:<15}")

    print("-"*80)
    print("\nResults saved to: results/multimodel_ablation/")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run multi-model BDD vs No-BDD ablation study")
    parser.add_argument("--dataset", "-d", default="data/DevEval-main/Experiments/prompt/LM_prompt_elements.jsonl")
    parser.add_argument("--mode", "-m", default="local_file_completion",
                       choices=["without_context", "local_file_completion", "local_file_infiling"])
    parser.add_argument("--samples", "-n", type=int, default=30)
    parser.add_argument("--seed", "-s", type=int, default=42)
    parser.add_argument("--models", nargs="+", default=["gpt-4.1", "gemini-2.5-flash"])

    args = parser.parse_args()

    run_multimodel_ablation(
        dataset_path=args.dataset,
        models=args.models,
        mode=args.mode,
        sample_size=args.samples,
        seed=args.seed
    )
