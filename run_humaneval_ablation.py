#!/usr/bin/env python3
"""
HumanEval Ablation Study: TCGP vs No-TCGP Code Generation

This script evaluates the impact of TCGP (Behavior-Driven Development) prompting
on code generation quality using the HumanEval benchmark (164 problems).

Metrics computed:
- Pass@1 with 95% confidence intervals
- Statistical comparison between conditions
"""

import json
import time
import random
import ast
import re
import math
import traceback
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple, List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import multiprocessing
from dotenv import load_dotenv
import os

load_dotenv()


def get_client(provider: str = "azure"):
    """Get API client based on provider."""
    if provider == "azure":
        from openai import OpenAI
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        # Ensure endpoint ends with trailing slash
        if not endpoint.endswith("/"):
            endpoint = endpoint + "/"
        return OpenAI(
            base_url=endpoint,
            api_key=os.getenv("AZURE_OPENAI_KEY")
        )
    elif provider == "openai":
        from openai import OpenAI
        return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    elif provider == "anthropic":
        import anthropic
        return anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    elif provider == "gemini":
        import google.generativeai as genai
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        return genai
    else:
        raise ValueError(f"Unknown provider: {provider}")


class HumanEvalBddAgent:
    """TCGP-driven code generation agent for HumanEval."""

    def __init__(self, model: str = "gpt-4.1", provider: str = "azure"):
        self.client = get_client(provider)
        self.model = model
        self.provider = provider

    def _call_llm(self, prompt: str, temperature: float = 0.2, max_tokens: int = 1000) -> Tuple[str, int]:
        """Call LLM and return response with token count."""
        if self.provider == "anthropic":
            response = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text, response.usage.input_tokens + response.usage.output_tokens
        elif self.provider == "gemini":
            import google.generativeai as genai
            model = genai.GenerativeModel(self.model)
            generation_config = genai.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens
            )
            response = model.generate_content(prompt, generation_config=generation_config)
            # Estimate tokens (Gemini doesn't always return exact counts)
            tokens = len(prompt.split()) + len(response.text.split()) if response.text else 0
            return response.text if response.text else "", tokens
        else:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature
            )
            return response.choices[0].message.content, response.usage.total_tokens

    def generate_with_tcgp(self, prompt: str, entry_point: str) -> Tuple[str, float, int, str]:
        """Generate code with TCGP methodology.

        Returns: (code, duration, tokens, bdd_scenarios)
        """
        start = time.time()
        total_tokens = 0

        # Step 1: Generate concrete TCGP test scenarios
        bdd_prompt = f"""Analyze this function signature and docstring, then generate CONCRETE test scenarios.

{prompt}

Generate 3-4 specific test scenarios with exact inputs and expected outputs.

Format:
```
Scenario 1: [description]
  Input: [actual value like [1, 2, 3] or "hello"]
  Expected: [actual expected result]

Scenario 2: [description]
  Input: [actual value]
  Expected: [actual expected result]

Edge Case:
  Input: [edge case value]
  Expected: [expected result]
```

Use REAL values, not placeholders. Extract examples from the docstring if available.
"""
        bdd_scenarios, tokens = self._call_llm(bdd_prompt, temperature=0.3, max_tokens=600)
        total_tokens += tokens

        # Step 2: Generate code with TCGP context
        code_prompt = f"""Complete this Python function. The function signature and docstring are provided.

{prompt}

Your implementation must pass these test scenarios:
{bdd_scenarios}

REQUIREMENTS:
1. Return ONLY the function body (the code that goes inside the function)
2. Do NOT include the function signature (no 'def' line)
3. Use proper 4-space indentation
4. Every code path must return a value
5. Handle edge cases

Return ONLY the implementation code, no explanations.
"""
        code, tokens = self._call_llm(code_prompt, temperature=0.2, max_tokens=800)
        total_tokens += tokens

        # Clean up the generated code
        code = self._extract_body(code)
        duration = time.time() - start

        return code, duration, total_tokens, bdd_scenarios

    def generate_without_tcgp(self, prompt: str, entry_point: str) -> Tuple[str, float, int]:
        """Generate code without TCGP methodology.

        Returns: (code, duration, tokens)
        """
        start = time.time()

        code_prompt = f"""Complete this Python function. The function signature and docstring are provided.

{prompt}

REQUIREMENTS:
1. Return ONLY the function body (the code that goes inside the function)
2. Do NOT include the function signature (no 'def' line)
3. Use proper 4-space indentation
4. Every code path must return a value
5. Handle edge cases (empty inputs, None values, etc.)

Think step by step about what the function should do, then provide the implementation.
Return ONLY the implementation code, no explanations.
"""
        code, tokens = self._call_llm(code_prompt, temperature=0.2, max_tokens=800)
        code = self._extract_body(code)
        duration = time.time() - start

        return code, duration, tokens

    def _extract_body(self, code: str) -> str:
        """Extract and clean function body from LLM response."""
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
            if stripped.startswith('def ') and '(' in stripped and ':' in stripped:
                start_idx = i + 1
                break

        lines = lines[start_idx:]
        if not lines:
            return "    pass"

        # Ensure proper indentation (4 spaces base)
        result_lines = []
        for line in lines:
            stripped = line.strip()
            if not stripped:
                result_lines.append('')
                continue
            # Calculate current indent
            current_indent = len(line) - len(line.lstrip())
            # Ensure at least 4 spaces
            if current_indent < 4:
                result_lines.append('    ' + stripped)
            else:
                result_lines.append(line)

        result = '\n'.join(result_lines)

        # Validate syntax
        try:
            ast.parse("def _test():\n" + result)
        except SyntaxError:
            # Fallback: uniform 4-space indent
            result = '\n'.join('    ' + line.strip() if line.strip() else '' for line in lines)

        return result if result.strip() else "    pass"


def execute_test(full_code: str, test_code: str, entry_point: str, timeout: int = 5) -> Tuple[bool, str]:
    """Execute generated code with test cases.

    Returns: (passed, error_message)
    """
    # Combine code and tests
    exec_code = full_code + "\n\n" + test_code + f"\n\ncheck({entry_point})"

    def run_code():
        exec_globals = {}
        exec(exec_code, exec_globals)
        return True, ""

    # Run with timeout using multiprocessing
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(run_code)
            try:
                return future.result(timeout=timeout)
            except TimeoutError:
                return False, "Timeout"
    except AssertionError as e:
        return False, f"AssertionError: {e}"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def wilson_ci(successes: int, total: int, confidence: float = 0.95) -> Tuple[float, float]:
    """Compute Wilson score confidence interval."""
    if total == 0:
        return 0.0, 0.0

    z = 1.96 if confidence == 0.95 else 2.576  # 95% or 99%
    p = successes / total

    denominator = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denominator
    margin = z * math.sqrt((p * (1 - p) + z**2 / (4 * total)) / total) / denominator

    return max(0, center - margin), min(1, center + margin)


def cohens_d(n1: int, pass1: int, n2: int, pass2: int) -> float:
    """Compute Cohen's d effect size for two proportions."""
    p1 = pass1 / n1 if n1 > 0 else 0
    p2 = pass2 / n2 if n2 > 0 else 0

    # Pooled standard deviation for proportions
    pooled_var = (p1 * (1 - p1) + p2 * (1 - p2)) / 2
    pooled_std = math.sqrt(pooled_var) if pooled_var > 0 else 1

    return (p1 - p2) / pooled_std if pooled_std > 0 else 0


def run_humaneval_ablation(
    data_path: str = "data/humaneval/humaneval.jsonl",
    model: str = "gpt-4.1",
    provider: str = "azure",
    sample_size: Optional[int] = None,
    seed: int = 42,
    output_dir: str = "results/humaneval_ablation"
):
    """Run HumanEval ablation study comparing TCGP vs No-TCGP."""

    print("=" * 70, flush=True)
    print("HUMANEVAL ABLATION STUDY: TCGP vs No-TCGP", flush=True)
    print("=" * 70, flush=True)
    print(f"Model: {model}", flush=True)
    print(f"Provider: {provider}", flush=True)
    print(f"Seed: {seed}", flush=True)

    # Load dataset
    random.seed(seed)
    problems = []
    with open(data_path, 'r') as f:
        for line in f:
            problems.append(json.loads(line))

    if sample_size and sample_size < len(problems):
        problems = random.sample(problems, sample_size)
        print(f"Sample size: {sample_size}")
    else:
        print(f"Using all {len(problems)} problems")

    print("=" * 70)

    # Initialize agent
    agent = HumanEvalBddAgent(model=model, provider=provider)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Results storage
    bdd_results = []
    no_bdd_results = []
    bdd_passed = 0
    no_bdd_passed = 0

    for i, problem in enumerate(problems):
        task_id = problem['task_id']
        prompt = problem['prompt']
        test = problem['test']
        entry_point = problem['entry_point']

        print(f"\n[{i+1}/{len(problems)}] {task_id}", flush=True)

        # Generate with TCGP
        try:
            code, duration, tokens, bdd_scenarios = agent.generate_with_tcgp(prompt, entry_point)
            full_code = prompt + code
            passed, error = execute_test(full_code, test, entry_point)

            if passed:
                bdd_passed += 1
                print(f"  TCGP: PASS ({duration:.1f}s)")
            else:
                print(f"  TCGP: FAIL ({duration:.1f}s) - {error[:50]}")

            bdd_results.append({
                "task_id": task_id,
                "with_bdd": True,
                "passed": passed,
                "generated_code": code,
                "full_code": full_code,
                "bdd_scenarios": bdd_scenarios,
                "duration_seconds": duration,
                "tokens_used": tokens,
                "error": error if not passed else None
            })
        except Exception as e:
            print(f"  TCGP: ERROR - {e}")
            bdd_results.append({
                "task_id": task_id,
                "with_bdd": True,
                "passed": False,
                "error": str(e)
            })

        # Generate without TCGP
        try:
            code, duration, tokens = agent.generate_without_tcgp(prompt, entry_point)
            full_code = prompt + code
            passed, error = execute_test(full_code, test, entry_point)

            if passed:
                no_bdd_passed += 1
                print(f"  No-TCGP: PASS ({duration:.1f}s)")
            else:
                print(f"  No-TCGP: FAIL ({duration:.1f}s) - {error[:50]}")

            no_bdd_results.append({
                "task_id": task_id,
                "with_bdd": False,
                "passed": passed,
                "generated_code": code,
                "full_code": full_code,
                "duration_seconds": duration,
                "tokens_used": tokens,
                "error": error if not passed else None
            })
        except Exception as e:
            print(f"  No-TCGP: ERROR - {e}")
            no_bdd_results.append({
                "task_id": task_id,
                "with_bdd": False,
                "passed": False,
                "error": str(e)
            })

    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    bdd_file = output_path / f"bdd_{model}_{timestamp}.jsonl"
    no_bdd_file = output_path / f"no_bdd_{model}_{timestamp}.jsonl"

    with open(bdd_file, 'w') as f:
        for r in bdd_results:
            f.write(json.dumps(r) + '\n')

    with open(no_bdd_file, 'w') as f:
        for r in no_bdd_results:
            f.write(json.dumps(r) + '\n')

    # Compute statistics
    n = len(problems)
    bdd_rate = bdd_passed / n if n > 0 else 0
    no_bdd_rate = no_bdd_passed / n if n > 0 else 0

    bdd_ci = wilson_ci(bdd_passed, n)
    no_bdd_ci = wilson_ci(no_bdd_passed, n)

    effect_size = cohens_d(n, bdd_passed, n, no_bdd_passed)

    # Print summary
    print("\n" + "=" * 70)
    print("HUMANEVAL PASS@1 RESULTS")
    print("=" * 70)
    print(f"\n{'Condition':<15} {'Pass@1':<12} {'95% CI':<20} {'Passed':<10}")
    print("-" * 60)
    print(f"{'TCGP':<15} {bdd_rate*100:>6.1f}%      [{bdd_ci[0]*100:.1f}%, {bdd_ci[1]*100:.1f}%]       {bdd_passed}/{n}")
    print(f"{'No-TCGP':<15} {no_bdd_rate*100:>6.1f}%      [{no_bdd_ci[0]*100:.1f}%, {no_bdd_ci[1]*100:.1f}%]       {no_bdd_passed}/{n}")
    print("-" * 60)
    print(f"\nDifference: {(bdd_rate - no_bdd_rate)*100:+.1f}%")
    print(f"Cohen's d: {effect_size:.3f}", end="")
    if abs(effect_size) < 0.2:
        print(" (negligible)")
    elif abs(effect_size) < 0.5:
        print(" (small)")
    elif abs(effect_size) < 0.8:
        print(" (medium)")
    else:
        print(" (large)")

    print(f"\nResults saved to: {output_path}")

    # Save summary
    summary = {
        "model": model,
        "provider": provider,
        "n_problems": n,
        "seed": seed,
        "timestamp": timestamp,
        "bdd": {
            "passed": bdd_passed,
            "pass_rate": bdd_rate,
            "ci_lower": bdd_ci[0],
            "ci_upper": bdd_ci[1]
        },
        "no_bdd": {
            "passed": no_bdd_passed,
            "pass_rate": no_bdd_rate,
            "ci_lower": no_bdd_ci[0],
            "ci_upper": no_bdd_ci[1]
        },
        "difference": bdd_rate - no_bdd_rate,
        "cohens_d": effect_size
    }

    with open(output_path / f"summary_{timestamp}.json", 'w') as f:
        json.dump(summary, f, indent=2)

    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="HumanEval TCGP vs No-TCGP Ablation Study")
    parser.add_argument("--model", default="gpt-4.1", help="Model to use")
    parser.add_argument("--provider", default="azure", choices=["azure", "openai", "anthropic", "gemini"])
    parser.add_argument("--samples", type=int, default=None, help="Number of samples (default: all 164)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="results/humaneval_ablation")

    args = parser.parse_args()

    run_humaneval_ablation(
        model=args.model,
        provider=args.provider,
        sample_size=args.samples,
        seed=args.seed,
        output_dir=args.output
    )
