#!/usr/bin/env python3
"""
Enhanced ablation study with improvements targeting Pass@k.

Key improvements over v1:
1. Concrete BDD test cases (not just Given-When-Then)
2. Explicit return type requirements
3. Function signature context
4. Multi-candidate generation with syntax validation
"""

import json
import time
import random
import ast
import textwrap
import re
from pathlib import Path
from datetime import datetime
from openai import AzureOpenAI
from dotenv import load_dotenv
import os

load_dotenv()


class EnhancedBddAgent:
    """Enhanced BDD-driven code generation agent."""

    def __init__(self, model: str = "gpt-4.1"):
        self.client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_KEY"),
            api_version="2024-12-01-preview",
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
        )
        self.model = model

    def generate_with_bdd_v2(self, requirements: str, context: str = "",
                             return_type: str = "Any", n_candidates: int = 3) -> tuple:
        """Generate code with enhanced BDD methodology."""
        start = time.time()
        total_tokens = 0

        # Step 1: Generate CONCRETE BDD test cases
        context_str = f'Context:\n{context[:2000]}' if context else ''
        bdd_prompt = f"""Analyze these requirements and generate CONCRETE test cases.

Requirements: {requirements}
{context_str}

Generate 3-4 specific test cases with concrete inputs and expected outputs.

Format (use actual values, not placeholders):
```
Test 1: [description]
  Input: [actual input value]
  Expected Output: [actual expected result]

Test 2: [description]
  Input: [actual input value]
  Expected Output: [actual expected result]

Edge Case:
  Input: [edge case input]
  Expected Output: [expected result]
```

Be specific! Use real values like {{"key": "value"}}, [1, 2, 3], "actual string", etc.
"""

        bdd_response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": bdd_prompt}],
            max_tokens=600,
            temperature=0.3
        )
        bdd_scenarios = bdd_response.choices[0].message.content
        total_tokens += bdd_response.usage.total_tokens

        # Step 2: Generate code with enhanced prompt
        context_str2 = f'Context:\n{context[:1500]}' if context else ''
        code_prompt = f"""Generate a Python function body based on these requirements and test cases.

Requirements: {requirements}
{context_str2}

Concrete Test Cases (your code MUST pass these):
{bdd_scenarios}

CRITICAL REQUIREMENTS:
1. Return type: {return_type}
2. EVERY code path must end with a return statement
3. Handle all edge cases shown in the test cases
4. Use 4-space indentation for base level
5. elif/else must align with their matching if
6. except/finally must align with their matching try

Return ONLY the function body (no 'def' line).
"""

        # Generate multiple candidates
        candidates = []
        for i in range(n_candidates):
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": code_prompt}],
                max_tokens=1000,
                temperature=0.2 + (i * 0.1)  # Vary temperature slightly
            )
            code = response.choices[0].message.content
            total_tokens += response.usage.total_tokens
            candidates.append(code)

        # Select best candidate
        best_code = self._select_best_candidate(candidates)
        duration = time.time() - start

        return best_code, duration, total_tokens, bdd_scenarios

    def generate_without_bdd_v2(self, requirements: str, context: str = "",
                                return_type: str = "Any", n_candidates: int = 3) -> tuple:
        """Generate code without BDD but with other enhancements."""
        start = time.time()

        context_str = f'Context:\n{context[:2000]}' if context else ''
        prompt = f"""Generate a Python function body based on these requirements.

Requirements: {requirements}
{context_str}

CRITICAL REQUIREMENTS:
1. Return type: {return_type}
2. EVERY code path must end with a return statement
3. Handle edge cases (empty inputs, None values, invalid types)
4. Use 4-space indentation for base level
5. elif/else must align with their matching if
6. except/finally must align with their matching try

Think step by step:
1. What are the main cases to handle?
2. What edge cases might occur?
3. What should be returned in each case?

Return ONLY the function body (no 'def' line).
"""

        # Generate multiple candidates
        candidates = []
        total_tokens = 0
        for i in range(n_candidates):
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000,
                temperature=0.2 + (i * 0.1)
            )
            code = response.choices[0].message.content
            total_tokens += response.usage.total_tokens
            candidates.append(code)

        best_code = self._select_best_candidate(candidates)
        duration = time.time() - start

        return best_code, duration, total_tokens

    def _select_best_candidate(self, candidates: list) -> str:
        """Select the best code candidate based on validation."""
        scored = []
        for code in candidates:
            processed = extract_function_body(code)
            score = self._score_code(processed)
            scored.append((score, processed))

        # Sort by score descending
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[0][1]

    def _score_code(self, code: str) -> int:
        """Score code quality."""
        score = 0

        # Syntax validity
        try:
            test_code = "def _test():\n" + code
            ast.parse(test_code)
            score += 10
        except SyntaxError:
            return 0  # Invalid syntax is immediate disqualification

        # Has return statement
        if 'return ' in code:
            score += 5

        # Has multiple return paths (handles cases)
        return_count = code.count('return ')
        score += min(return_count, 3)

        # Has error handling
        if 'try:' in code or 'except' in code:
            score += 2

        # Has type checks
        if 'isinstance(' in code:
            score += 1

        # Reasonable length (not too short, not too long)
        lines = len([l for l in code.split('\n') if l.strip()])
        if 3 <= lines <= 50:
            score += 2

        return score


def extract_function_body(code: str) -> str:
    """Extract and normalize function body with proper indentation."""
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

    # Dedent and re-indent
    dedented = textwrap.dedent('\n'.join(lines))
    lines = dedented.split('\n')

    # Fix control structure alignment
    fixed_lines = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            fixed_lines.append('')
            continue

        current_indent = len(line) - len(line.lstrip())
        fixed_lines.append(' ' * (4 + current_indent) + stripped)

    result = '\n'.join(fixed_lines)

    # Validate
    try:
        ast.parse("def _test():\n" + result)
    except SyntaxError:
        # Fallback
        result = '\n'.join('    ' + line.strip() if line.strip() else '' for line in lines)

    return result if result.strip() else "    pass"


def extract_return_type(requirements: str) -> str:
    """Extract return type from requirements."""
    req_lower = requirements.lower()

    if 'return: bool' in req_lower or 'returns: bool' in req_lower:
        return 'bool'
    elif ':return: str' in req_lower or 'returns: str' in req_lower:
        return 'str'
    elif ':return: int' in req_lower or 'returns: int' in req_lower:
        return 'int'
    elif ':return: list' in req_lower or 'returns: list' in req_lower:
        return 'list'
    elif ':return: dict' in req_lower or 'returns: dict' in req_lower:
        return 'dict'
    elif 'no return' in req_lower or 'return: none' in req_lower:
        return 'None'
    else:
        return 'Any'


def run_enhanced_ablation(
    dataset_path: str,
    mode: str = "local_file_completion",
    sample_size: int = 20,
    seed: int = 42,
    n_candidates: int = 3
):
    """Run enhanced ablation study."""
    print("=" * 70)
    print("ENHANCED ABLATION STUDY V2: BDD vs No-BDD")
    print("=" * 70)
    print(f"Model: gpt-4.1")
    print(f"Mode: {mode}")
    print(f"Sample size: {sample_size}")
    print(f"Candidates per generation: {n_candidates}")
    print(f"Seed: {seed}")
    print("=" * 70)

    # Load dataset
    random.seed(seed)
    tests = []
    with open(dataset_path, 'r') as f:
        for line in f:
            test = json.loads(line)
            # DevEval has 'requirement' dict with 'Functionality' and 'Arguments'
            req = test.get('requirement', {})
            requirements_text = f"{req.get('Functionality', '')}\n{req.get('Arguments', '')}"
            processed = {
                "namespace": test['namespace'],
                "requirements": requirements_text,
                "return_type": extract_return_type(requirements_text),
            }
            # DevEval doesn't have contexts_above, we'd need to read from source
            processed['context'] = ''
            tests.append(processed)

    if len(tests) > sample_size:
        tests = random.sample(tests, sample_size)

    print(f"Loaded {len(tests)} test cases\n")

    agent = EnhancedBddAgent()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    bdd_results = []
    no_bdd_results = []

    for i, test in enumerate(tests):
        print(f"\n[{i+1}/{len(tests)}] {test['namespace']}")
        print(f"  Return type: {test['return_type']}")

        # With BDD
        try:
            code, duration, tokens, bdd_scenarios = agent.generate_with_bdd_v2(
                test['requirements'],
                test.get('context', ''),
                test['return_type'],
                n_candidates
            )
            bdd_results.append({
                "namespace": test['namespace'],
                "mode": mode,
                "with_bdd": True,
                "version": "v2",
                "success": True,
                "generated_code": code,
                "bdd_scenarios": bdd_scenarios,
                "duration_seconds": duration,
                "tokens_used": tokens,
                "n_candidates": n_candidates,
                "error": None
            })
            print(f"  BDD: {duration:.1f}s, {tokens} tokens")
        except Exception as e:
            bdd_results.append({
                "namespace": test['namespace'],
                "mode": mode,
                "with_bdd": True,
                "version": "v2",
                "success": False,
                "error": str(e)
            })
            print(f"  BDD: ERROR - {e}")

        # Without BDD
        try:
            code, duration, tokens = agent.generate_without_bdd_v2(
                test['requirements'],
                test.get('context', ''),
                test['return_type'],
                n_candidates
            )
            no_bdd_results.append({
                "namespace": test['namespace'],
                "mode": mode,
                "with_bdd": False,
                "version": "v2",
                "success": True,
                "generated_code": code,
                "duration_seconds": duration,
                "tokens_used": tokens,
                "n_candidates": n_candidates,
                "error": None
            })
            print(f"  No-BDD: {duration:.1f}s, {tokens} tokens")
        except Exception as e:
            no_bdd_results.append({
                "namespace": test['namespace'],
                "mode": mode,
                "with_bdd": False,
                "version": "v2",
                "success": False,
                "error": str(e)
            })
            print(f"  No-BDD: ERROR - {e}")

    # Save results
    output_dir = Path("results/ablation_v2")
    output_dir.mkdir(parents=True, exist_ok=True)

    bdd_file = output_dir / f"bdd_{mode}_{timestamp}.jsonl"
    no_bdd_file = output_dir / f"no_bdd_{mode}_{timestamp}.jsonl"

    with open(bdd_file, 'w') as f:
        for r in bdd_results:
            f.write(json.dumps(r) + '\n')

    with open(no_bdd_file, 'w') as f:
        for r in no_bdd_results:
            f.write(json.dumps(r) + '\n')

    # Print summary
    print("\n" + "=" * 70)
    print("ENHANCED ABLATION V2 RESULTS")
    print("=" * 70)

    bdd_success = sum(1 for r in bdd_results if r.get('success', False))
    no_bdd_success = sum(1 for r in no_bdd_results if r.get('success', False))

    print(f"\nAPI Success: BDD {bdd_success}/{len(tests)}, No-BDD {no_bdd_success}/{len(tests)}")
    print(f"Results saved to: {output_dir}")
    print("\nNext step: Run Pass@k evaluation on these results")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--candidates", type=int, default=3)
    parser.add_argument("--mode", default="local_file_completion")
    args = parser.parse_args()

    run_enhanced_ablation(
        "data/DevEval-main/data.jsonl",
        mode=args.mode,
        sample_size=args.samples,
        seed=args.seed,
        n_candidates=args.candidates
    )
