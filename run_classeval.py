#!/usr/bin/env python3
"""
ClassEval benchmark evaluation: BDD vs CoT vs Direct for class-level code generation.

ClassEval contains 100 class-level Python tasks with 410 methods and ~33 test cases per class.
"""

import json
import os
import re
import time
import signal
from datetime import datetime
from pathlib import Path
from contextlib import contextmanager
from datasets import load_dataset
from dotenv import load_dotenv

load_dotenv()

RESULTS_DIR = Path(__file__).parent / "results" / "classeval"


class TimeoutError(Exception):
    pass


@contextmanager
def time_limit(seconds):
    """Simple timeout using signal."""
    def signal_handler(signum, frame):
        raise TimeoutError("Timed out!")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


def create_bdd_prompt(skeleton: str, class_description: str, methods_info: list) -> str:
    """Create BDD-style prompt for class generation."""
    # Build BDD scenarios for each method
    scenarios = []
    for i, method in enumerate(methods_info[:5], 1):  # Limit to 5 methods for context
        method_name = method.get('method_name', 'unknown')
        method_desc = method.get('method_description', '')

        # Extract docstring examples if present
        examples = re.findall(r'>>>\s*(.+)', method_desc)

        scenario = f"""Scenario {i}: {method_name}
  Given a class instance is created
  When the {method_name} method is called"""
        if examples:
            scenario += f"\n  Then it should produce correct results as shown in docstring examples"
        else:
            scenario += f"\n  Then it should behave according to the method description"
        scenarios.append(scenario)

    bdd_scenarios = "\n\n".join(scenarios)

    prompt = f"""You are implementing a Python class. First, analyze the requirements using BDD scenarios, then implement the complete class.

## Class Skeleton (with method signatures and docstrings):
```python
{skeleton}
```

## BDD Scenarios to satisfy:
{bdd_scenarios}

## Instructions:
1. Review each BDD scenario carefully
2. Consider edge cases: empty inputs, None values, boundary conditions
3. Implement ALL methods in the skeleton with complete, working code
4. Return ONLY the complete Python class code, no explanations

```python
"""
    return prompt


def create_cot_prompt(skeleton: str, class_description: str, methods_info: list) -> str:
    """Create Chain-of-Thought prompt for class generation."""
    prompt = f"""You are implementing a Python class. Think step by step about each method before implementing.

## Class Skeleton (with method signatures and docstrings):
```python
{skeleton}
```

## Instructions:
Think through each method step by step:
1. What is the purpose of each method?
2. What are the inputs and expected outputs?
3. What edge cases should be handled?
4. How do the methods interact with each other?

After thinking through the design, implement ALL methods with complete, working code.

Let me think step by step...

```python
"""
    return prompt


def create_direct_prompt(skeleton: str, class_description: str, methods_info: list) -> str:
    """Create direct prompt for class generation."""
    prompt = f"""Complete the following Python class by implementing all methods:

```python
{skeleton}
```

Return ONLY the complete Python class code with all methods implemented:

```python
"""
    return prompt


def extract_code(response: str) -> str:
    """Extract Python code from LLM response."""
    # Try to find code between ```python and ```
    matches = re.findall(r'```python\s*(.*?)```', response, re.DOTALL)
    if matches:
        # Return the longest match (likely the full class)
        return max(matches, key=len).strip()

    # Try without language specifier
    matches = re.findall(r'```\s*(.*?)```', response, re.DOTALL)
    if matches:
        return max(matches, key=len).strip()

    # Handle truncated responses (no closing backticks)
    # Look for ```python at start and take everything after
    if '```python' in response:
        code = response.split('```python', 1)[1]
        # Remove any trailing ``` if present
        if '```' in code:
            code = code.split('```')[0]
        return code.strip()

    if '```' in response:
        code = response.split('```', 1)[1]
        # Remove language specifier if present (e.g., "python\n")
        if code.startswith(('python\n', 'Python\n')):
            code = code.split('\n', 1)[1] if '\n' in code else code
        # Remove any trailing ```
        if '```' in code:
            code = code.split('```')[0]
        return code.strip()

    # Return as-is if no code blocks
    return response.strip()


def run_test(code: str, test_code: str, import_statement: str, timeout: int = 10) -> tuple:
    """
    Run test code against generated class.
    Returns (passed: bool, error: str or None)
    """
    full_code = f"{import_statement}\n\n{code}\n\n{test_code}"

    try:
        with time_limit(timeout):
            exec_globals = {}
            exec(full_code, exec_globals)
        return True, None
    except TimeoutError:
        return False, "timeout"
    except AssertionError as e:
        return False, f"assertion: {str(e)[:100]}"
    except Exception as e:
        return False, f"{type(e).__name__}: {str(e)[:100]}"


def get_client(provider: str):
    """Get API client based on provider."""
    if provider == "azure":
        from openai import OpenAI
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        if not endpoint.endswith("/"):
            endpoint = endpoint + "/"
        return OpenAI(
            base_url=endpoint,
            api_key=os.getenv("AZURE_OPENAI_KEY")
        )
    elif provider == "azure_models":
        # For Azure AI endpoints that require AzureOpenAI client with api_version
        from openai import AzureOpenAI
        return AzureOpenAI(
            api_version="2024-12-01-preview",
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_KEY")
        )
    elif provider == "azure_responses":
        # For Azure responses API (gpt-5.3-codex, Llama, codex-mini)
        return None  # We'll use requests directly
    elif provider == "gemini":
        import google.generativeai as genai
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        return genai
    elif provider == "anthropic":
        import anthropic
        return anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    else:
        raise ValueError(f"Unknown provider: {provider}")


def call_llm(prompt: str, model: str, provider: str, max_tokens: int = 8000) -> tuple:
    """Call LLM and return (response, tokens, duration)."""
    client = get_client(provider)
    start = time.time()

    if provider in ("azure", "azure_models"):
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=max_tokens
        )
        duration = time.time() - start
        content = response.choices[0].message.content
        tokens = response.usage.total_tokens if response.usage else 0
    elif provider == "azure_responses":
        import requests
        endpoint = os.getenv("AZURE_RESPONSES_ENDPOINT", "https://chalan-resource.cognitiveservices.azure.com")
        api_key = os.getenv("AZURE_RESPONSES_KEY")
        url = f"{endpoint}/openai/responses?api-version=2025-04-01-preview"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        data = {
            "input": prompt,
            "max_output_tokens": max_tokens,
            "model": model
        }
        resp = requests.post(url, headers=headers, json=data, timeout=300)
        resp.raise_for_status()
        result = resp.json()
        duration = time.time() - start
        # Extract text from output[0].content[0].text
        output = result.get("output", [])
        if output and len(output) > 0:
            content_list = output[0].get("content", [])
            if content_list and len(content_list) > 0:
                content = content_list[0].get("text", "")
            else:
                content = ""
        else:
            content = ""
        tokens = result.get("usage", {}).get("total_tokens", 0)
    elif provider == "gemini":
        gen_model = client.GenerativeModel(model)
        response = gen_model.generate_content(
            prompt,
            generation_config=client.types.GenerationConfig(
                temperature=0.2,
                max_output_tokens=max_tokens
            )
        )
        duration = time.time() - start
        content = response.text
        tokens = response.usage_metadata.total_token_count if hasattr(response, 'usage_metadata') else 0
    elif provider == "anthropic":
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        duration = time.time() - start
        content = response.content[0].text
        tokens = response.usage.input_tokens + response.usage.output_tokens
    else:
        raise ValueError(f"Unknown provider: {provider}")

    return content, tokens, duration


def evaluate_problem(problem: dict, condition: str, model: str, provider: str) -> dict:
    """Evaluate a single ClassEval problem."""
    skeleton = problem['skeleton']
    class_description = problem['class_description']
    methods_info = problem['methods_info']
    if isinstance(methods_info, str):
        methods_info = json.loads(methods_info)

    test_code = problem['test']
    import_statement = problem['import_statement']

    # Create prompt based on condition
    if condition == "bdd":
        prompt = create_bdd_prompt(skeleton, class_description, methods_info)
    elif condition == "cot":
        prompt = create_cot_prompt(skeleton, class_description, methods_info)
    else:  # direct
        prompt = create_direct_prompt(skeleton, class_description, methods_info)

    # Call LLM
    try:
        response, tokens, duration = call_llm(prompt, model, provider)
        code = extract_code(response)
        error = None
    except Exception as e:
        response = ""
        code = ""
        tokens = 0
        duration = 0
        error = f"API error: {str(e)}"

    # Run tests
    if code and not error:
        passed, test_error = run_test(code, test_code, import_statement)
        if test_error:
            error = test_error
    else:
        passed = False

    return {
        "task_id": problem['task_id'],
        "class_name": problem['class_name'],
        "condition": condition,
        "passed": passed,
        "generated_code": code,
        "tokens_used": tokens,
        "duration_seconds": duration,
        "error": error
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run ClassEval BDD vs CoT evaluation")
    parser.add_argument("--model", type=str, default="gpt-4o", help="Model to use")
    parser.add_argument("--provider", type=str, default="azure", choices=["azure", "azure_models", "azure_responses", "gemini", "anthropic"],
                        help="API provider")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of problems")
    parser.add_argument("--start", type=int, default=0, help="Start from problem index (for resuming)")
    parser.add_argument("--conditions", nargs="+", default=["bdd", "cot", "direct"],
                        help="Conditions to evaluate")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading ClassEval dataset...")
    dataset = load_dataset("FudanSELab/ClassEval", split="test")
    print(f"Loaded {len(dataset)} problems")

    if args.limit:
        dataset = dataset.select(range(min(args.limit, len(dataset))))
        print(f"Limited to {len(dataset)} problems")

    # Apply start offset
    if args.start > 0:
        dataset = dataset.select(range(args.start, len(dataset)))
        print(f"Starting from problem {args.start}, {len(dataset)} problems remaining")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create output files (append mode if resuming)
    output_files = {}
    file_mode = 'a' if args.start > 0 else 'w'
    for condition in args.conditions:
        filepath = RESULTS_DIR / f"{condition}_{args.model}_{timestamp}.jsonl"
        output_files[condition] = open(filepath, file_mode)

    results = {cond: {"passed": 0, "total": 0} for cond in args.conditions}

    print(f"\nRunning evaluation on {args.model}...")
    print(f"Conditions: {args.conditions}")
    print("-" * 60)

    for i, problem in enumerate(dataset):
        task_id = problem['task_id']
        print(f"\n[{i+1}/{len(dataset)}] {task_id} - {problem['class_name']}")

        for condition in args.conditions:
            print(f"  {condition}...", end=" ", flush=True)

            result = evaluate_problem(problem, condition, args.model, args.provider)

            # Save result
            output_files[condition].write(json.dumps(result) + '\n')
            output_files[condition].flush()

            # Track stats
            results[condition]["total"] += 1
            if result["passed"]:
                results[condition]["passed"] += 1
                print("✓", end="")
            else:
                print(f"✗ ({result['error'][:30] if result['error'] else 'failed'})", end="")
            print()

    # Close files
    for f in output_files.values():
        f.close()

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for condition in args.conditions:
        passed = results[condition]["passed"]
        total = results[condition]["total"]
        rate = passed / total * 100 if total > 0 else 0
        print(f"{condition:10} {passed}/{total} = {rate:.1f}%")

    # Save summary
    summary = {
        "model": args.model,
        "timestamp": timestamp,
        "n_problems": len(dataset),
        "conditions": {
            cond: {
                "passed": results[cond]["passed"],
                "total": results[cond]["total"],
                "pass_rate": results[cond]["passed"] / results[cond]["total"] if results[cond]["total"] > 0 else 0
            }
            for cond in args.conditions
        }
    }

    summary_path = RESULTS_DIR / f"summary_{args.model}_{timestamp}.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    main()
