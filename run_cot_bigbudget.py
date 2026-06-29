#!/usr/bin/env python3
"""
run_cot_bigbudget.py — Token-budget sweep for Chain-of-Thought on LiveCodeBench.

WHY THIS EXISTS
---------------
The manuscript reports that CoT "catastrophically fails" on LiveCodeBench (e.g. GPT-4o 2%
vs Direct 16%), and attributes it to *token exhaustion*: the model spends its output budget
reasoning and never emits code (NO_OUTPUT). In the harness, LiveCodeBench is a single call
at max_tokens=4000 (run_livecodebench.call_llm default).

The obvious reviewer objection is: "that's just an under-budget artifact — give CoT more
tokens and it recovers." This script tests that directly. It re-runs the CoT condition at a
sweep of token budgets and records, for each budget, both the Pass@1 rate AND the NO_OUTPUT
rate (fraction of problems where no executable code was produced). The result either:
  (a) shows CoT recovering as the budget grows  -> the collapse is budget-conditioned, or
  (b) shows CoT still failing at large budgets   -> the failure is intrinsic, not just the cap.
Either outcome is reportable and strengthens the paper's Threats-to-Validity treatment.

It reuses run_livecodebench.py's prompts, LLM call, code extraction, and test runner, so the
methodology is identical to the main experiment except for the token budget.

USAGE
-----
    # GPT-4o, CoT, 50 problems, budgets 4000/8000/16000 (default)
    python run_cot_bigbudget.py --model gpt-4o --provider azure --limit 50

    # include Direct as a reference line, custom budgets
    python run_cot_bigbudget.py --model gpt-4o --provider azure --limit 50 \
        --conditions cot direct --budgets 4000 8000 16000 32000

OUTPUT
------
    results/livecodebench_bigbudget/<condition>_<model>_<budget>_<ts>.jsonl  # per-problem
    results/livecodebench_bigbudget/budget_sweep_<model>_<ts>.csv            # budget vs rates
    results/livecodebench_bigbudget/budget_sweep_<model>_<ts>.md
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

from datasets import load_dataset

# Reuse the EXACT methodology from the main LiveCodeBench experiment.
from run_livecodebench import (
    create_cot_prompt,
    create_tcgp_prompt,
    create_direct_prompt,
    call_llm,
    extract_code,
    run_test,
)

HERE = Path(__file__).parent
OUT_DIR = HERE / "results" / "livecodebench_bigbudget"

PROMPT_FN = {"cot": create_cot_prompt, "bdd": create_tcgp_prompt, "direct": create_direct_prompt}


def parse_public_tests(problem: dict):
    pts = problem.get("public_test_cases", [])
    if isinstance(pts, str):
        try:
            pts = json.loads(pts)
        except Exception:
            pts = []
    return pts


def evaluate(problem: dict, condition: str, model: str, provider: str, max_tokens: int) -> dict:
    """Mirror run_livecodebench.evaluate_problem, but with an explicit max_tokens budget."""
    question = problem.get("question_content", problem.get("question", ""))
    starter_code = problem.get("starter_code", "")
    prompt = PROMPT_FN[condition](question, starter_code)

    try:
        response, tokens, duration = call_llm(prompt, model, provider, max_tokens=max_tokens)
        code = extract_code(response)
        error = None
    except Exception as e:
        response, code, tokens, duration, error = "", "", 0, 0, f"API error: {e}"

    no_output = (code.strip() == "")

    passed = False
    test_results = []
    public_tests = parse_public_tests(problem)
    if code and not error and public_tests:
        all_passed = True
        for test in public_tests[:3]:
            test_input = test.get("input", "")
            expected = test.get("output", test.get("expected_output", ""))
            if test_input and expected:
                ok, actual, terr = run_test(code, test_input, expected)
                test_results.append(ok)
                if not ok:
                    all_passed = False
        passed = all_passed and len(test_results) > 0

    return {
        "question_id": problem.get("question_id", problem.get("id", "unknown")),
        "condition": condition,
        "max_tokens": max_tokens,
        "passed": passed,
        "no_output": no_output,
        "tokens_used": tokens,
        "duration_seconds": duration,
        "error": error,
    }


def main():
    ap = argparse.ArgumentParser(description="CoT token-budget sweep on LiveCodeBench.")
    ap.add_argument("--model", default="gpt-4o")
    ap.add_argument("--provider", default="azure",
                    choices=["azure", "azure_models", "azure_responses", "openai", "anthropic", "gemini"])
    ap.add_argument("--limit", type=int, default=50, help="Number of problems (default: 50, matching the paper subset)")
    ap.add_argument("--version", default="release_v2", help="(kept for parity; loader uses bzantium/livecodebench)")
    ap.add_argument("--conditions", nargs="+", default=["cot"],
                    help="Conditions to sweep (default: cot). Add 'direct' for a reference line.")
    ap.add_argument("--budgets", nargs="+", type=int, default=[4000, 8000, 16000],
                    help="max_tokens budgets to sweep (default: 4000 8000 16000)")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("Loading LiveCodeBench dataset...")
    dataset = load_dataset("bzantium/livecodebench", split="test")
    if args.limit:
        dataset = dataset.select(range(min(args.limit, len(dataset))))
    n = len(dataset)
    print(f"Evaluating {n} problems | conditions={args.conditions} | budgets={args.budgets}")

    # rows: (condition, budget) -> counters
    summary = {}
    for condition in args.conditions:
        for budget in args.budgets:
            fp = OUT_DIR / f"{condition}_{args.model}_{budget}_{ts}.jsonl"
            passed = no_out = 0
            with open(fp, "w") as fh:
                for i, problem in enumerate(dataset):
                    r = evaluate(problem, condition, args.model, args.provider, budget)
                    fh.write(json.dumps(r) + "\n")
                    passed += int(r["passed"])
                    no_out += int(r["no_output"])
                    mark = "P" if r["passed"] else ("_" if r["no_output"] else ".")
                    print(f"  [{condition} @ {budget}] {i+1}/{n} {mark}", end="\r", flush=True)
            print()
            summary[(condition, budget)] = {
                "pass_rate": passed / n if n else 0.0,
                "no_output_rate": no_out / n if n else 0.0,
                "passed": passed, "no_output": no_out, "n": n,
            }

    # Write CSV + markdown
    csv = ["condition,max_tokens,pass_rate,no_output_rate,passed,no_output,n"]
    md = [f"# CoT token-budget sweep — {args.model} ({args.provider})",
          f"\nLiveCodeBench, n={n} problems, {ts}\n",
          "| Condition | max_tokens | Pass@1 | NO_OUTPUT | passed/n |",
          "|---|---|---|---|---|"]
    for (condition, budget), s in summary.items():
        csv.append(f"{condition},{budget},{s['pass_rate']:.4f},{s['no_output_rate']:.4f},"
                   f"{s['passed']},{s['no_output']},{s['n']}")
        md.append(f"| {condition} | {budget} | {s['pass_rate']*100:.1f}% | "
                  f"{s['no_output_rate']*100:.1f}% | {s['passed']}/{s['n']} |")

    md.append("\n**Reading this:** if Pass@1 climbs and NO_OUTPUT falls as max_tokens grows, "
              "the CoT collapse is budget-conditioned. If Pass@1 stays low even at the largest "
              "budget, the failure is intrinsic to CoT on hard problems, not merely the token cap.")

    (OUT_DIR / f"budget_sweep_{args.model}_{ts}.csv").write_text("\n".join(csv) + "\n")
    (OUT_DIR / f"budget_sweep_{args.model}_{ts}.md").write_text("\n".join(md) + "\n")
    print(f"\nWrote sweep summary to {OUT_DIR}")


if __name__ == "__main__":
    main()
