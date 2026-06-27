#!/usr/bin/env python3
"""
Cost Analysis for TCGP vs CoT Experiments
Calculates actual dollar costs based on API pricing (April 2025)
"""

import json
from pathlib import Path

# API Pricing (per 1M tokens) - April 2025
# Format: (input_price, output_price)
PRICING = {
    # OpenAI / Azure
    'gpt-4o': (2.50, 10.00),
    'gpt-4o-mini': (0.15, 0.60),
    'gpt-4.1': (2.00, 8.00),  # Estimated based on GPT-4 Turbo
    'gpt-35-turbo': (0.50, 1.50),
    'gpt-5.3-codex': (3.00, 12.00),  # Estimated premium model
    'codex-mini': (0.15, 0.60),  # Similar to mini models

    # Google Gemini
    'gemini-2.5-flash': (0.075, 0.30),  # Gemini Flash pricing
    'gemini-3.1-flash': (0.075, 0.30),

    # Anthropic Claude
    'claude-sonnet-4': (3.00, 15.00),  # Claude 3.5 Sonnet pricing
    'claude-haiku-4.5': (0.25, 1.25),  # Claude 3 Haiku pricing

    # Meta Llama (via Azure/Together)
    'llama-3.3-70b': (0.90, 0.90),  # Together AI pricing
}

# Experimental data: avg tokens per problem (input + output combined for simplicity)
# From our experiments
EXPERIMENT_DATA = {
    'GPT-4o': {'bdd': 923, 'cot': 1667, 'direct': 289, 'n_problems': 164},
    'GPT-5.3-codex': {'bdd': 870, 'cot': 1583, 'direct': 301, 'n_problems': 164},
    'Llama-3.3-70B': {'bdd': 915, 'cot': 1545, 'direct': 291, 'n_problems': 164},
    'GPT-4.1': {'bdd': 855, 'cot': 1640, 'direct': 291, 'n_problems': 164},
    'gpt-35-turbo': {'bdd': 910, 'cot': 1652, 'direct': 292, 'n_problems': 164},
    'Gemini-3.1-Flash': {'bdd': 477, 'cot': 990, 'direct': 156, 'n_problems': 164},
    'Claude-Haiku-4.5': {'bdd': 1206, 'cot': 1642, 'direct': 355, 'n_problems': 164},
    'Claude-Sonnet-4': {'bdd': 1130, 'cot': 1747, 'direct': 355, 'n_problems': 164},
    'Gemini-2.5-Flash': {'bdd': 311, 'cot': 297, 'direct': 142, 'n_problems': 164},
    'codex-mini': {'bdd': 1713, 'cot': 1918, 'direct': 871, 'n_problems': 164},
}

# Pass rates for cost-effectiveness calculation
PASS_RATES = {
    'GPT-4o': {'bdd': 81.7, 'cot': 89.0, 'direct': 86.0},
    'GPT-5.3-codex': {'bdd': 86.6, 'cot': 90.2, 'direct': 81.7},
    'Llama-3.3-70B': {'bdd': 76.8, 'cot': 80.5, 'direct': 76.8},
    'GPT-4.1': {'bdd': 68.9, 'cot': 66.5, 'direct': 70.7},
    'gpt-35-turbo': {'bdd': 86.6, 'cot': 72.6, 'direct': 70.1},
    'Gemini-3.1-Flash': {'bdd': 29.3, 'cot': 26.2, 'direct': 28.0},
    'Claude-Haiku-4.5': {'bdd': 18.9, 'cot': 23.8, 'direct': 19.5},
    'Claude-Sonnet-4': {'bdd': 14.6, 'cot': 15.9, 'direct': 6.7},
    'Gemini-2.5-Flash': {'bdd': 23.2, 'cot': 10.4, 'direct': 16.5},
    'codex-mini': {'bdd': 14.6, 'cot': 12.2, 'direct': 2.4},
}

def get_pricing_key(model_name):
    """Map model name to pricing key."""
    model_lower = model_name.lower()
    if 'gpt-4o' in model_lower and 'mini' not in model_lower:
        return 'gpt-4o'
    elif 'gpt-5.3' in model_lower or 'codex' in model_lower and 'mini' not in model_lower:
        return 'gpt-5.3-codex'
    elif 'llama' in model_lower:
        return 'llama-3.3-70b'
    elif 'gpt-4.1' in model_lower:
        return 'gpt-4.1'
    elif 'gpt-35' in model_lower or 'gpt-3.5' in model_lower:
        return 'gpt-35-turbo'
    elif 'gemini-3' in model_lower or 'gemini-3.1' in model_lower:
        return 'gemini-3.1-flash'
    elif 'gemini-2' in model_lower or 'gemini-2.5' in model_lower:
        return 'gemini-2.5-flash'
    elif 'haiku' in model_lower:
        return 'claude-haiku-4.5'
    elif 'sonnet' in model_lower:
        return 'claude-sonnet-4'
    elif 'codex-mini' in model_lower:
        return 'codex-mini'
    return None

def calculate_cost(model_name, tokens, n_problems=164):
    """Calculate cost in dollars for a given number of tokens."""
    pricing_key = get_pricing_key(model_name)
    if pricing_key is None:
        return None

    input_price, output_price = PRICING[pricing_key]
    # Assume 30% input, 70% output (typical for code generation)
    input_tokens = tokens * 0.3
    output_tokens = tokens * 0.7

    total_tokens = tokens * n_problems
    cost = (input_tokens * n_problems * input_price / 1_000_000) + \
           (output_tokens * n_problems * output_price / 1_000_000)

    return cost

def calculate_cost_per_success(model_name, condition):
    """Calculate cost per successful solution."""
    data = EXPERIMENT_DATA.get(model_name)
    pass_rate = PASS_RATES.get(model_name, {}).get(condition, 0)

    if data is None or pass_rate == 0:
        return None

    tokens = data[condition]
    n_problems = data['n_problems']
    total_cost = calculate_cost(model_name, tokens, n_problems)

    if total_cost is None:
        return None

    n_successes = n_problems * (pass_rate / 100)
    return total_cost / n_successes if n_successes > 0 else float('inf')

def generate_cost_table():
    """Generate cost comparison table."""
    print("=" * 100)
    print("COST ANALYSIS: TCGP vs CoT vs Direct (HumanEval, 164 problems)")
    print("=" * 100)
    print()

    # Header
    print(f"{'Model':<20} | {'TCGP Cost':>10} | {'CoT Cost':>10} | {'Direct Cost':>10} | {'TCGP Savings':>12} | {'Best Value':>12}")
    print("-" * 100)

    total_bdd = 0
    total_cot = 0
    total_direct = 0

    for model in EXPERIMENT_DATA.keys():
        data = EXPERIMENT_DATA[model]

        bdd_cost = calculate_cost(model, data['bdd'])
        cot_cost = calculate_cost(model, data['cot'])
        direct_cost = calculate_cost(model, data['direct'])

        if bdd_cost and cot_cost and direct_cost:
            savings = ((cot_cost - bdd_cost) / cot_cost) * 100

            # Determine best value (highest pass rate per dollar)
            bdd_value = PASS_RATES[model]['bdd'] / bdd_cost if bdd_cost > 0 else 0
            cot_value = PASS_RATES[model]['cot'] / cot_cost if cot_cost > 0 else 0
            direct_value = PASS_RATES[model]['direct'] / direct_cost if direct_cost > 0 else 0

            best = 'TCGP' if bdd_value >= max(cot_value, direct_value) else \
                   ('CoT' if cot_value >= direct_value else 'Direct')

            print(f"{model:<20} | ${bdd_cost:>9.4f} | ${cot_cost:>9.4f} | ${direct_cost:>9.4f} | {savings:>10.1f}% | {best:>12}")

            total_bdd += bdd_cost
            total_cot += cot_cost
            total_direct += direct_cost

    print("-" * 100)
    print(f"{'TOTAL (all models)':<20} | ${total_bdd:>9.4f} | ${total_cot:>9.4f} | ${total_direct:>9.4f} | {((total_cot - total_bdd) / total_cot * 100):>10.1f}%")
    print()

    return total_bdd, total_cot, total_direct

def generate_cost_per_success_table():
    """Generate cost-per-successful-solution table."""
    print()
    print("=" * 100)
    print("COST PER SUCCESSFUL SOLUTION (lower is better)")
    print("=" * 100)
    print()

    print(f"{'Model':<20} | {'TCGP $/success':>14} | {'CoT $/success':>14} | {'Direct $/success':>16} | {'Most Efficient':>14}")
    print("-" * 100)

    for model in EXPERIMENT_DATA.keys():
        bdd_cps = calculate_cost_per_success(model, 'bdd')
        cot_cps = calculate_cost_per_success(model, 'cot')
        direct_cps = calculate_cost_per_success(model, 'direct')

        if bdd_cps and cot_cps and direct_cps:
            best = 'TCGP' if bdd_cps <= min(cot_cps, direct_cps) else \
                   ('CoT' if cot_cps <= direct_cps else 'Direct')

            print(f"{model:<20} | ${bdd_cps:>13.6f} | ${cot_cps:>13.6f} | ${direct_cps:>15.6f} | {best:>14}")

    print()

def generate_latex_cost_table():
    """Generate LaTeX table for paper."""
    print()
    print("=" * 100)
    print("LATEX TABLE (for paper)")
    print("=" * 100)
    print()

    print(r"\begin{table}[t]")
    print(r"\centering")
    print(r"\caption{API cost comparison across conditions (HumanEval, 164 problems). Costs in USD based on April 2025 pricing.}")
    print(r"\label{tab:costs}")
    print(r"\begin{tabular}{lcccc}")
    print(r"\toprule")
    print(r"\textbf{Model} & \textbf{TCGP} & \textbf{CoT} & \textbf{Direct} & \textbf{TCGP Savings} \\")
    print(r"\midrule")

    for model in EXPERIMENT_DATA.keys():
        data = EXPERIMENT_DATA[model]

        bdd_cost = calculate_cost(model, data['bdd'])
        cot_cost = calculate_cost(model, data['cot'])
        direct_cost = calculate_cost(model, data['direct'])

        if bdd_cost and cot_cost and direct_cost:
            savings = ((cot_cost - bdd_cost) / cot_cost) * 100
            print(f"{model} & \\${bdd_cost:.3f} & \\${cot_cost:.3f} & \\${direct_cost:.3f} & {savings:.0f}\\% \\\\")

    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")

def main():
    print("\n" + "=" * 100)
    print("TCGP vs CoT COST ANALYSIS")
    print("Based on HumanEval experiments with 10 LLMs")
    print("=" * 100 + "\n")

    total_bdd, total_cot, total_direct = generate_cost_table()

    print(f"\n📊 SUMMARY:")
    print(f"   Total experiment cost (TCGP):    ${total_bdd:.2f}")
    print(f"   Total experiment cost (CoT):    ${total_cot:.2f}")
    print(f"   Total experiment cost (Direct): ${total_direct:.2f}")
    print(f"   TCGP saves {((total_cot - total_bdd) / total_cot * 100):.1f}% vs CoT")
    print()

    generate_cost_per_success_table()
    generate_latex_cost_table()

if __name__ == "__main__":
    main()
