#!/usr/bin/env python3
"""
Generate publication-quality figures for BDD vs CoT paper.
Consolidates all experimental results into comprehensive visualizations.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

# Set publication-quality defaults
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# Color scheme (colorblind-friendly)
COLORS = {
    'bdd': '#2E86AB',      # Blue
    'cot': '#A23B72',      # Magenta/Purple
    'direct': '#F18F01',   # Orange
    'threshold': '#C73E1D', # Red for threshold line
}

# Consolidated experimental data (all 10 models)
MODELS_DATA = {
    'GPT-4o': {'bdd': 81.7, 'cot': 89.0, 'direct': 86.0, 'bdd_tokens': 923, 'cot_tokens': 1667},
    'GPT-5.3-codex': {'bdd': 86.6, 'cot': 90.2, 'direct': 81.7, 'bdd_tokens': 870, 'cot_tokens': 1583},
    'Llama-3.3-70B': {'bdd': 76.8, 'cot': 80.5, 'direct': 76.8, 'bdd_tokens': 915, 'cot_tokens': 1545},
    'GPT-4.1': {'bdd': 68.9, 'cot': 66.5, 'direct': 70.7, 'bdd_tokens': 855, 'cot_tokens': 1640},
    'gpt-35-turbo': {'bdd': 86.6, 'cot': 72.6, 'direct': 70.1, 'bdd_tokens': 910, 'cot_tokens': 1652},
    'Gemini-3.1-Flash-Lite-Preview': {'bdd': 29.3, 'cot': 26.2, 'direct': 28.0, 'bdd_tokens': 477, 'cot_tokens': 990},
    'Claude-Haiku-4.5': {'bdd': 18.9, 'cot': 23.8, 'direct': 19.5, 'bdd_tokens': 1206, 'cot_tokens': 1642},
    'Claude-Sonnet-4': {'bdd': 14.6, 'cot': 15.9, 'direct': 6.7, 'bdd_tokens': 1130, 'cot_tokens': 1747},
    'Gemini-2.5-Flash': {'bdd': 23.2, 'cot': 10.4, 'direct': 16.5, 'bdd_tokens': 311, 'cot_tokens': 297},
    'codex-mini': {'bdd': 14.6, 'cot': 12.2, 'direct': 2.4, 'bdd_tokens': 1713, 'cot_tokens': 1918},
}

# HumanEval+ robustness data (matches Table~\ref{tab:humanevalplus} in main.tex)
ROBUSTNESS_DATA = {
    'GPT-4o': {'bdd': 97.8, 'cot': 97.3, 'direct': 97.9},
    'GPT-5.3-codex': {'bdd': 98.6, 'cot': 97.3, 'direct': 98.5},
    'gpt-35-turbo': {'bdd': 97.9, 'cot': 97.5, 'direct': 97.4},
    'Llama-3.3-70B': {'bdd': 95.2, 'cot': 94.7, 'direct': 94.4},
    'Gemini-2.5-Flash': {'bdd': 97.4, 'cot': 100.0, 'direct': 100.0},
}

OUTPUT_DIR = Path(__file__).parent.parent / 'diagrams'
OUTPUT_DIR.mkdir(exist_ok=True)


def fig1_multimodel_comparison():
    """
    Figure 1: Grouped bar chart showing BDD vs CoT vs Direct across all 10 models.
    Models sorted by baseline (Direct) capability.
    """
    # Sort models by Direct (baseline) performance
    sorted_models = sorted(MODELS_DATA.keys(), key=lambda x: MODELS_DATA[x]['direct'], reverse=True)

    fig, ax = plt.subplots(figsize=(12, 5))

    x = np.arange(len(sorted_models))
    width = 0.25

    bdd_vals = [MODELS_DATA[m]['bdd'] for m in sorted_models]
    cot_vals = [MODELS_DATA[m]['cot'] for m in sorted_models]
    direct_vals = [MODELS_DATA[m]['direct'] for m in sorted_models]

    bars1 = ax.bar(x - width, bdd_vals, width, label='TCGP', color=COLORS['bdd'], edgecolor='white', linewidth=0.5)
    bars2 = ax.bar(x, cot_vals, width, label='CoT', color=COLORS['cot'], edgecolor='white', linewidth=0.5)
    bars3 = ax.bar(x + width, direct_vals, width, label='Direct', color=COLORS['direct'], edgecolor='white', linewidth=0.5)

    # Highlight winner for each model with a marker
    for i, model in enumerate(sorted_models):
        d = MODELS_DATA[model]
        winner = max(['bdd', 'cot', 'direct'], key=lambda c: d[c])
        winner_val = d[winner]
        x_pos = x[i] + (width if winner == 'direct' else (-width if winner == 'bdd' else 0))
        ax.plot(x_pos, winner_val + 2.5, marker='v', color='black', markersize=6)

    ax.set_xlabel('Model (sorted by baseline capability)')
    ax.set_ylabel('HumanEval Pass@1 (%)')
    ax.set_xticks(x)
    ax.set_xticklabels(sorted_models, rotation=45, ha='right')
    ax.legend(loc='upper right')
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig1-multimodel-comparison.png')
    plt.savefig(OUTPUT_DIR / 'fig1-multimodel-comparison.pdf')
    plt.close()
    print(f"Saved: fig1-multimodel-comparison.png/pdf")


def fig2_capability_threshold():
    """
    Figure 2: Scatter plot showing model capability threshold.
    X-axis: Baseline (Direct) Pass@1
    Y-axis: BDD - CoT difference
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    x_vals = []
    y_vals = []
    colors = []
    labels = []

    for model, data in MODELS_DATA.items():
        direct = data['direct']
        bdd_minus_cot = data['bdd'] - data['cot']
        x_vals.append(direct)
        y_vals.append(bdd_minus_cot)
        labels.append(model)

        # Color by winner
        if bdd_minus_cot > 0:
            colors.append(COLORS['bdd'])
        else:
            colors.append(COLORS['cot'])

    # Plot points
    scatter = ax.scatter(x_vals, y_vals, c=colors, s=120, edgecolors='black', linewidth=1, zorder=5)

    # Add model labels
    for i, label in enumerate(labels):
        offset_y = 1.5 if y_vals[i] > 0 else -2.5
        ax.annotate(label, (x_vals[i], y_vals[i]),
                   textcoords="offset points", xytext=(0, offset_y),
                   ha='center', fontsize=8, alpha=0.9)

    # Add observed-pattern line at x=70% (paper's bootstrap CI: 50-71%)
    ax.axvline(x=70, color=COLORS['threshold'], linestyle='--', linewidth=2, alpha=0.7)

    # Add horizontal line at y=0
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=1, alpha=0.5)

    # Add region annotations
    ax.fill_betweenx([-15, 20], 0, 70, alpha=0.1, color=COLORS['bdd'])
    ax.fill_betweenx([-15, 20], 70, 100, alpha=0.1, color=COLORS['cot'])
    ax.text(34, 16, 'TCGP zone', fontsize=11, color=COLORS['bdd'], fontweight='bold', alpha=0.7)
    ax.text(78, 16, 'CoT zone', fontsize=11, color=COLORS['cot'], fontweight='bold', alpha=0.7)

    # Linear regression line
    z = np.polyfit(x_vals, y_vals, 1)
    p = np.poly1d(z)
    x_line = np.linspace(0, 90, 100)
    ax.plot(x_line, p(x_line), 'k--', alpha=0.3, linewidth=1)

    # Calculate correlation
    corr = np.corrcoef(x_vals, y_vals)[0, 1]
    ax.text(5, -12, f'r = {corr:.2f}', fontsize=10, style='italic')

    ax.set_xlabel('Baseline (Direct) Pass@1 (%)')
    ax.set_ylabel('TCGP − CoT Difference (%)')
    ax.set_xlim(0, 95)
    ax.set_ylim(-15, 20)
    ax.grid(alpha=0.3)

    # Custom legend
    bdd_patch = mpatches.Patch(color=COLORS['bdd'], label='TCGP wins')
    cot_patch = mpatches.Patch(color=COLORS['cot'], label='CoT wins')
    ax.legend(handles=[bdd_patch, cot_patch], loc='lower left')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig2-capability-threshold.png')
    plt.savefig(OUTPUT_DIR / 'fig2-capability-threshold.pdf')
    plt.close()
    print(f"Saved: fig2-capability-threshold.png/pdf")


def fig3_cot_hurts_performance():
    """
    Figure 3: Bar chart showing models where CoT hurts performance (CoT < Direct).
    """
    # Models where CoT performs worse than Direct
    hurt_models = {m: d for m, d in MODELS_DATA.items() if d['cot'] < d['direct']}

    if not hurt_models:
        print("No models where CoT hurts performance")
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    models = list(hurt_models.keys())
    x = np.arange(len(models))
    width = 0.35

    cot_vals = [hurt_models[m]['cot'] for m in models]
    direct_vals = [hurt_models[m]['direct'] for m in models]
    diff_vals = [hurt_models[m]['cot'] - hurt_models[m]['direct'] for m in models]

    bars1 = ax.bar(x - width/2, direct_vals, width, label='Direct', color=COLORS['direct'], edgecolor='white')
    bars2 = ax.bar(x + width/2, cot_vals, width, label='CoT', color=COLORS['cot'], edgecolor='white')

    # Add difference annotations
    for i, diff in enumerate(diff_vals):
        ax.annotate(f'{diff:.1f}%', (x[i] + width/2, cot_vals[i] + 1),
                   ha='center', fontsize=9, color=COLORS['threshold'], fontweight='bold')

    ax.set_xlabel('Model')
    ax.set_ylabel('HumanEval Pass@1 (%)')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig3-cot-hurts.png')
    plt.savefig(OUTPUT_DIR / 'fig3-cot-hurts.pdf')
    plt.close()
    print(f"Saved: fig3-cot-hurts.png/pdf")


def fig4_token_efficiency():
    """
    Figure 4: Token efficiency comparison - BDD vs CoT.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Token usage comparison
    models = list(MODELS_DATA.keys())
    x = np.arange(len(models))
    width = 0.35

    bdd_tokens = [MODELS_DATA[m]['bdd_tokens'] for m in models]
    cot_tokens = [MODELS_DATA[m]['cot_tokens'] for m in models]

    ax1.barh(x - width/2, bdd_tokens, width, label='TCGP', color=COLORS['bdd'], edgecolor='white')
    ax1.barh(x + width/2, cot_tokens, width, label='CoT', color=COLORS['cot'], edgecolor='white')

    ax1.set_xlabel('Average Tokens per Problem')
    ax1.set_ylabel('Model')
    ax1.set_yticks(x)
    ax1.set_yticklabels(models)
    ax1.legend(loc='lower right')
    ax1.grid(axis='x', alpha=0.3)

    # Right: Efficiency ratio (TCGP tokens / CoT tokens)
    efficiency = [bdd_tokens[i] / cot_tokens[i] * 100 for i in range(len(models))]
    colors = [COLORS['bdd'] if e < 100 else COLORS['cot'] for e in efficiency]

    ax2.barh(x, efficiency, color=colors, edgecolor='white')
    ax2.axvline(x=100, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    ax2.set_xlabel('TCGP Tokens as % of CoT Tokens')
    ax2.set_ylabel('Model')
    ax2.set_yticks(x)
    ax2.set_yticklabels(models)
    ax2.grid(axis='x', alpha=0.3)

    # Add percentage labels
    for i, eff in enumerate(efficiency):
        ax2.annotate(f'{eff:.0f}%', (eff + 2, i), va='center', fontsize=8)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig4-token-efficiency.png')
    plt.savefig(OUTPUT_DIR / 'fig4-token-efficiency.pdf')
    plt.close()
    print(f"Saved: fig4-token-efficiency.png/pdf")


def fig5_robustness():
    """
    Figure 5: HumanEval+ robustness analysis.
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    models = list(ROBUSTNESS_DATA.keys())
    x = np.arange(len(models))
    width = 0.25

    bdd_vals = [ROBUSTNESS_DATA[m]['bdd'] for m in models]
    cot_vals = [ROBUSTNESS_DATA[m]['cot'] for m in models]
    direct_vals = [ROBUSTNESS_DATA[m]['direct'] for m in models]

    ax.bar(x - width, bdd_vals, width, label='TCGP', color=COLORS['bdd'], edgecolor='white')
    ax.bar(x, cot_vals, width, label='CoT', color=COLORS['cot'], edgecolor='white')
    ax.bar(x + width, direct_vals, width, label='Direct', color=COLORS['direct'], edgecolor='white')

    ax.set_xlabel('Model')
    ax.set_ylabel('Robustness (%)')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(90, 100)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig5-robustness.png')
    plt.savefig(OUTPUT_DIR / 'fig5-robustness.pdf')
    plt.close()
    print(f"Saved: fig5-robustness.png/pdf")


def fig6_summary_heatmap():
    """
    Figure 6: Heatmap summary showing winner for each model.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Sort models by Direct performance
    sorted_models = sorted(MODELS_DATA.keys(), key=lambda x: MODELS_DATA[x]['direct'], reverse=True)

    # Create matrix: rows=models, cols=TCGP/CoT/Direct
    conditions = ['TCGP', 'CoT', 'Direct']
    data = []
    annotations = []

    for model in sorted_models:
        row = [MODELS_DATA[model]['bdd'], MODELS_DATA[model]['cot'], MODELS_DATA[model]['direct']]
        data.append(row)
        annotations.append([f'{v:.1f}%' for v in row])

    data = np.array(data)

    # Create heatmap
    im = ax.imshow(data, cmap='YlGnBu', aspect='auto', vmin=0, vmax=100)

    # Add text annotations
    for i in range(len(sorted_models)):
        for j in range(len(conditions)):
            text_color = 'white' if data[i, j] > 50 else 'black'
            ax.text(j, i, annotations[i][j], ha='center', va='center', color=text_color, fontsize=9)

            # Highlight winner with bold
            row_max = max(data[i, :])
            if data[i, j] == row_max:
                ax.text(j, i, annotations[i][j], ha='center', va='center',
                       color=text_color, fontsize=9, fontweight='bold')

    ax.set_xticks(np.arange(len(conditions)))
    ax.set_xticklabels(conditions)
    ax.set_yticks(np.arange(len(sorted_models)))
    ax.set_yticklabels(sorted_models)

    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Pass@1 (%)', rotation=-90, va='bottom')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig6-summary-heatmap.png')
    plt.savefig(OUTPUT_DIR / 'fig6-summary-heatmap.pdf')
    plt.close()
    print(f"Saved: fig6-summary-heatmap.png/pdf")


def fig7_winner_pie():
    """
    Figure 7: Pie chart showing overall winner distribution.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    # Left: BDD vs CoT winner count
    bdd_wins = sum(1 for d in MODELS_DATA.values() if d['bdd'] > d['cot'])
    cot_wins = len(MODELS_DATA) - bdd_wins

    ax1.pie([bdd_wins, cot_wins], labels=['BDD Wins', 'CoT Wins'],
            colors=[COLORS['bdd'], COLORS['cot']],
            autopct='%1.0f%%', startangle=90, explode=(0.05, 0.05))
    ax1.set_title(f'BDD vs CoT: {bdd_wins}/{len(MODELS_DATA)} vs {cot_wins}/{len(MODELS_DATA)}')

    # Right: Models where CoT hurts vs helps
    cot_helps = sum(1 for d in MODELS_DATA.values() if d['cot'] > d['direct'])
    cot_hurts = sum(1 for d in MODELS_DATA.values() if d['cot'] < d['direct'])
    cot_same = len(MODELS_DATA) - cot_helps - cot_hurts

    colors2 = ['#4CAF50', '#F44336', '#9E9E9E']  # Green, Red, Gray
    ax2.pie([cot_helps, cot_hurts, cot_same],
            labels=['CoT Helps', 'CoT Hurts', 'Same'],
            colors=colors2,
            autopct='%1.0f%%', startangle=90)
    ax2.set_title(f'CoT vs Direct: Helps {cot_helps}, Hurts {cot_hurts}, Same {cot_same}')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig7-winner-summary.png')
    plt.savefig(OUTPUT_DIR / 'fig7-winner-summary.pdf')
    plt.close()
    print(f"Saved: fig7-winner-summary.png/pdf")


# LiveCodeBench failure-mode distributions (n=50 per model except where noted).
# Categories: PASSED (correct), NO_OUTPUT (token exhaustion / no executable code),
# WRONG_ANSWER, OTHER_ERROR (runtime / syntax). Source: results/analysis_output/error_analysis_results.json
LIVECODEBENCH_ERRORS = {
    'GPT-4o':       {'cot':    {'PASSED': 1,  'NO_OUTPUT': 37, 'WRONG_ANSWER': 8,  'OTHER_ERROR': 4},
                     'tcgp':   {'PASSED': 8,  'NO_OUTPUT': 10, 'WRONG_ANSWER': 13, 'OTHER_ERROR': 19},
                     'direct': {'PASSED': 8,  'NO_OUTPUT': 16, 'WRONG_ANSWER': 10, 'OTHER_ERROR': 16}},
    'GPT-4.1':      {'cot':    {'PASSED': 8,  'NO_OUTPUT': 33, 'WRONG_ANSWER': 7,  'OTHER_ERROR': 2},
                     'tcgp':   {'PASSED': 19, 'NO_OUTPUT': 14, 'WRONG_ANSWER': 11, 'OTHER_ERROR': 6},
                     'direct': {'PASSED': 8,  'NO_OUTPUT': 20, 'WRONG_ANSWER': 10, 'OTHER_ERROR': 12}},
    'GPT-5.3-codex':{'cot':    {'PASSED': 2,  'NO_OUTPUT': 0,  'WRONG_ANSWER': 0,  'OTHER_ERROR': 48},
                     'tcgp':   {'PASSED': 13, 'NO_OUTPUT': 0,  'WRONG_ANSWER': 6,  'OTHER_ERROR': 31},
                     'direct': {'PASSED': 27, 'NO_OUTPUT': 0,  'WRONG_ANSWER': 7,  'OTHER_ERROR': 16}},
    'gpt-35-turbo': {'cot':    {'PASSED': 12, 'NO_OUTPUT': 2,  'WRONG_ANSWER': 11, 'OTHER_ERROR': 25},
                     'tcgp':   {'PASSED': 15, 'NO_OUTPUT': 2,  'WRONG_ANSWER': 10, 'OTHER_ERROR': 23},
                     'direct': {'PASSED': 11, 'NO_OUTPUT': 6,  'WRONG_ANSWER': 8,  'OTHER_ERROR': 25}},
    'Gemini-3.1-Flash-Lite-Preview': {'cot':    {'PASSED': 30, 'NO_OUTPUT': 10, 'WRONG_ANSWER': 6, 'OTHER_ERROR': 4},
                         'tcgp':   {'PASSED': 33, 'NO_OUTPUT': 8,  'WRONG_ANSWER': 5, 'OTHER_ERROR': 4},
                         'direct': {'PASSED': 15, 'NO_OUTPUT': 19, 'WRONG_ANSWER': 5, 'OTHER_ERROR': 11}},
}

# Per-model HumanEval / ClassEval / LiveCodeBench pass rates (TCGP, CoT, Direct).
# Source: main paper Tables 4 and 7.
THREE_BENCH_DATA = {
    'GPT-4o':            {'HumanEval': (81.7, 89.0, 86.0), 'ClassEval': (98, 99, 97), 'LiveCodeBench': (16, 2, 16)},
    'GPT-5.3-codex':     {'HumanEval': (86.6, 90.2, 81.7), 'ClassEval': (95, 95, 95), 'LiveCodeBench': (26, 4, 54)},
    'GPT-4.1':           {'HumanEval': (68.9, 66.5, 70.7), 'ClassEval': (95, 98, 95), 'LiveCodeBench': (38, 16, 16)},
    'gpt-35-turbo':      {'HumanEval': (86.6, 72.6, 70.1), 'ClassEval': (96, 97, 95), 'LiveCodeBench': (30, 24, 22)},
    'Claude-Sonnet-4':   {'HumanEval': (14.6, 15.9,  6.7), 'ClassEval': (95, 96, 95), 'LiveCodeBench': (28, 28, 38)},
    'Claude-Haiku-4.5':  {'HumanEval': (18.9, 23.8, 19.5), 'ClassEval': (96, 96, 96), 'LiveCodeBench': (22, 14, 16)},
    'Gemini-3.1-Flash-Lite-Preview':  {'HumanEval': (29.3, 26.2, 28.0), 'ClassEval': (96, 97, 96), 'LiveCodeBench': (66, 60, 30)},
    'Gemini-2.5-Flash':  {'HumanEval': (23.2, 10.4, 16.5), 'ClassEval': (68, 69, 81), 'LiveCodeBench': None},
}


def fig8_token_exhaustion():
    """Stacked bars per model showing LiveCodeBench outcome distribution
    for CoT vs TCGP vs Direct. Highlights NO_OUTPUT (token exhaustion).
    """
    models = list(LIVECODEBENCH_ERRORS.keys())
    conditions = [('CoT', 'cot'), ('TCGP', 'tcgp'), ('Direct', 'direct')]
    categories = ['PASSED', 'NO_OUTPUT', 'WRONG_ANSWER', 'OTHER_ERROR']
    cat_colors = {
        'PASSED':       '#4CAF50',  # green
        'NO_OUTPUT':    '#C73E1D',  # red — the headline failure mode
        'WRONG_ANSWER': '#F18F01',  # orange
        'OTHER_ERROR':  '#9E9E9E',  # gray
    }

    fig, ax = plt.subplots(figsize=(11, 5))
    bar_width = 0.25
    n_models = len(models)
    x_base = np.arange(n_models)

    for ci, (cond_label, cond_key) in enumerate(conditions):
        x = x_base + (ci - 1) * bar_width
        bottoms = np.zeros(n_models)
        for cat in categories:
            vals = np.array([LIVECODEBENCH_ERRORS[m][cond_key].get(cat, 0) for m in models])
            totals = np.array([sum(LIVECODEBENCH_ERRORS[m][cond_key].values()) for m in models])
            pct = vals / totals * 100
            label = cat if ci == 0 else None  # legend once
            ax.bar(x, pct, bar_width, bottom=bottoms, color=cat_colors[cat],
                   edgecolor='white', linewidth=0.4, label=label)
            bottoms += pct
        # Condition label below the group
        for xi in x:
            ax.text(xi, -3, cond_label, ha='center', va='top', fontsize=8, color='#444')

    ax.set_xticks(x_base)
    ax.set_xticklabels(models, rotation=15, ha='right')
    ax.set_ylim(-7, 105)
    ax.set_ylabel('Outcome distribution (\\%)' if False else 'Outcome distribution (%)')
    ax.set_xlabel('')
    ax.legend(loc='upper right', ncol=4, fontsize=8, frameon=True)
    ax.grid(axis='y', alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig8-token-exhaustion.png')
    plt.savefig(OUTPUT_DIR / 'fig8-token-exhaustion.pdf')
    plt.close()
    print(f"Saved: fig8-token-exhaustion.png/pdf")


def fig9_three_benchmarks():
    """Three-panel grouped bar chart: TCGP/CoT/Direct pass rate per model on
    HumanEval, ClassEval, LiveCodeBench. Shows task-difficulty interaction.
    """
    benches = ['HumanEval', 'ClassEval', 'LiveCodeBench']
    cond_keys = ['TCGP', 'CoT', 'Direct']  # display order
    cond_color = {'TCGP': COLORS['bdd'], 'CoT': COLORS['cot'], 'Direct': COLORS['direct']}

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), sharey=False)
    width = 0.27

    for ax, bench in zip(axes, benches):
        models = [m for m, d in THREE_BENCH_DATA.items() if d.get(bench) is not None]
        x = np.arange(len(models))
        for i, cond in enumerate(cond_keys):
            # data is (TCGP, CoT, Direct) tuple; index by position
            cond_idx = {'TCGP': 0, 'CoT': 1, 'Direct': 2}[cond]
            vals = [THREE_BENCH_DATA[m][bench][cond_idx] for m in models]
            ax.bar(x + (i - 1) * width, vals, width,
                   label=cond if bench == 'HumanEval' else None,
                   color=cond_color[cond], edgecolor='white', linewidth=0.4)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=35, ha='right', fontsize=8)
        ax.set_ylim(0, 105)
        ax.set_ylabel('Pass@1 (%)' if bench == 'HumanEval' else '')
        ax.grid(axis='y', alpha=0.3)
        ax.set_axisbelow(True)
        # Difficulty label inside the panel
        difficulty = {'HumanEval': 'medium', 'ClassEval': 'easy', 'LiveCodeBench': 'hard'}[bench]
        ax.text(0.02, 0.96, f'{bench}\n({difficulty})', transform=ax.transAxes,
                ha='left', va='top', fontsize=10, fontweight='bold')

    axes[0].legend(loc='upper right', fontsize=9, frameon=True)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig9-three-benchmarks.png')
    plt.savefig(OUTPUT_DIR / 'fig9-three-benchmarks.pdf')
    plt.close()
    print(f"Saved: fig9-three-benchmarks.png/pdf")


def print_latex_table():
    """Print the LaTeX table for the paper."""
    print("\n" + "="*70)
    print("LATEX TABLE (Table 4)")
    print("="*70)

    # Sort by Direct performance
    sorted_models = sorted(MODELS_DATA.keys(), key=lambda x: MODELS_DATA[x]['direct'], reverse=True)

    print(r"\begin{table}[t]")
    print(r"\centering")
    print(r"\caption{Multi-model comparison of BDD vs CoT vs Direct prompting (HumanEval, n=164).}")
    print(r"\begin{tabular}{lccccc}")
    print(r"\toprule")
    print(r"\textbf{Model} & \textbf{BDD} & \textbf{CoT} & \textbf{Direct} & \textbf{BDD$-$CoT} & \textbf{Winner} \\")
    print(r"\midrule")

    bdd_wins = 0
    cot_wins = 0

    for model in sorted_models:
        d = MODELS_DATA[model]
        diff = d['bdd'] - d['cot']
        winner = 'BDD' if diff > 0 else 'CoT'
        if diff > 0:
            bdd_wins += 1
        else:
            cot_wins += 1

        bdd_str = f"\\textbf{{{d['bdd']:.1f}\\%}}" if diff > 0 else f"{d['bdd']:.1f}\\%"
        cot_str = f"\\textbf{{{d['cot']:.1f}\\%}}" if diff < 0 else f"{d['cot']:.1f}\\%"
        diff_str = f"+{diff:.1f}\\%" if diff > 0 else f"$-${abs(diff):.1f}\\%"

        print(f"{model} & {bdd_str} & {cot_str} & {d['direct']:.1f}\\% & {diff_str} & {winner} \\\\")

    print(r"\midrule")
    print(f"\\multicolumn{{4}}{{l}}{{\\textbf{{Overall: BDD wins {bdd_wins}/{len(MODELS_DATA)} ({100*bdd_wins/len(MODELS_DATA):.0f}\\%), CoT wins {cot_wins}/{len(MODELS_DATA)} ({100*cot_wins/len(MODELS_DATA):.0f}\\%)}}}} & \\multicolumn{{2}}{{r}}{{}} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")


def main():
    print("Generating publication-quality figures...")
    print(f"Output directory: {OUTPUT_DIR}")

    fig1_multimodel_comparison()
    fig2_capability_threshold()
    fig3_cot_hurts_performance()
    fig4_token_efficiency()
    fig5_robustness()
    fig6_summary_heatmap()
    fig7_winner_pie()
    fig8_token_exhaustion()
    fig9_three_benchmarks()

    print_latex_table()

    print("\n" + "="*70)
    print("All figures generated successfully!")
    print("="*70)


if __name__ == "__main__":
    main()
