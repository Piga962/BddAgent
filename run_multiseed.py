#!/usr/bin/env python3
"""
run_multiseed.py — Multi-seed robustness driver for the HumanEval TCGP-vs-CoT-vs-Direct
comparison.

WHY THIS EXISTS
---------------
The manuscript's headline HumanEval results use a single seed (seed=42). Most per-model
TCGP-CoT margins are under 5 percentage points, so a reviewer will reasonably ask whether
those margins survive decoding-time variance. This script re-runs the existing
`run_tcgp_vs_cot.py` comparison across several seeds and reports the mean, spread, and a
95% confidence interval for each condition's Pass@1 and for the TCGP-CoT margin.

It is a thin wrapper: it does NOT reimplement the experiment. It shells out to
`run_tcgp_vs_cot.py` once per (model, seed), then reads the `summary_*.json` that run
produces and aggregates across seeds.

IMPORTANT CAVEAT ON "SEED"
--------------------------
`run_tcgp_vs_cot.py` forwards the seed only to providers that expose a deterministic-seed
parameter (OpenAI / Azure-Responses). For Anthropic and Gemini (and plain Azure chat
completions) the seed is ignored, but generation temperature is > 0, so repeated runs are
still independent draws. In both cases, running multiple times estimates run-to-run
(decoding) variance — which is exactly what we want to put error bars on. Treat each
"seed" as an independent replicate.

USAGE
-----
    # default: gpt-4o on Azure, seeds 42/123/2024, full 164-problem HumanEval
    python run_multiseed.py

    # several models, custom seeds, quick smoke test on 20 problems
    python run_multiseed.py \
        --runs gpt-4o:azure gpt-35-turbo:azure gemini-2.5-flash:gemini \
        --seeds 42 123 2024 7 \
        --samples 20

OUTPUT
------
    results/multiseed/<model>_aggregate.json   # per-model aggregate stats
    results/multiseed/multiseed_summary.csv     # one row per (model, condition)
    results/multiseed/multiseed_report.md       # human-readable table
"""

import argparse
import json
import math
import statistics
import subprocess
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).parent
COMPARISON_SCRIPT = HERE / "run_tcgp_vs_cot.py"
BDD_VS_COT_DIR = HERE / "results" / "bdd_vs_cot"
OUT_DIR = HERE / "results" / "multiseed"

CONDITIONS = ["bdd", "cot", "direct"]  # "bdd" is TCGP in the harness's naming


def run_one(model: str, provider: str, seed: int, samples, output_dir: Path) -> dict:
    """Invoke run_tcgp_vs_cot.py once and return the summary dict it wrote."""
    before = set(output_dir.glob("summary_*.json")) if output_dir.exists() else set()

    cmd = [
        sys.executable, str(COMPARISON_SCRIPT),
        "--model", model,
        "--provider", provider,
        "--seed", str(seed),
        "--output", str(output_dir),
    ]
    if samples is not None:
        cmd += ["--samples", str(samples)]

    print(f"\n>>> {model} ({provider}) seed={seed}")
    print("    " + " ".join(cmd))
    subprocess.run(cmd, check=True)

    after = set(output_dir.glob("summary_*.json"))
    new = sorted(after - before, key=lambda p: p.stat().st_mtime)
    if not new:
        raise RuntimeError(
            f"No new summary_*.json appeared in {output_dir} after the run. "
            "Did run_tcgp_vs_cot.py change its output convention?"
        )
    summary = json.loads(new[-1].read_text())
    # sanity check
    if summary.get("seed") != seed or summary.get("model") != model:
        print(f"    WARNING: summary seed/model = {summary.get('seed')}/{summary.get('model')} "
              f"(expected {seed}/{model}); using newest summary anyway.")
    return summary


def ci95(values) -> tuple:
    """Normal-approx 95% CI of the mean for a small sample of replicates."""
    k = len(values)
    mean = statistics.mean(values)
    if k < 2:
        return mean, mean, mean, 0.0
    sd = statistics.stdev(values)
    half = 1.96 * sd / math.sqrt(k)
    return mean, mean - half, mean + half, sd


def aggregate(model: str, provider: str, summaries: list) -> dict:
    """Aggregate per-condition pass rates and the TCGP-CoT margin across replicates."""
    agg = {"model": model, "provider": provider, "n_seeds": len(summaries),
           "seeds": [s.get("seed") for s in summaries],
           "n_problems": summaries[0].get("n_problems"), "conditions": {}, "pairwise": {}}

    for cond in CONDITIONS:
        rates = [s["conditions"][cond]["pass_rate"] for s in summaries]
        mean, lo, hi, sd = ci95(rates)
        agg["conditions"][cond] = {
            "per_seed_pass_rate": rates,
            "mean_pass_rate": mean, "sd": sd,
            "ci95_lower": lo, "ci95_upper": hi,
        }

    # TCGP (bdd) minus CoT margin, per replicate
    margins = [s["pairwise"]["bdd_vs_cot"]["difference"] for s in summaries]
    mean, lo, hi, sd = ci95(margins)
    crosses_zero = lo <= 0.0 <= hi
    agg["pairwise"]["tcgp_minus_cot"] = {
        "per_seed_margin": margins,
        "mean_margin": mean, "sd": sd,
        "ci95_lower": lo, "ci95_upper": hi,
        "ci_crosses_zero": crosses_zero,
        "note": "If the 95% CI crosses zero, the single-seed winner is not robust to decoding variance.",
    }
    return agg


def main():
    ap = argparse.ArgumentParser(description="Multi-seed robustness driver for TCGP vs CoT vs Direct (HumanEval).")
    ap.add_argument("--runs", nargs="+", default=["gpt-4o:azure"],
                    help="Space-separated model:provider pairs, e.g. gpt-4o:azure gemini-2.5-flash:gemini")
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 2024],
                    help="Seeds / independent replicates (default: 42 123 2024)")
    ap.add_argument("--samples", type=int, default=None,
                    help="Number of HumanEval problems (default: all 164)")
    ap.add_argument("--bdd-output", default=str(BDD_VS_COT_DIR),
                    help="Where run_tcgp_vs_cot.py writes its per-run results")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    bdd_output = Path(args.bdd_output)

    csv_rows = ["model,condition,mean_pass_rate,sd,ci95_lower,ci95_upper,n_seeds"]
    md = [f"# Multi-seed robustness ({datetime.now().strftime('%Y-%m-%d %H:%M')})",
          f"\nSeeds/replicates: {args.seeds}  |  samples: {args.samples or 'all 164'}\n"]

    for run in args.runs:
        if ":" not in run:
            print(f"Skipping malformed --runs entry '{run}' (expected model:provider)")
            continue
        model, provider = run.split(":", 1)

        summaries = []
        for seed in args.seeds:
            try:
                summaries.append(run_one(model, provider, seed, args.samples, bdd_output))
            except subprocess.CalledProcessError as e:
                print(f"    RUN FAILED (model={model}, seed={seed}): {e}")
            except RuntimeError as e:
                print(f"    {e}")

        if not summaries:
            print(f"No successful runs for {model}; skipping aggregation.")
            continue

        agg = aggregate(model, provider, summaries)
        (OUT_DIR / f"{model}_aggregate.json").write_text(json.dumps(agg, indent=2))

        # CSV + markdown
        for cond in CONDITIONS:
            c = agg["conditions"][cond]
            csv_rows.append(f"{model},{cond},{c['mean_pass_rate']:.4f},{c['sd']:.4f},"
                            f"{c['ci95_lower']:.4f},{c['ci95_upper']:.4f},{agg['n_seeds']}")

        m = agg["pairwise"]["tcgp_minus_cot"]
        md.append(f"## {model} ({provider}) — {agg['n_seeds']} replicates")
        md.append("| Condition | Mean Pass@1 | SD | 95% CI |")
        md.append("|---|---|---|---|")
        for cond, label in [("bdd", "TCGP"), ("cot", "CoT"), ("direct", "Direct")]:
            c = agg["conditions"][cond]
            md.append(f"| {label} | {c['mean_pass_rate']*100:.1f}% | {c['sd']*100:.1f} pp | "
                      f"[{c['ci95_lower']*100:.1f}%, {c['ci95_upper']*100:.1f}%] |")
        verdict = "NOT robust (CI crosses 0)" if m["ci_crosses_zero"] else "robust (CI excludes 0)"
        md.append(f"\n**TCGP − CoT margin:** {m['mean_margin']*100:+.1f} pp "
                  f"(95% CI [{m['ci95_lower']*100:+.1f}, {m['ci95_upper']*100:+.1f}] pp) → **{verdict}**\n")

    (OUT_DIR / "multiseed_summary.csv").write_text("\n".join(csv_rows) + "\n")
    (OUT_DIR / "multiseed_report.md").write_text("\n".join(md) + "\n")
    print(f"\nWrote aggregates to {OUT_DIR}")
    print("  - multiseed_summary.csv")
    print("  - multiseed_report.md")


if __name__ == "__main__":
    main()
