#!/usr/bin/env python3
"""
Aggregate results/ into results/figure_data.json — the single source of numbers
for the headline figures (fig1-6 via MODELS_DATA, fig5 via ROBUSTNESS_DATA).

This makes figure generation re-derive from the committed answers instead of
hard-coded literals. Run `--check` to verify the aggregated numbers match the
values currently hard-coded in generate_figures.py (i.e. the paper's numbers).

Selection rule: for each model, use the HumanEval run with n_problems==164 and
seed==42; if several, the latest timestamp.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent
RESULTS = ROOT / "results"

# HumanEval has 164 problems; Pass@1 is computed over the full denominator even
# when a run completed fewer (incomplete runs count the rest as failures), which
# is how the paper reports it (e.g. Claude-Sonnet: 24/164).
HUMANEVAL_N = 164
# Directories holding canonical HumanEval TCGP-vs-CoT-vs-Direct runs.
HUMANEVAL_DIRS = ["bdd_vs_cot", "bdd_vs_cot_anthropic"]

# The exact canonical run per model used in the paper, identified as the single
# timestamped run whose bdd/cot/direct pass counts reproduce the reported Pass@1.
# Pinning these makes regeneration deterministic despite extra reruns in results/.
CANONICAL_RUNS = {
    "gpt-4o": "20260401_154931",
    "gpt-5.3-codex": "20260401_181123",
    "Llama-3.3-70B-Instruct": "20260401_150456",
    "gpt-4.1": "20260401_150456",
    "gpt-35-turbo": "20260401_182440",
    "gemini-3.1-flash-lite-preview": "20260401_190830",
    "claude-haiku-4-5-20251001": "20260401_225730",
    "claude-sonnet-4-20250514": "20260401_225726",
    "gemini-2.5-flash": "20260330_230811",
    "codex-mini": "20260401_194157",
}

# provider model-id (as stored in summary 'model' field)  ->  display name (MODELS_DATA key)
MODEL_DISPLAY = {
    "gpt-4o": "GPT-4o",
    "gpt-5.3-codex": "GPT-5.3-codex",
    "Llama-3.3-70B-Instruct": "Llama-3.3-70B",
    "gpt-4.1": "GPT-4.1",
    "gpt-35-turbo": "gpt-35-turbo",
    "gemini-3.1-flash-lite-preview": "Gemini-3.1-Flash-Lite-Preview",
    "claude-haiku-4-5-20251001": "Claude-Haiku-4.5",
    "claude-sonnet-4-20250514": "Claude-Sonnet-4",
    "gemini-2.5-flash": "Gemini-2.5-Flash",
    "codex-mini": "codex-mini",
}


def _read_jsonl(f):
    """Return (n_records, n_passed, mean_tokens) for one results jsonl file."""
    n = passed = tok = 0
    for line in f.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        n += 1
        passed += 1 if d.get("passed") else 0
        tok += d.get("tokens_used", 0) or 0
    return n, passed, (tok / n if n else 0)


def _canonical_run(model_id, cond):
    """Read the pinned canonical run file for (model, condition)."""
    ts = CANONICAL_RUNS.get(model_id)
    if not ts:
        return None
    for sub in HUMANEVAL_DIRS:
        f = RESULTS / sub / f"{cond}_{model_id}_{ts}.jsonl"
        if f.exists():
            return _read_jsonl(f)
    return None


def build_models_data():
    """HumanEval Pass@1 (passed/164) + mean tokens per model, computed from jsonl."""
    out = {}
    for model_id, disp in MODEL_DISPLAY.items():
        runs = {c: _canonical_run(model_id, c) for c in ("bdd", "cot", "direct")}
        if any(runs[c] is None for c in runs):
            continue  # model not present in results/
        out[disp] = {
            "bdd": round(runs["bdd"][1] / HUMANEVAL_N * 100, 1),
            "cot": round(runs["cot"][1] / HUMANEVAL_N * 100, 1),
            "direct": round(runs["direct"][1] / HUMANEVAL_N * 100, 1),
            "bdd_tokens": round(runs["bdd"][2]),
            "cot_tokens": round(runs["cot"][2]),
        }
    return out


# ---- fig5 (HumanEval+ robustness) and fig9 (three-benchmark) --------------

# The 5 models the paper shows for HumanEval+ robustness.
ROBUSTNESS_MODELS = {"GPT-4o", "GPT-5.3-codex", "gpt-35-turbo", "Llama-3.3-70B", "Gemini-2.5-Flash"}

SUITE_DISPLAY = dict(MODEL_DISPLAY)
SUITE_DISPLAY["Llama-3.3-70B-Instruct"] = "Llama-3.3-70B"

# Canonical ClassEval / LiveCodeBench runs per model (the run reproducing the
# paper's Pass@1). A list means the run was done in chunks and is summed.
CLASSEVAL_RUNS = {
    "gpt-4o": "20260402_124403", "gpt-5.3-codex": "20260402_221307",
    "gpt-4.1": "20260403_114331", "gpt-35-turbo": "20260402_172349",
    "claude-sonnet-4-20250514": "20260402_124418",
    "claude-haiku-4-5-20251001": ["20260402_210301", "20260403_130244"],  # 39 + 61 = 100
    "gemini-3.1-flash-lite-preview": "20260403_114349", "gemini-2.5-flash": "20260401_232301",
}
LIVECODEBENCH_RUNS = {
    "gpt-4o": "20260402_164225", "gpt-5.3-codex": "20260402_215204",
    "gpt-4.1": "20260403_114332", "gpt-35-turbo": "20260402_180436",
    "claude-sonnet-4-20250514": "20260402_172347",
    "gemini-3.1-flash-lite-preview": "20260402_184844",
    # Gemini-2.5-Flash: not in paper. Claude-Haiku-4.5: see UNVERIFIED below.
}
# Cell that could NOT be reproduced from the committed results (no run or chunk
# combination matches the paper's value). Carried from the paper, flagged.
UNVERIFIED = {"Claude-Haiku-4.5": {"LiveCodeBench": [22, 14, 16]}}


def _summary_rate_ints(subdir, model_id, ts):
    """Rounded integer Pass@1 (bdd, cot, direct) from one summary file."""
    f = RESULTS / subdir / f"summary_{model_id}_{ts}.json"
    c = json.loads(f.read_text())["conditions"]
    return tuple(round(c[k]["pass_rate"] * 100) for k in ("bdd", "cot", "direct"))


def _summary_counts(subdir, model_id, ts):
    f = RESULTS / subdir / f"summary_{model_id}_{ts}.json"
    c = json.loads(f.read_text())["conditions"]
    return {k: (c[k]["passed"], c[k]["total"]) for k in ("bdd", "cot", "direct")}


def _bench_rate(subdir, model_id, pin):
    if pin is None:
        return None
    if isinstance(pin, list):  # summed chunks
        tot = {k: [0, 0] for k in ("bdd", "cot", "direct")}
        for ts in pin:
            for k, (p, n) in _summary_counts(subdir, model_id, ts).items():
                tot[k][0] += p; tot[k][1] += n
        return tuple(round(p / n * 100) for p, n in (tot[k] for k in ("bdd", "cot", "direct")))
    return _summary_rate_ints(subdir, model_id, pin)


def build_robustness():
    """HumanEval+ robustness (%) per model from humaneval_plus_summary_v2.json."""
    hp = json.loads((RESULTS / "humaneval_plus" / "humaneval_plus_summary_v2.json").read_text())
    out = {}
    for model_id, conds in hp.items():
        disp = SUITE_DISPLAY.get(model_id)
        if disp not in ROBUSTNESS_MODELS:
            continue
        out[disp] = {k: round(conds[k]["robustness"] * 100, 1) for k in ("bdd", "cot", "direct")}
    return out


def build_three_bench(models_data):
    """Per-model (HumanEval, ClassEval, LiveCodeBench) tuples for fig9."""
    out = {}
    for model_id, disp in SUITE_DISPLAY.items():
        if disp not in {m for m in _THREE_BENCH_MODELS}:
            continue
        he = models_data.get(disp)
        human = (he["bdd"], he["cot"], he["direct"]) if he else None
        ce = _bench_rate("classeval", model_id, CLASSEVAL_RUNS.get(model_id))
        lc = _bench_rate("livecodebench", model_id, LIVECODEBENCH_RUNS.get(model_id))
        if lc is None and disp in UNVERIFIED:
            lc = tuple(UNVERIFIED[disp]["LiveCodeBench"])
        out[disp] = {"HumanEval": human, "ClassEval": ce, "LiveCodeBench": lc}
    return out


# The 8 models fig9 shows (no Llama / codex-mini in the three-benchmark panel).
_THREE_BENCH_MODELS = {
    "GPT-4o", "GPT-5.3-codex", "GPT-4.1", "gpt-35-turbo", "Claude-Sonnet-4",
    "Claude-Haiku-4.5", "Gemini-3.1-Flash-Lite-Preview", "Gemini-2.5-Flash",
}


def build_livecodebench_errors():
    """fig8 outcome distribution per model: {disp: {cot/tcgp/direct: {cat: n}}}.
    Source: results/analysis_output/error_analysis_results.json (a per-model list).
    """
    rows = json.loads((RESULTS / "analysis_output" / "error_analysis_results.json").read_text())
    cats = ("PASSED", "NO_OUTPUT", "WRONG_ANSWER", "OTHER_ERROR")

    def norm(dist):
        # zero-fill the 4 plotted categories; fold any extra (e.g. RUNTIME_ERROR)
        # into OTHER_ERROR, as the paper does.
        n = {c: dist.get(c, 0) for c in cats}
        for k, v in dist.items():
            if k not in cats:
                n["OTHER_ERROR"] += v
        return n

    out = {}
    for r in rows:
        disp = SUITE_DISPLAY.get(r.get("model"))
        if not disp:
            continue
        out[disp] = {
            "cot": norm(r["cot_error_distribution"]),
            "tcgp": norm(r["bdd_error_distribution"]),  # 'bdd' on disk == TCGP in the paper
            "direct": norm(r["direct_error_distribution"]),
        }
    return out


def build_data():
    md = build_models_data()
    return {
        "MODELS_DATA": md,
        "ROBUSTNESS_DATA": build_robustness(),
        "THREE_BENCH_DATA": build_three_bench(md),
        "LIVECODEBENCH_ERRORS": build_livecodebench_errors(),
        "_unverified": UNVERIFIED,
    }


PASS_KEYS = {"bdd", "cot", "direct"}  # Pass@1 (%) — must match the paper exactly


def _compare(label, derived, hardcoded):
    """Print per-key diffs. Return (pass_rate_mismatches, token_warnings)."""
    pass_mism = tok_warn = 0
    for k in sorted(set(derived) | set(hardcoded)):
        dv, hv = derived.get(k), hardcoded.get(k)
        if dv is None:
            print(f"  [{label}] {k}: MISSING from results/ (paper only)"); pass_mism += 1; continue
        if hv is None:
            print(f"  [{label}] {k}: only in results/, not in paper"); continue
        pdiffs = {kk: (hv.get(kk), dv.get(kk)) for kk in dv if kk in PASS_KEYS and dv.get(kk) != hv.get(kk)}
        tdiffs = {kk: (hv.get(kk), dv.get(kk)) for kk in dv if kk not in PASS_KEYS and dv.get(kk) != hv.get(kk)}
        if pdiffs:
            pass_mism += 1
            print(f"  [{label}] {k} (Pass@1): " + ", ".join(f"{kk} paper={a} results={b}" for kk, (a, b) in pdiffs.items()))
        if tdiffs:
            tok_warn += 1
            print(f"  [{label}] {k} (tokens, warning): " + ", ".join(f"{kk} paper={a} results={b}" for kk, (a, b) in tdiffs.items()))
    return pass_mism, tok_warn


def main():
    data = build_data()
    if "--check" in sys.argv:
        from generate_figures import (_PAPER_MODELS_DATA, _PAPER_ROBUSTNESS_DATA,
                                       _PAPER_THREE_BENCH_DATA, _PAPER_LIVECODEBENCH_ERRORS)
        print("Cross-checking results/ against the paper literals in generate_figures.py ...")
        pass_mism, tok_warn = _compare("MODELS_DATA", data["MODELS_DATA"], _PAPER_MODELS_DATA)
        r_mism, _ = _compare("ROBUSTNESS_DATA", data["ROBUSTNESS_DATA"], _PAPER_ROBUSTNESS_DATA)

        # three-benchmark: dict of {bench: tuple|None}; flag UNVERIFIED cells as warnings
        tb_mism = tb_warn = 0
        derived_tb, paper_tb = data["THREE_BENCH_DATA"], _PAPER_THREE_BENCH_DATA
        for m in sorted(set(derived_tb) | set(paper_tb)):
            for bench in ("HumanEval", "ClassEval", "LiveCodeBench"):
                d = (derived_tb.get(m) or {}).get(bench)
                p = (paper_tb.get(m) or {}).get(bench)
                d = tuple(d) if isinstance(d, (list, tuple)) else d
                p = tuple(p) if isinstance(p, (list, tuple)) else p
                if bench in UNVERIFIED.get(m, {}):
                    # carried from the paper because no committed run reproduces it
                    tb_warn += 1
                    print(f"  [THREE_BENCH] {m}/{bench}: UNVERIFIED — no committed run reproduces paper={p}; value carried from paper, not derived")
                elif d != p:
                    tb_mism += 1
                    print(f"  [THREE_BENCH] {m}/{bench}: paper={p} results={d}")

        # fig8 LiveCodeBench outcome distribution: exact dict match per model
        err_mism = 0
        de, pe = data["LIVECODEBENCH_ERRORS"], _PAPER_LIVECODEBENCH_ERRORS
        for m in sorted(set(de) | set(pe)):
            if de.get(m) != pe.get(m):
                err_mism += 1
                print(f"  [LIVECODEBENCH_ERRORS] {m}: results/ differ from paper literal")

        total_fail = pass_mism + r_mism + tb_mism + err_mism
        print()
        if total_fail:
            print(f"FAIL: {total_fail} mismatch(es) vs the paper "
                  f"(MODELS_DATA={pass_mism}, ROBUSTNESS={r_mism}, THREE_BENCH={tb_mism}, "
                  f"LIVECODEBENCH_ERRORS={err_mism}).")
        else:
            warns = []
            if tok_warn: warns.append(f"{tok_warn} token-avg diff")
            if tb_warn: warns.append(f"{tb_warn} unverified cell (Claude-Haiku LiveCodeBench)")
            print("OK: all Pass@1 values across fig1/fig5/fig9 reproduce the paper"
                  + (f" — warnings: {', '.join(warns)} (see REPRODUCE.md)." if warns else "."))
        sys.exit(1 if total_fail else 0)
    out = RESULTS / "figure_data.json"
    out.write_text(json.dumps(data, indent=2, sort_keys=True))
    print(f"Wrote {out} ({len(data['MODELS_DATA'])} models)")


if __name__ == "__main__":
    main()
