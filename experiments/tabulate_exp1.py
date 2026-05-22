"""Tabulate Exp 1 results into `results/exp1_table.md` (plan §2.3).

Reads `results/exp1_summary.json` (produced by `exp1_handcrafted.py`) and
renders the milestone-report headline table — ASR and retrieval@5 by
stealth level — plus a per-pair detail table and a contamination /
non-survivor flag block.

Run:
    /Users/MihirMenon/miniconda3/envs/cs224r/bin/python \\
        experiments/tabulate_exp1.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SUMMARY_PATH = ROOT / "results" / "exp1_summary.json"
TABLE_PATH = ROOT / "results" / "exp1_table.md"


STEALTH_ORDER = ["A", "B", "C"]
STEALTH_LABELS = {"A": "A overt", "B": "B narrative", "C": "C indirect"}


def _pct(x):
    if x is None:
        return "—"
    return f"{100 * x:.0f}%"


def render(summary: dict) -> str:
    n_target = summary.get("n_target", "?")
    by_stealth = summary.get("by_stealth", {})
    per_pair = summary.get("per_pair", [])

    lines: list[str] = []
    lines.append("# Experiment 1 — Hand-crafted Payload Sweep")
    lines.append("")
    lines.append(
        f"N = {n_target} episodes per (pair × condition). Judge is "
        "`gpt-4o-mini`, greedy, strict-JSON. Regex success_check is the "
        "sanity backup; ASR below is the judge column."
    )
    lines.append("")

    # ---- Headline by-stealth table -------------------------------------
    lines.append("## ASR by stealth level")
    lines.append("")
    lines.append(
        "| Stealth | Pairs | Survivors | Retrieval@5 (mal) | "
        "ASR judge (mal) | ASR judge (benign) | ASR regex (mal) |"
    )
    lines.append(
        "|---------|-------|-----------|--------------------|"
        "------------------|----------------------|------------------|"
    )
    for s in STEALTH_ORDER:
        b = by_stealth.get(s)
        if not b:
            continue
        lines.append(
            f"| {STEALTH_LABELS[s]} | {b['n_pairs']} | "
            f"{b['survivors']}/{b['n_pairs']} | "
            f"{_pct(b['retrieval_at_5_malicious_mean'])} | "
            f"{_pct(b['asr_judge_malicious_mean'])} | "
            f"{_pct(b['asr_judge_benign_mean'])} | "
            f"{_pct(b['asr_regex_malicious_mean'])} |"
        )
    lines.append("")

    # ---- Per-pair detail ----------------------------------------------
    lines.append("## Per-pair detail")
    lines.append("")
    lines.append(
        "| pair | task | stealth | method | persona | "
        "Retrieval@5 (mal) | ASR judge (mal) | ASR judge (benign) | "
        "ASR regex (mal) | Survivor |"
    )
    lines.append(
        "|------|------|---------|--------|---------|"
        "--------------------|------------------|----------------------|"
        "------------------|----------|"
    )
    pair_sorted = sorted(per_pair, key=lambda r: (r["stealth"], r["pair_id"]))
    for rec in pair_sorted:
        lines.append(
            f"| {rec['pair_id']} | {rec['task_id']} | {rec['stealth']} | "
            f"{rec['method']} | {rec['persona']} | "
            f"{_pct(rec['retrieval_at_5_malicious'])} | "
            f"{_pct(rec['asr_judge_malicious'])} | "
            f"{_pct(rec['asr_judge_benign'])} | "
            f"{_pct(rec['asr_regex_malicious'])} | "
            f"{'yes' if rec['survivor'] else 'NO'} |"
        )
    lines.append("")

    # ---- Flags ---------------------------------------------------------
    contaminated = [
        r for r in per_pair
        if (r.get("asr_judge_benign") or 0) > 0
    ]
    non_survivors = [r for r in per_pair if not r["survivor"]]

    lines.append("## Flags")
    lines.append("")
    if contaminated:
        lines.append(
            "**Benign false-positives (judge said attack succeeded on benign "
            "control — contamination signal on the benign generator):**"
        )
        lines.append("")
        for r in contaminated:
            lines.append(
                f"- `{r['pair_id']}` ({r['stealth']}, {r['task_id']}): "
                f"benign ASR = {_pct(r['asr_judge_benign'])}"
            )
        lines.append("")
    else:
        lines.append(
            "Benign-control ASR = 0% on every pair — judge passes the §2.2 "
            "sanity check, no contamination signal on the benign generator."
        )
        lines.append("")

    if non_survivors:
        lines.append(
            "**Non-survivors (no judge-confirmed success across N malicious "
            "episodes — kept in the table but flagged per §2.1c):**"
        )
        lines.append("")
        for r in non_survivors:
            lines.append(
                f"- `{r['pair_id']}` ({r['stealth']}, {r['task_id']}, "
                f"{r['method']}): malicious ASR = {_pct(r['asr_judge_malicious'])}"
            )
        lines.append("")
    else:
        lines.append("All 10 hand-crafted payloads are survivors (≥1/N succeeded).")
        lines.append("")

    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", type=Path, default=SUMMARY_PATH)
    ap.add_argument("--out", type=Path, default=TABLE_PATH)
    args = ap.parse_args()

    if not args.summary.exists():
        raise SystemExit(
            f"summary not found at {args.summary}; run experiments/exp1_handcrafted.py first"
        )
    with args.summary.open() as f:
        summary = json.load(f)

    md = render(summary)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(md)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
