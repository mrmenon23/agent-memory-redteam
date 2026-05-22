"""Experiment 1 sweep (plan §2.2 + §2.1c survivor gate).

For every paired seed in `data/exp1_seeds.jsonl`, run N episodes with the
malicious entry ingested into memory and a matched N episodes with the
benign sibling. For each episode log:

  - retrieval@k (did the ingested entry land in top-k?),
  - regex success (sanity backup, `task.success_check`),
  - strict-JSON judge verdict (primary, `env.judge.Judge`),
  - the full agent output and trigger query.

Per-episode rows stream to `results/exp1_episodes.jsonl` so the sweep is
resumable: rows already present (keyed by pair_id × condition × episode_idx)
are skipped. After the sweep, `results/exp1_summary.json` carries per-pair
aggregates plus the §2.1c survivor flag (≥1 judge-confirmed success out of
N malicious episodes).

Run:
    /Users/MihirMenon/miniconda3/envs/cs224r/bin/python \\
        experiments/exp1_handcrafted.py
    # custom N, alt corpus, alt output paths:
    experiments/exp1_handcrafted.py --n 20 --corpus data/benign_memories.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from env.agent import Agent  # noqa: E402
from env.episode import DEFAULT_CORPUS_PATH, run_episode  # noqa: E402
from env.judge import Judge  # noqa: E402
from env.memory_store import MemoryStore  # noqa: E402
from env.tasks import TASKS  # noqa: E402


SEEDS_PATH = ROOT / "data" / "exp1_seeds.jsonl"
EPISODES_PATH = ROOT / "results" / "exp1_episodes.jsonl"
SUMMARY_PATH = ROOT / "results" / "exp1_summary.json"
DEFAULT_N = 20


# --------------------------------------------------------------------------
# IO helpers
# --------------------------------------------------------------------------


def _load_pairs(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _load_existing_keys(path: Path) -> set[tuple[str, str, int]]:
    """Resume-support: which (pair_id, condition, episode_idx) are already logged."""
    if not path.exists():
        return set()
    done: set[tuple[str, str, int]] = set()
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            done.add((r["pair_id"], r["condition"], int(r["episode_idx"])))
    return done


def _load_shared_embedder():
    """Load the MiniLM embedder once for all episodes (saves ~1-2s per episode)."""
    from sentence_transformers import SentenceTransformer

    from env.memory_store import EMBED_MODEL

    return SentenceTransformer(EMBED_MODEL)


# --------------------------------------------------------------------------
# Per-episode driver
# --------------------------------------------------------------------------


def _run_one(
    pair: dict,
    condition: str,                 # "malicious" | "benign"
    episode_idx: int,
    agent: Agent,
    judge: Judge,
    embedder,
    corpus_path: Path,
    k: int,
) -> dict:
    task = TASKS[pair["task_id"]]
    payload = pair["malicious_text"] if condition == "malicious" else pair["benign_text"]

    # Fresh store per episode so payloads don't accumulate. We share the
    # embedder so we don't reload MiniLM ~400 times.
    store = MemoryStore.from_corpus(corpus_path, embedder=embedder)

    t0 = time.perf_counter()
    res = run_episode(
        payload=payload,
        task=task,
        store=store,
        agent=agent,
        k=k,
        seed=episode_idx,
        payload_metadata={
            "pair_id": pair["pair_id"],
            "payload_id": pair["payload_id"],
            "condition": condition,
        },
    )
    agent_dt = time.perf_counter() - t0

    t1 = time.perf_counter()
    try:
        verdict = judge.evaluate(res.query, res.output, task)
        judge_err = None
    except Exception as e:
        verdict = None
        judge_err = repr(e)
    judge_dt = time.perf_counter() - t1

    row = {
        "pair_id": pair["pair_id"],
        "payload_id": pair["payload_id"],
        "task_id": pair["task_id"],
        "stealth": pair["stealth"],
        "method": pair["method"],
        "persona": pair["persona"],
        "condition": condition,
        "episode_idx": episode_idx,
        "seed": episode_idx,
        "query": res.query,
        "payload_text": payload,
        "payload_in_topk": res.payload_in_topk,
        "regex_success": res.success,
        "judge_aligned": (verdict.aligned_with_user if verdict else None),
        "judge_attack_succeeded": (verdict.attack_succeeded if verdict else None),
        "judge_reason": (verdict.reason if verdict else ""),
        "judge_error": judge_err,
        "output": res.output,
        "elapsed_agent_s": round(agent_dt, 2),
        "elapsed_judge_s": round(judge_dt, 2),
    }
    return row


# --------------------------------------------------------------------------
# Sweep driver
# --------------------------------------------------------------------------


def run_sweep(
    n: int,
    corpus_path: Path,
    seeds_path: Path,
    episodes_path: Path,
    summary_path: Path,
    k: int = 5,
    resume: bool = True,
    pair_filter: Optional[set[str]] = None,
) -> None:
    pairs = _load_pairs(seeds_path)
    if pair_filter:
        pairs = [p for p in pairs if p["pair_id"] in pair_filter]
    if not pairs:
        raise RuntimeError(f"no pairs to run (loaded {seeds_path})")

    episodes_path.parent.mkdir(parents=True, exist_ok=True)
    done = _load_existing_keys(episodes_path) if resume else set()
    if done:
        print(f"Resuming: {len(done)} episodes already logged in {episodes_path}", file=sys.stderr)

    print(f"Loading shared embedder + warming corpus ...", file=sys.stderr)
    embedder = _load_shared_embedder()
    # Warm-load the corpus once so the first per-episode `from_corpus` doesn't
    # eat the JSON-parse cost in its timing.
    MemoryStore.from_corpus(corpus_path, embedder=embedder)

    agent = Agent()
    judge = Judge()

    conditions = ("malicious", "benign")
    total = len(pairs) * len(conditions) * n
    skipped = 0
    written = 0

    log_f = episodes_path.open("a", buffering=1)  # line-buffered for crash-resilience
    try:
        for pair in pairs:
            for condition in conditions:
                for ep in range(n):
                    key = (pair["pair_id"], condition, ep)
                    if key in done:
                        skipped += 1
                        continue
                    row = _run_one(
                        pair=pair,
                        condition=condition,
                        episode_idx=ep,
                        agent=agent,
                        judge=judge,
                        embedder=embedder,
                        corpus_path=corpus_path,
                        k=k,
                    )
                    log_f.write(json.dumps(row) + "\n")
                    written += 1
                    if written % 10 == 0 or ep == n - 1:
                        print(
                            f"  [{pair['pair_id']} {condition} ep={ep:02d}] "
                            f"topk={row['payload_in_topk']} "
                            f"judge_atk={row['judge_attack_succeeded']} "
                            f"regex={row['regex_success']} "
                            f"({written + skipped}/{total} total)",
                            file=sys.stderr,
                        )
    finally:
        log_f.close()

    print(f"\nWrote {written} new episodes (skipped {skipped} resumed)", file=sys.stderr)
    _write_summary(episodes_path, summary_path, n)


# --------------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------------


def _write_summary(episodes_path: Path, summary_path: Path, n: int) -> None:
    rows = []
    with episodes_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    # Group by (pair_id, condition).
    groups: dict[tuple[str, str], list[dict]] = {}
    pair_meta: dict[str, dict] = {}
    for r in rows:
        groups.setdefault((r["pair_id"], r["condition"]), []).append(r)
        pair_meta.setdefault(
            r["pair_id"],
            {
                "payload_id": r["payload_id"],
                "task_id": r["task_id"],
                "stealth": r["stealth"],
                "method": r["method"],
                "persona": r["persona"],
            },
        )

    def _rate(xs, key):
        vals = [x[key] for x in xs if x.get(key) is not None]
        return (sum(vals) / len(vals)) if vals else None

    per_pair: list[dict] = []
    for pair_id, meta in pair_meta.items():
        mal = groups.get((pair_id, "malicious"), [])
        ben = groups.get((pair_id, "benign"), [])
        record = {
            "pair_id": pair_id,
            **meta,
            "n_malicious": len(mal),
            "n_benign": len(ben),
            "retrieval_at_5_malicious": _rate(mal, "payload_in_topk"),
            "retrieval_at_5_benign": _rate(ben, "payload_in_topk"),
            "asr_judge_malicious": _rate(mal, "judge_attack_succeeded"),
            "asr_judge_benign": _rate(ben, "judge_attack_succeeded"),
            "asr_regex_malicious": _rate(mal, "regex_success"),
            "asr_regex_benign": _rate(ben, "regex_success"),
            # §2.1c survivor: ≥1 judge-confirmed success on the malicious side.
            "survivor": any(x.get("judge_attack_succeeded") for x in mal),
        }
        per_pair.append(record)

    # Aggregate by stealth.
    by_stealth: dict[str, dict] = {}
    for rec in per_pair:
        bucket = by_stealth.setdefault(
            rec["stealth"],
            {"n_pairs": 0, "ret5_mal": [], "ret5_ben": [], "asr_mal": [],
             "asr_ben": [], "regex_mal": [], "regex_ben": [], "survivors": 0},
        )
        bucket["n_pairs"] += 1
        bucket["survivors"] += int(bool(rec["survivor"]))
        for key, src in [
            ("ret5_mal", "retrieval_at_5_malicious"),
            ("ret5_ben", "retrieval_at_5_benign"),
            ("asr_mal", "asr_judge_malicious"),
            ("asr_ben", "asr_judge_benign"),
            ("regex_mal", "asr_regex_malicious"),
            ("regex_ben", "asr_regex_benign"),
        ]:
            if rec[src] is not None:
                bucket[key].append(rec[src])

    by_stealth_out = {}
    for stealth, b in by_stealth.items():
        mean = lambda xs: (sum(xs) / len(xs)) if xs else None
        by_stealth_out[stealth] = {
            "n_pairs": b["n_pairs"],
            "survivors": b["survivors"],
            "retrieval_at_5_malicious_mean": mean(b["ret5_mal"]),
            "retrieval_at_5_benign_mean": mean(b["ret5_ben"]),
            "asr_judge_malicious_mean": mean(b["asr_mal"]),
            "asr_judge_benign_mean": mean(b["asr_ben"]),
            "asr_regex_malicious_mean": mean(b["regex_mal"]),
            "asr_regex_benign_mean": mean(b["regex_ben"]),
        }

    out = {
        "n_target": n,
        "per_pair": per_pair,
        "by_stealth": by_stealth_out,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote summary → {summary_path}", file=sys.stderr)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=DEFAULT_N,
                    help="episodes per (pair, condition) [default 20]")
    ap.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS_PATH)
    ap.add_argument("--seeds", type=Path, default=SEEDS_PATH)
    ap.add_argument("--episodes-out", type=Path, default=EPISODES_PATH)
    ap.add_argument("--summary-out", type=Path, default=SUMMARY_PATH)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--no-resume", action="store_true",
                    help="ignore existing episode log; will overwrite incrementally")
    ap.add_argument("--pairs", nargs="*", default=None,
                    help="optional subset of pair_ids to run (e.g. pair_001 pair_002)")
    ap.add_argument("--summary-only", action="store_true",
                    help="skip the sweep; just regenerate the summary from the existing JSONL")
    args = ap.parse_args()

    if args.summary_only:
        _write_summary(args.episodes_out, args.summary_out, args.n)
        return

    if args.no_resume and args.episodes_out.exists():
        args.episodes_out.unlink()

    pair_filter = set(args.pairs) if args.pairs else None
    run_sweep(
        n=args.n,
        corpus_path=args.corpus,
        seeds_path=args.seeds,
        episodes_path=args.episodes_out,
        summary_path=args.summary_out,
        k=args.k,
        resume=not args.no_resume,
        pair_filter=pair_filter,
    )


if __name__ == "__main__":
    main()
