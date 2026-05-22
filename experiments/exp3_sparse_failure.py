"""Experiment 3 driver: sparse-reward PPO on the two-phase memory MDP (plan §3.2 / §3.3).

Goal is *not* to make RL work — it is to show that vanilla PPO with a
terminal +1/0 reward flatlines on the slot-product action space defined
in `rl/action_space.py`. A flat curve is the result the proposal needs
to motivate the composite reward design.

Pipeline per episode:
  1. sample action from `PayloadActionSpace`
  2. decode action → payload string (the template carries retrieval bait
     but not the attack token; the policy has to compose the attack from
     slot pieces)
  3. `run_episode(payload, task)` — Phase 1 ingest into a fresh copy of
     the benign store, Phase 2 retrieve top-k + agent.act
  4. reward = 1.0 if `task.success_check(output)` else 0.0
  5. append rollout sample to PPO buffer; flush when batch fills

Per-episode rows stream to `results/exp3_episodes.jsonl`; per-update
rows stream to `results/exp3_updates.jsonl`. Both files are
line-buffered so a Ctrl-C still leaves you with a complete partial log
that `plot_exp3.py` can render.

Cost: ~1 OpenAI call per episode (the agent; no judge in the training
loop). At gpt-4o-mini pricing, 2k episodes ≈ a few cents and ~30–60 min
wall-clock. The `--episodes` and `--wall-clock-cap` flags both bound it.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch

# Allow running as a script from repo root: `python experiments/exp3_sparse_failure.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from env.agent import Agent
from env.episode import DEFAULT_CORPUS_PATH, EpisodeResult, run_episode
from env.memory_store import MemoryStore
from env.tasks import TASKS
from rl.action_space import (
    DEFAULT_N_SLOTS,
    DEFAULT_TASK_ID,
    DEFAULT_VOCAB_SIZE,
    PayloadActionSpace,
)
from rl.policy import MultiCategoricalPolicy
from rl.ppo_sparse import PPOConfig, PPOSparseTrainer, RolloutSample


RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


class _StoreFactory:
    """Snapshot the benign corpus once, hand out fresh stores per episode.

    `MemoryStore.from_corpus` reads JSONL + builds a FAISS index every
    call. At ~50ms each, 2k rebuilds is ~100s of overhead. Snapshotting
    the embedded vectors + entries once and re-adding them into a fresh
    `IndexFlatIP` per episode is ~5x cheaper, and we still get the
    isolation guarantee (no payload accumulation across episodes).
    """

    def __init__(self, corpus_path: Path, embedder):
        base = MemoryStore.from_corpus(corpus_path, embedder=embedder)
        n = len(base)
        self._dim = base._dim
        self._vectors = np.zeros((n, self._dim), dtype="float32")
        for i in range(n):
            self._vectors[i] = base._index.reconstruct(i)
        self._entries = list(base._entries)
        self._embedder = embedder

    def fresh(self) -> MemoryStore:
        store = MemoryStore(embedder=self._embedder, dim=self._dim)
        store._index.add(self._vectors)
        store._entries = list(self._entries)
        return store


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--task",
        default=DEFAULT_TASK_ID,
        help=(
            "Task id (default T1_brand_hijack; the action space's oracle "
            "winning_action_example only resolves for T1 at default settings)"
        ),
    )
    p.add_argument("--episodes", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument(
        "--wall-clock-cap",
        type=float,
        default=2 * 3600,
        help="Hard cap in seconds (plan §3.2: 2-hour cap).",
    )
    p.add_argument("--n-slots", type=int, default=DEFAULT_N_SLOTS)
    p.add_argument("--vocab-size", type=int, default=DEFAULT_VOCAB_SIZE)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--clip-eps", type=float, default=0.2)
    p.add_argument("--ppo-epochs", type=int, default=4)
    p.add_argument("--minibatch-size", type=int, default=64)
    p.add_argument("--entropy-coef", type=float, default=0.01)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--k", type=int, default=5, help="Retrieval top-k.")
    p.add_argument(
        "--corpus",
        type=Path,
        default=DEFAULT_CORPUS_PATH,
        help="Benign corpus JSONL.",
    )
    p.add_argument(
        "--episodes-out",
        type=Path,
        default=RESULTS_DIR / "exp3_episodes.jsonl",
    )
    p.add_argument(
        "--updates-out",
        type=Path,
        default=RESULTS_DIR / "exp3_updates.jsonl",
    )
    p.add_argument(
        "--plot",
        action="store_true",
        help="Also render the reward curve to results/exp3_curve.png on exit.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.task not in TASKS:
        raise SystemExit(f"Unknown task {args.task!r}; known: {list(TASKS)}")
    task = TASKS[args.task]

    torch.manual_seed(args.seed)
    np_rng = np.random.default_rng(args.seed)

    action_space = PayloadActionSpace(
        task_id=args.task,
        n_slots=args.n_slots,
        vocab_size=args.vocab_size,
    )

    policy = MultiCategoricalPolicy(
        n_slots=args.n_slots,
        vocab_size=args.vocab_size,
        hidden_dim=args.hidden_dim,
    )
    trainer = PPOSparseTrainer(
        policy,
        config=PPOConfig(
            lr=args.lr,
            clip_eps=args.clip_eps,
            n_epochs=args.ppo_epochs,
            minibatch_size=args.minibatch_size,
            entropy_coef=args.entropy_coef,
        ),
    )

    # Shared agent + embedder across episodes (Exp 1 pattern).
    from env.memory_store import _load_default_embedder

    embedder = _load_default_embedder()
    store_factory = _StoreFactory(args.corpus, embedder)
    agent = Agent()

    # Constant observation for this contextual-bandit MDP.
    obs = torch.zeros(1, dtype=torch.float32)

    # Oracle winning action (T1 only at defaults) — logged for sanity so
    # we can confirm post-hoc that there *was* a valid action within
    # reach; a flat curve then implicates exploration, not reachability.
    oracle = action_space.winning_action_example()

    args.episodes_out.parent.mkdir(parents=True, exist_ok=True)
    args.updates_out.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"Exp 3 sparse-PPO: task={args.task} episodes={args.episodes} "
        f"batch={args.batch_size} vocab={args.vocab_size} n_slots={args.n_slots}",
        flush=True,
    )
    print(f"  action space size: {args.vocab_size}^{args.n_slots} = "
          f"{action_space.n_actions_total:.2e}", flush=True)
    print(f"  attack reachable in vocab: {action_space.attack_reachable()}", flush=True)
    print(f"  oracle winning action: {oracle}", flush=True)
    print(f"  episodes_out: {args.episodes_out}", flush=True)

    buffer: list[RolloutSample] = []
    t_start = time.perf_counter()
    n_success = 0
    last_log_t = t_start

    with open(args.episodes_out, "w", buffering=1) as ep_f, open(
        args.updates_out, "w", buffering=1
    ) as up_f:
        ep_f.write(
            json.dumps(
                {
                    "event": "config",
                    "task": args.task,
                    "n_slots": args.n_slots,
                    "vocab_size": args.vocab_size,
                    "n_actions_total": action_space.n_actions_total,
                    "attack_reachable": action_space.attack_reachable(),
                    "oracle_action": list(oracle) if oracle else None,
                    "k": args.k,
                    "batch_size": args.batch_size,
                    "lr": args.lr,
                    "ppo_epochs": args.ppo_epochs,
                    "seed": args.seed,
                }
            )
            + "\n"
        )

        for ep_idx in range(args.episodes):
            elapsed = time.perf_counter() - t_start
            if elapsed > args.wall_clock_cap:
                print(
                    f"hit wall-clock cap ({args.wall_clock_cap:.0f}s) at "
                    f"episode {ep_idx}",
                    flush=True,
                )
                break

            action_t, log_prob, value = trainer.act(obs)
            action = tuple(int(a) for a in action_t.tolist())
            payload = action_space.decode(action)

            store = store_factory.fresh()
            ep_seed = int(np_rng.integers(0, 2**31 - 1))
            result: EpisodeResult = run_episode(
                payload=payload,
                task=task,
                store=store,
                agent=agent,
                k=args.k,
                seed=ep_seed,
            )
            reward = 1.0 if result.success else 0.0
            n_success += int(result.success)

            buffer.append(
                RolloutSample(
                    obs=obs.clone(),
                    action=action_t.clone(),
                    log_prob=log_prob,
                    value=value,
                    reward=reward,
                )
            )

            ep_f.write(
                json.dumps(
                    {
                        "event": "episode",
                        "episode": ep_idx,
                        "task": args.task,
                        "reward": reward,
                        "success": bool(result.success),
                        "action": list(action),
                        "payload": payload,
                        "query": result.query,
                        "payload_in_topk": result.payload_in_topk,
                        "log_prob": log_prob,
                        "value_pred": value,
                        "output": result.output,
                        "elapsed_s": elapsed,
                    }
                )
                + "\n"
            )

            if len(buffer) >= args.batch_size:
                stats = trainer.update(buffer)
                up_f.write(
                    json.dumps(
                        {
                            "event": "update",
                            "episode": ep_idx + 1,
                            **{
                                k: v
                                for k, v in asdict(stats).items()
                            },
                        }
                    )
                    + "\n"
                )
                buffer.clear()

            now = time.perf_counter()
            if now - last_log_t > 30 or ep_idx == args.episodes - 1:
                rate = (ep_idx + 1) / max(now - t_start, 1e-6)
                print(
                    f"  ep {ep_idx + 1}/{args.episodes} "
                    f"successes={n_success} "
                    f"rate={rate:.2f} ep/s "
                    f"elapsed={now - t_start:.1f}s",
                    flush=True,
                )
                last_log_t = now

        # Flush a trailing partial batch so we don't lose the last
        # `batch_size - 1` samples to a SIGINT-style exit.
        if buffer:
            stats = trainer.update(buffer)
            up_f.write(
                json.dumps(
                    {
                        "event": "update",
                        "episode": ep_idx + 1,
                        "partial": True,
                        **{k: v for k, v in asdict(stats).items()},
                    }
                )
                + "\n"
            )
            buffer.clear()

    total = time.perf_counter() - t_start
    print(
        f"\nDone. {n_success} successes / {ep_idx + 1} episodes "
        f"(ASR={n_success / max(ep_idx + 1, 1):.3f}) in {total:.1f}s",
        flush=True,
    )

    if args.plot:
        # Defer import so matplotlib isn't needed when only training.
        from experiments.plot_exp3 import plot_from_jsonl

        out = RESULTS_DIR / "exp3_curve.png"
        plot_from_jsonl(args.episodes_out, out)
        print(f"wrote {out}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
