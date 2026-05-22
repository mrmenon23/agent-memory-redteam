"""Render the Exp 3 training curve (plan §3.3).

Reads `results/exp3_episodes.jsonl` (written by `exp3_sparse_failure.py`)
and writes `results/exp3_curve.png`: per-episode reward + moving-average
overlay. Separate from training so the plot can be regenerated from a
partial / completed log without re-running the sweep.

Expected shape under sparse reward: flat near 0 across all 2k episodes
(see plan §3.3). If the curve climbs, the experiment is *more*
interesting, not less — report it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
DEFAULT_EPISODES = RESULTS_DIR / "exp3_episodes.jsonl"
DEFAULT_OUT = RESULTS_DIR / "exp3_curve.png"


def _moving_average(xs: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or xs.size == 0:
        return xs.astype(float)
    w = min(window, xs.size)
    kernel = np.ones(w) / w
    pad = np.full(w - 1, xs[0], dtype=float)
    padded = np.concatenate([pad, xs.astype(float)])
    return np.convolve(padded, kernel, mode="valid")


def _load(path: Path) -> tuple[dict, list[dict]]:
    config: dict = {}
    episodes: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("event") == "config":
                config = row
            elif row.get("event") == "episode":
                episodes.append(row)
    return config, episodes


def plot_from_jsonl(
    episodes_path: Path | str,
    out_path: Path | str,
    window: int = 50,
) -> Path:
    episodes_path = Path(episodes_path)
    out_path = Path(out_path)

    config, episodes = _load(episodes_path)
    if not episodes:
        raise SystemExit(f"no episode rows in {episodes_path}")

    eps = np.array([r["episode"] for r in episodes], dtype=int)
    rewards = np.array([r["reward"] for r in episodes], dtype=float)
    in_topk = np.array(
        [bool(r.get("payload_in_topk", False)) for r in episodes], dtype=float
    )
    asr = rewards.mean()

    ma_reward = _moving_average(rewards, window)
    ma_retrieval = _moving_average(in_topk, window)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(
        eps, rewards, s=6, alpha=0.15, color="C3",
        label="per-episode reward (0/1)",
    )
    ax.plot(eps, ma_reward, color="C3", lw=2, label=f"reward (mov. avg, w={window})")
    ax.plot(
        eps,
        ma_retrieval,
        color="C0",
        lw=1.5,
        ls="--",
        alpha=0.7,
        label=f"retrieval@k (mov. avg, w={window})",
    )
    ax.axhline(0.0, color="black", lw=0.5, alpha=0.3)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward / retrieval rate")
    ax.set_ylim(-0.05, 1.05)

    task = config.get("task", "?")
    n_actions = config.get("n_actions_total", None)
    n_actions_str = f"{n_actions:.1e}" if n_actions else "?"
    title = (
        f"Exp 3 — vanilla sparse-PPO on {task}\n"
        f"action space {n_actions_str} | "
        f"{len(episodes)} episodes | ASR={asr:.3f}"
    )
    ax.set_title(title)
    ax.legend(loc="upper left", framealpha=0.9)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=Path, default=DEFAULT_EPISODES)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--window", type=int, default=50)
    args = p.parse_args()
    out = plot_from_jsonl(args.episodes, args.out, window=args.window)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
