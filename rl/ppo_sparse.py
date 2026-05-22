"""Vanilla PPO with terminal-only reward (plan §3.2).

The Exp 3 MDP is degenerate-by-design: one action → one terminal reward.
That makes this effectively a contextual-bandit PPO — there's no GAE,
no λ, no bootstrapping past the next step. Return = reward; advantage =
reward − V(s). The shape of the loss (clipped surrogate + value loss +
entropy bonus) is preserved so the same trainer plugs into the
composite-reward run later, where multi-step rewards do matter.

Reward source: `env.tasks.Task.success_check` — the regex sanity-backup
predicate. The strict-JSON LLM judge from §2.2 is the *eval* primary,
not the training-loop primary: ~2k judge calls would double the API
budget and the judge is slower than the agent. Per §2.2 the regex is a
faithful sanity backup, and the goal here is to show the regex reward
*can't* be optimized by vanilla PPO — so any tightening from a smarter
reward only helps the experiment make its point.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import torch
import torch.nn as nn

from .policy import MultiCategoricalPolicy


@dataclass
class RolloutSample:
    obs: torch.Tensor       # (obs_dim,)
    action: torch.Tensor    # (n_slots,) long
    log_prob: float         # scalar
    value: float            # scalar
    reward: float           # scalar


@dataclass
class UpdateStats:
    step: int
    policy_loss: float
    value_loss: float
    entropy: float
    approx_kl: float
    clip_fraction: float
    mean_reward: float
    n_samples: int
    elapsed_s: float


@dataclass
class PPOConfig:
    lr: float = 3e-4
    clip_eps: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    n_epochs: int = 4
    minibatch_size: int = 64
    max_grad_norm: float = 0.5
    normalize_advantages: bool = True


class PPOSparseTrainer:
    def __init__(
        self,
        policy: MultiCategoricalPolicy,
        config: PPOConfig | None = None,
        device: str = "cpu",
    ):
        self.policy = policy.to(device)
        self.config = config or PPOConfig()
        self.device = device
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=self.config.lr
        )
        self._step = 0
        self._update_log: list[UpdateStats] = []

    @property
    def step(self) -> int:
        return self._step

    @property
    def update_log(self) -> list[UpdateStats]:
        return list(self._update_log)

    @torch.no_grad()
    def act(self, obs: torch.Tensor) -> tuple[torch.Tensor, float, float]:
        """Sample one action; return (action, log_prob, value)."""
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        out = self.policy.sample(obs.to(self.device))
        return (
            out.action.squeeze(0).cpu(),
            float(out.log_prob.item()),
            float(out.value.item()),
        )

    def update(self, batch: Sequence[RolloutSample]) -> UpdateStats:
        t0 = time.perf_counter()
        cfg = self.config

        obs = torch.stack([s.obs for s in batch]).to(self.device)
        actions = torch.stack([s.action for s in batch]).to(self.device)
        old_log_probs = torch.tensor(
            [s.log_prob for s in batch], dtype=torch.float32, device=self.device
        )
        old_values = torch.tensor(
            [s.value for s in batch], dtype=torch.float32, device=self.device
        )
        rewards = torch.tensor(
            [s.reward for s in batch], dtype=torch.float32, device=self.device
        )

        # One-step MDP: return == reward; advantage = reward - V_old.
        returns = rewards
        advantages = returns - old_values
        if cfg.normalize_advantages and advantages.numel() > 1:
            std = advantages.std()
            if torch.isfinite(std) and std.item() > 1e-6:
                advantages = (advantages - advantages.mean()) / (std + 1e-8)

        n = obs.size(0)
        policy_losses: list[float] = []
        value_losses: list[float] = []
        entropies: list[float] = []
        approx_kls: list[float] = []
        clip_fracs: list[float] = []

        for _ in range(cfg.n_epochs):
            perm = torch.randperm(n, device=self.device)
            for start in range(0, n, cfg.minibatch_size):
                idx = perm[start : start + cfg.minibatch_size]
                mb_obs = obs[idx]
                mb_actions = actions[idx]
                mb_old_logp = old_log_probs[idx]
                mb_adv = advantages[idx]
                mb_ret = returns[idx]

                new_logp, new_value, entropy = self.policy.evaluate(
                    mb_obs, mb_actions
                )
                ratio = torch.exp(new_logp - mb_old_logp)
                unclipped = ratio * mb_adv
                clipped = (
                    torch.clamp(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps)
                    * mb_adv
                )
                policy_loss = -torch.min(unclipped, clipped).mean()
                value_loss = (new_value - mb_ret).pow(2).mean()
                entropy_mean = entropy.mean()

                loss = (
                    policy_loss
                    + cfg.value_coef * value_loss
                    - cfg.entropy_coef * entropy_mean
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.policy.parameters(), cfg.max_grad_norm
                )
                self.optimizer.step()

                with torch.no_grad():
                    approx_kl = (mb_old_logp - new_logp).mean().clamp_min(0.0)
                    clip_frac = (
                        (ratio - 1.0).abs().gt(cfg.clip_eps).float().mean()
                    )

                policy_losses.append(float(policy_loss.item()))
                value_losses.append(float(value_loss.item()))
                entropies.append(float(entropy_mean.item()))
                approx_kls.append(float(approx_kl.item()))
                clip_fracs.append(float(clip_frac.item()))

        self._step += 1
        stats = UpdateStats(
            step=self._step,
            policy_loss=_mean(policy_losses),
            value_loss=_mean(value_losses),
            entropy=_mean(entropies),
            approx_kl=_mean(approx_kls),
            clip_fraction=_mean(clip_fracs),
            mean_reward=float(rewards.mean().item()),
            n_samples=n,
            elapsed_s=time.perf_counter() - t0,
        )
        self._update_log.append(stats)
        return stats


def _mean(xs: Iterable[float]) -> float:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0
