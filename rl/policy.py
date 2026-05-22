"""Actor-critic policy for Exp 3 (plan §3.2).

Multi-categorical action: one categorical head per payload slot over a
top-2k BPE vocab. Joint log-prob is the sum over slot heads (the slots
are independent under the policy — the only coupling is through the
shared MLP backbone). The state passed to the policy is a constant 1-d
zero (Exp 3's MDP is effectively a contextual bandit on a fixed task),
so the "MLP" is really a learnable bias plus a small value head; the
plan §3.2 still calls for an MLP, and keeping the structure here means
the same policy class drops into the composite-reward run later.

The point of Exp 3 is to show vanilla PPO can't find the (≤10⁷ winning /
1.6e13 total)-needle in the slot-product action space when the reward
signal is +1/0 on the task success regex. So this policy is meant to be
*small and ordinary*, not tuned for the task.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.distributions import Categorical


@dataclass
class PolicyOutput:
    action: torch.Tensor       # (B, n_slots) long
    log_prob: torch.Tensor     # (B,) float
    value: torch.Tensor        # (B,) float
    entropy: torch.Tensor      # (B,) float


class MultiCategoricalPolicy(nn.Module):
    def __init__(
        self,
        n_slots: int,
        vocab_size: int,
        hidden_dim: int = 64,
        obs_dim: int = 1,
    ):
        super().__init__()
        self.n_slots = n_slots
        self.vocab_size = vocab_size
        self.obs_dim = obs_dim

        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.policy_head = nn.Linear(hidden_dim, n_slots * vocab_size)
        self.value_head = nn.Linear(hidden_dim, 1)

        # Small init on the policy head so initial logits are near-uniform
        # (otherwise random init can over-weight a small number of tokens
        # and bias the first few thousand samples).
        nn.init.orthogonal_(self.policy_head.weight, gain=0.01)
        nn.init.zeros_(self.policy_head.bias)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        nn.init.zeros_(self.value_head.bias)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.backbone(obs)
        logits = self.policy_head(h).view(-1, self.n_slots, self.vocab_size)
        value = self.value_head(h).squeeze(-1)
        return logits, value

    @torch.no_grad()
    def sample(self, obs: torch.Tensor) -> PolicyOutput:
        logits, value = self.forward(obs)
        dist = Categorical(logits=logits)
        action = dist.sample()                       # (B, n_slots)
        log_prob = dist.log_prob(action).sum(-1)     # (B,)
        entropy = dist.entropy().sum(-1)             # (B,)
        return PolicyOutput(
            action=action,
            log_prob=log_prob,
            value=value,
            entropy=entropy,
        )

    def evaluate(
        self, obs: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, value = self.forward(obs)
        dist = Categorical(logits=logits)
        log_prob = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)
        return log_prob, value, entropy
