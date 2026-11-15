"""Helpers to apply the legal-action mask to network outputs.

Functions accept numpy arrays or torch tensors (torch is only imported if you pass tensors).
Masks are bool with True = legal; they may carry a leading batch dimension.
"""
from __future__ import annotations

import numpy as np

NEG_INF = -1e9  # large negative instead of -inf: avoids NaNs when a row is fully masked


def _is_torch(x) -> bool:
    return type(x).__module__.startswith('torch')


def masked_fill(values, mask, fill: float = NEG_INF):
    """Return ``values`` with illegal entries replaced by ``fill`` (use on logits or Q-values)."""
    if _is_torch(values):
        import torch
        mask = torch.as_tensor(mask, dtype=torch.bool, device=values.device)
        return values.masked_fill(~mask, fill)
    values = np.asarray(values, dtype=np.float64)
    return np.where(np.asarray(mask, dtype=bool), values, fill)


def masked_argmax(values, mask):
    """Index of the best legal action (greedy DQN action / greedy policy)."""
    m = masked_fill(values, mask)
    if _is_torch(m):
        return m.argmax(dim=-1)
    return np.argmax(m, axis=-1)


def masked_softmax(logits, mask):
    """Probability distribution over legal actions only."""
    m = masked_fill(logits, mask)
    if _is_torch(m):
        import torch
        return torch.softmax(m, dim=-1)
    m = m - m.max(axis=-1, keepdims=True)
    e = np.exp(m) * np.asarray(mask, dtype=bool)
    return e / e.sum(axis=-1, keepdims=True)


def sample_masked(logits, mask, rng: np.random.Generator = None) -> int:
    """Sample one legal action from softmax(logits) (numpy, single state)."""
    rng = rng or np.random.default_rng()
    probs = masked_softmax(np.asarray(logits, dtype=np.float64), mask)
    return int(rng.choice(len(probs), p=probs))


def epsilon_greedy(values, mask, epsilon: float, rng: np.random.Generator = None) -> int:
    """With probability epsilon a random legal action, otherwise the masked argmax (numpy)."""
    rng = rng or np.random.default_rng()
    if rng.random() < epsilon:
        return int(rng.choice(np.flatnonzero(mask)))
    return int(masked_argmax(np.asarray(values), mask))
