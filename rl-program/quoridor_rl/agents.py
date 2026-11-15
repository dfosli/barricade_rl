"""Agent interface and simple baseline agents.

Every agent implements ``act(obs, mask) -> int``:

* ``obs``  – observation vector from :func:`quoridor_rl.encoding.encode_observation`
* ``mask`` – (140,) bool array of legal actions
* returns a canonical action index (0..139) where ``mask[index]`` is True

That is all the env, the evaluator and the GUI need, so your RL agent only has to implement it.
"""
from __future__ import annotations

from typing import Callable, Optional, Protocol, runtime_checkable

import numpy as np

from .encoding import H_WALL_START, NUM_MOVE_ACTIONS, V_WALL_START, index_to_action, observation_to_state
from .rules import apply_action, distance_map, path_length, shortest_path, walls_on_path


@runtime_checkable
class Agent(Protocol):
    def act(self, obs: np.ndarray, mask: np.ndarray) -> int:
        ...


class RandomAgent:
    """Uniformly random legal action.

    move_probability: if set, first decide "pawn move" (with this probability) vs "wall",
    then pick uniformly inside that group. Plain uniform play spends all walls very quickly,
    because ~100 of the ~105 legal actions are walls.
    """

    def __init__(self, seed: Optional[int] = None, move_probability: Optional[float] = None):
        self.rng = np.random.default_rng(seed)
        self.move_probability = move_probability

    def act(self, obs: np.ndarray, mask: np.ndarray) -> int:
        legal = np.flatnonzero(mask)
        if self.move_probability is not None:
            moves, walls = legal[legal < NUM_MOVE_ACTIONS], legal[legal >= NUM_MOVE_ACTIONS]
            if len(walls) and (not len(moves) or self.rng.random() >= self.move_probability):
                legal = walls
            elif len(moves):
                legal = moves
        return int(self.rng.choice(legal))


class GreedyPathAgent:
    """Always steps along a shortest path to the goal, never places walls.
    With ``epsilon`` > 0 it plays a random legal action that fraction of the time.
    A useful, cheap sanity-check opponent: an agent that can't beat it hasn't learned walls."""

    def __init__(self, epsilon: float = 0.0, seed: Optional[int] = None):
        self.epsilon = epsilon
        self.rng = np.random.default_rng(seed)

    def act(self, obs: np.ndarray, mask: np.ndarray) -> int:
        legal = np.flatnonzero(mask)
        if self.epsilon and self.rng.random() < self.epsilon:
            return int(self.rng.choice(legal))
        state = observation_to_state(obs)  # canonical: we are player 0
        dist = distance_map(state, 0)
        moves = [a for a in legal if a < NUM_MOVE_ACTIONS]
        if not moves:
            return int(self.rng.choice(legal))
        scores = []
        for a in moves:
            dest = index_to_action(state, a)
            scores.append(dist[dest.row, dest.col])
        best = min(scores)
        return int(self.rng.choice([a for a, s in zip(moves, scores) if s == best]))


class HeuristicAgent:
    """1-ply search, the same idea as the engine's built-in AI at level 0, but ~100x faster:
    plays the action that minimises (my path length - opponent path length) afterwards.
    Ties prefer pawn moves (saves walls), then random. ``epsilon`` adds random play.
    Good training opponent; use opponent='builtin' only for occasional evaluation."""

    def __init__(self, epsilon: float = 0.0, seed: Optional[int] = None):
        self.epsilon = epsilon
        self.rng = np.random.default_rng(seed)

    def act(self, obs: np.ndarray, mask: np.ndarray) -> int:
        legal = np.flatnonzero(mask)
        if self.epsilon and self.rng.random() < self.epsilon:
            return int(self.rng.choice(legal))
        state = observation_to_state(obs)  # canonical: we are player 0, heading to row 0
        my_path, opp_path = shortest_path(state, 0), shortest_path(state, 1)
        base = len(my_path) - len(opp_path)
        # Only walls touching one of the two shortest paths can change the score
        h_my, v_my = walls_on_path(my_path)
        h_opp, v_opp = walls_on_path(opp_path)
        relevant = np.zeros(len(mask), dtype=bool)
        relevant[H_WALL_START:V_WALL_START] = (h_my | h_opp).ravel()
        relevant[V_WALL_START:] = (v_my | v_opp).ravel()

        my_dist = distance_map(state, 0)
        best_score, best = None, []
        for a in legal:
            action = index_to_action(state, a)
            if a < NUM_MOVE_ACTIONS:
                if action.row == 0:
                    return int(a)  # winning move
                score = int(my_dist[action.row, action.col]) - (len(opp_path) - 1)
            elif relevant[a]:
                nxt = apply_action(state, action, validate=False)
                score = path_length(nxt, 0) - path_length(nxt, 1)
            else:
                score = base
            key = (score, a >= NUM_MOVE_ACTIONS)  # lower is better; moves win ties
            if best_score is None or key < best_score:
                best_score, best = key, [a]
            elif key == best_score:
                best.append(a)
        return int(self.rng.choice(best))


class FunctionAgent:
    """Wraps a plain function ``f(obs, mask) -> int`` into an Agent."""

    def __init__(self, fn: Callable[[np.ndarray, np.ndarray], int]):
        self.fn = fn

    def act(self, obs: np.ndarray, mask: np.ndarray) -> int:
        return int(self.fn(obs, mask))


def as_agent(obj) -> Agent:
    """Accept an Agent or a function and return an Agent."""
    if hasattr(obj, 'act'):
        return obj
    if callable(obj):
        return FunctionAgent(obj)
    raise TypeError('Expected an object with .act(obs, mask) or a callable, got %r' % (obj,))
