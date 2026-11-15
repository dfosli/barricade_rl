"""Gym-style Quoridor environment (no gym dependency).

    env = QuoridorEnv(opponent='builtin', opponent_level=0)
    obs, info = env.reset()
    while True:
        action = my_agent.act(obs, info['action_mask'])
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break

Two modes:

* **vs. opponent** (``opponent`` = 'random' | 'greedy' | 'heuristic' | 'builtin' | Agent | function):
  each ``step`` plays your action *and* the opponent's reply. Observations, masks and rewards are
  always for your agent (``env.agent_player``).
* **self-play** (``opponent=None``): each ``step`` plays one action for whoever is to move.
  The returned observation/mask are for the *next* player to move (``info['current_player']``), and
  the reward is for the player who just moved (``info['mover']``). When a game ends, remember that
  the loser's last transition should also get -1.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

from .agents import GreedyPathAgent, HeuristicAgent, RandomAgent, as_agent
from .encoding import NUM_ACTIONS, action_mask, encode_observation, index_to_action, obs_size
from .rules import Action, GameState, apply_action, path_length, render_ascii


class QuoridorEnv:
    def __init__(self,
                 opponent: Union[None, str, Any] = 'random',
                 opponent_level: int = 0,
                 agent_player: Union[int, str] = 0,
                 max_turns: int = 200,
                 include_distances: bool = True,
                 distance_reward: float = 0.0,
                 seed: Optional[int] = None):
        """
        opponent:          None (self-play), 'random', 'greedy', 'heuristic', 'builtin' (engine AI),
                           or any Agent / function(obs, mask) -> action index.
        opponent_level:    search depth of the 'builtin' opponent (0 = fastest/weakest).
        agent_player:      0 (red, moves first), 1 (blue, moves second) or 'random' (each reset).
        max_turns:         total actions (both players) before the game is truncated (draw).
        include_distances: add the two shortest-path lengths to the observation (294 vs 292 values).
        distance_reward:   reward shaping coefficient: + coef * change of (opp_dist - my_dist)
                           per step. 0 = only the sparse +1 / -1 at the end.
        """
        self.rng = np.random.default_rng(seed)
        self.opponent_kind = opponent
        self.opponent_level = opponent_level
        self.agent_player_setting = agent_player
        self.max_turns = max_turns
        self.include_distances = include_distances
        self.distance_reward = distance_reward

        self.self_play = opponent is None
        self._builtin = None
        self._opponent = None
        if opponent == 'builtin':
            from .engine_bridge import BuiltinAI  # imports pygame, so only when needed
            self._builtin = BuiltinAI(level=opponent_level)
        elif opponent == 'random':
            self._opponent = RandomAgent(seed=None if seed is None else seed + 1)
        elif opponent == 'greedy':
            self._opponent = GreedyPathAgent(seed=None if seed is None else seed + 1)
        elif opponent == 'heuristic':
            self._opponent = HeuristicAgent(seed=None if seed is None else seed + 1)
        elif opponent is not None:
            self._opponent = as_agent(opponent)

        self.state: GameState = GameState.initial()
        self.agent_player = 0
        self.last_opponent_action: Optional[Action] = None
        self._mask_cache = (None, None)

    # ------------------------------------------------------------------ properties
    @property
    def observation_size(self) -> int:
        return obs_size(self.include_distances)

    @property
    def num_actions(self) -> int:
        return NUM_ACTIONS

    @property
    def current_player(self) -> int:
        return self.state.current_player

    # ------------------------------------------------------------------ core API
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.state = GameState.initial()
        self.last_opponent_action = None
        if self.agent_player_setting == 'random':
            self.agent_player = int(self.rng.integers(2))
        else:
            self.agent_player = int(self.agent_player_setting)

        if self._builtin is not None:
            self._builtin.reset(ai_player=1 - self.agent_player)
        if not self.self_play and self.agent_player == 1:
            self._opponent_turn()
        return self._obs(), self._info(mover=None)

    def step(self, action: int):
        """Play canonical action index ``action`` (0..139) for the player to move.

        Returns (obs, reward, terminated, truncated, info).
        """
        if self.state.is_terminal:
            raise RuntimeError('Game is over, call reset()')
        mask = self.action_mask()
        if not mask[int(action)]:
            raise ValueError('Illegal action %i for player %i. Use info["action_mask"].'
                             % (int(action), self.state.current_player))

        mover = self.state.current_player
        me = mover if self.self_play else self.agent_player
        before = self._dist_diff(me) if self.distance_reward else 0.0

        self._play(index_to_action(self.state, int(action)))
        if not self.self_play and not self.state.is_terminal:
            self._opponent_turn()

        reward = 0.0
        winner = self.state.winner
        if winner is not None:
            reward = 1.0 if winner == me else -1.0
        elif self.distance_reward:
            reward = self.distance_reward * (self._dist_diff(me) - before)

        terminated = winner is not None
        truncated = not terminated and self.state.turn >= self.max_turns
        return self._obs(), reward, terminated, truncated, self._info(mover=mover)

    def action_mask(self) -> np.ndarray:
        """(140,) bool mask of legal actions for the player to move (cached per state)."""
        if self._mask_cache[0] is not self.state:
            self._mask_cache = (self.state, action_mask(self.state))
        return self._mask_cache[1]

    def legal_actions(self) -> np.ndarray:
        return np.flatnonzero(self.action_mask())

    def sample_action(self) -> int:
        """Random legal action (handy for epsilon-greedy exploration)."""
        return int(self.rng.choice(self.legal_actions()))

    def render(self) -> str:
        """ASCII picture of the board (absolute coordinates). Print it."""
        return render_ascii(self.state)

    # ------------------------------------------------------------------ internals
    def _play(self, action: Action) -> None:
        self.state = apply_action(self.state, action, validate=False)
        if self._builtin is not None:
            self._builtin.observe(action)

    def _opponent_turn(self) -> None:
        if self._builtin is not None:
            action = self._builtin.choose()
        else:
            mask = self.action_mask()
            fmt = getattr(self._opponent, 'include_distances', self.include_distances)
            obs = encode_observation(self.state, include_distances=fmt)
            action = index_to_action(self.state, self._opponent.act(obs, mask))
        self.last_opponent_action = action
        self._play(action)

    def _dist_diff(self, player: int) -> float:
        return float(path_length(self.state, 1 - player) - path_length(self.state, player))

    def _viewer(self) -> int:
        return self.state.current_player if self.self_play else self.agent_player

    def _obs(self) -> np.ndarray:
        return encode_observation(self.state, player=self._viewer(), include_distances=self.include_distances)

    def _info(self, mover: Optional[int]) -> Dict:
        viewer = self._viewer()
        if self.state.is_terminal or viewer != self.state.current_player:
            mask = np.zeros(NUM_ACTIONS, dtype=bool)
        else:
            mask = self.action_mask().copy()
        return {
            'action_mask': mask,
            'current_player': self.state.current_player,
            'agent_player': self.agent_player,
            'mover': mover,
            'winner': self.state.winner,
            'turn': self.state.turn,
            'walls_left': self.state.walls_left,
            'last_opponent_action': self.last_opponent_action,
        }
