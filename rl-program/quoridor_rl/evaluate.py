"""Play games between agents and measure win rates."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .agents import as_agent
from .encoding import action_mask, encode_observation, index_to_action
from .env import QuoridorEnv
from .rules import Action, GameState, apply_action, render_ascii


@dataclass
class GameRecord:
    winner: Optional[int]          # 0, 1 or None (draw / truncated)
    turns: int
    actions: List[Action] = field(default_factory=list)
    final_state: Optional[GameState] = None


def play_game(agent0, agent1, max_turns: int = 200, verbose: bool = False) -> GameRecord:
    """Play one game between two Agents (agent0 = red, moves first). No pygame needed.
    Each agent gets observations in its own format (``agent.include_distances``, default True)."""
    agents = (as_agent(agent0), as_agent(agent1))
    formats = [getattr(a, 'include_distances', True) for a in agents]
    state = GameState.initial()
    actions: List[Action] = []
    while not state.is_terminal and state.turn < max_turns:
        p = state.current_player
        mask = action_mask(state)
        obs = encode_observation(state, include_distances=formats[p])
        index = int(agents[p].act(obs, mask))
        if not mask[index]:
            raise ValueError('Agent %i chose illegal action %i' % (p, index))
        action = index_to_action(state, index)
        actions.append(action)
        state = apply_action(state, action, validate=False)
        if verbose:
            print('P%i: %r\n%s\n' % (p, action, render_ascii(state)))
    return GameRecord(state.winner, state.turn, actions, state)


def evaluate(agent, opponent: Any = 'builtin', n_games: int = 20, opponent_level: int = 0,
             max_turns: int = 200, include_distances: Optional[bool] = None,
             seed: Optional[int] = None) -> Dict:
    """Win rate of ``agent`` against ``opponent`` ('random' | 'greedy' | 'heuristic' | 'builtin' | Agent).

    Sides alternate every game (even games: agent is red and moves first).
    include_distances: observation format; default = ``agent.include_distances`` (or True).
    Returns a dict with wins / losses / draws / win_rate / avg_turns / win_rate_first / win_rate_second.
    """
    agent = as_agent(agent)
    if include_distances is None:
        include_distances = getattr(agent, 'include_distances', True)
    stats = {'wins': 0, 'losses': 0, 'draws': 0, 'turns': 0, 'wins_first': 0, 'wins_second': 0}
    envs = {p: QuoridorEnv(opponent=opponent, opponent_level=opponent_level, agent_player=p,
                           max_turns=max_turns, include_distances=include_distances,
                           seed=None if seed is None else seed + p)
            for p in (0, 1)}
    t0 = time.time()
    for g in range(n_games):
        env = envs[g % 2]
        obs, info = env.reset()
        done = False
        reward = 0.0
        while not done:
            obs, reward, terminated, truncated, info = env.step(agent.act(obs, info['action_mask']))
            done = terminated or truncated
        stats['turns'] += info['turn']
        if info['winner'] is None:
            stats['draws'] += 1
        elif info['winner'] == env.agent_player:
            stats['wins'] += 1
            stats['wins_first' if env.agent_player == 0 else 'wins_second'] += 1
        else:
            stats['losses'] += 1
    n_first, n_second = (n_games + 1) // 2, n_games // 2
    return {
        'games': n_games,
        'wins': stats['wins'],
        'losses': stats['losses'],
        'draws': stats['draws'],
        'win_rate': stats['wins'] / max(n_games, 1),
        'win_rate_first': stats['wins_first'] / max(n_first, 1),
        'win_rate_second': stats['wins_second'] / max(n_second, 1),
        'avg_turns': stats['turns'] / max(n_games, 1),
        'seconds': time.time() - t0,
    }
