"""Glue between the pure-Python rules (``quoridor_rl.rules``) and the pygame engine (``engine/src``).

* :func:`new_engine_board` – headless engine Board (no window), optionally with built-in AIs
* :func:`state_from_board` / :func:`apply_to_board` – convert / sync between the two worlds
* :class:`BuiltinAI` – the engine's minimax AI as an opponent for the RL env
* :class:`AgentController` – plugs any ``Agent`` into the engine (so it can play in the GUI)
* :func:`launch_gui` – open the game window with your agent selectable in the menu
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Callable, Dict, Iterable, Optional

import numpy as np

from .encoding import action_mask, encode_observation, index_to_action
from .rules import NUM_WALLS, WALL_GRID, Action, GameState, PawnMove, WallPlacement

ENGINE_DIR = Path(__file__).resolve().parents[2] / 'engine'
if str(ENGINE_DIR) not in sys.path:
    sys.path.insert(0, str(ENGINE_DIR))

os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1')
import pygame  # noqa: E402

from src import config as engine_cfg  # noqa: E402
from src import core as engine_core  # noqa: E402
from src import logger as engine_logger  # noqa: E402
from src.ai.action import ActionMovePawn, ActionPlaceWall  # noqa: E402
from src.ai.ai import AI  # noqa: E402
from src.components.board import Board  # noqa: E402
from src.components.coord import Coord  # noqa: E402


def set_engine_verbose(verbose: bool) -> None:
    """The engine prints every move at INFO level; silence it for training."""
    engine_logger.LOG_LEVEL = engine_logger.LogLevel.INFO if verbose else engine_logger.LogLevel.WARN


def new_engine_board(ai_players: Iterable[int] = (), ai_level: int = 0, verbose: bool = False) -> Board:
    """Create a fresh headless engine Board (nothing is drawn, no window needed)."""
    set_engine_verbose(verbose)
    engine_cfg.CACHE_ENABLED = False
    engine_core.init()
    board = Board(pygame.Surface((800, 600)), ai_players=tuple(ai_players), ai_level=ai_level, headless=True)
    engine_core.BOARD = board
    return board


def state_from_board(board: Board, turn: int = 0) -> GameState:
    """Read the engine Board into a GameState."""
    h = np.zeros((WALL_GRID, WALL_GRID), dtype=bool)
    v = np.zeros((WALL_GRID, WALL_GRID), dtype=bool)
    for wall in board.walls:
        (h if wall.horiz else v)[wall.coord.row, wall.coord.col] = True
    pawns = tuple((p.coord.row, p.coord.col) for p in board.pawns)
    walls_left = tuple(int(p.walls) for p in board.pawns)
    return GameState(pawns, walls_left, h, v, board.player, turn)


def to_engine_action(board: Board, action: Action):
    """Our Action -> engine ActionMovePawn / ActionPlaceWall (for the current player)."""
    if isinstance(action, PawnMove):
        return ActionMovePawn(board.current_player.coord, Coord(action.row, action.col))
    return ActionPlaceWall(board.new_wall(Coord(action.row, action.col), action.horizontal))


def from_engine_action(engine_action) -> Action:
    """Engine ActionMovePawn / ActionPlaceWall -> our Action."""
    if isinstance(engine_action, ActionPlaceWall):
        return WallPlacement(engine_action.coord.row, engine_action.coord.col, bool(engine_action.horiz))
    return PawnMove(engine_action.dest.row, engine_action.dest.col)


def apply_to_board(board: Board, action: Action) -> None:
    """Play ``action`` for the current player on the engine board and pass the turn."""
    board.do_action(to_engine_action(board, action))
    if not board.finished:
        board.next_player()


class BuiltinAI:
    """The engine's alpha-beta AI, kept in sync with a GameState, usable as an env opponent.

    Call :meth:`reset` at the start of a game, :meth:`observe` after *every* action (both players),
    and :meth:`choose` when it is the AI's turn.
    """

    def __init__(self, level: int = 0, verbose: bool = False):
        self.level = level
        self.verbose = verbose
        self.board: Optional[Board] = None
        self.ai: Optional[AI] = None

    def reset(self, ai_player: int, first_player: int = 0) -> None:
        self.board = new_engine_board(ai_players=(ai_player,), ai_level=self.level, verbose=self.verbose)
        self.board.player = first_player
        self.ai = self.board.pawns[ai_player].AI

    def observe(self, action: Action) -> None:
        apply_to_board(self.board, action)

    def choose(self) -> Action:
        engine_action, _value = self.ai.move()
        return from_engine_action(engine_action)


class AgentController:
    """Makes any Agent behave like the engine's AI class (has ``.move()``), so it can play in the GUI.

    The board passed in is the live GUI board; the agent sees the usual observation and mask.
    The observation format follows ``agent.include_distances`` if the agent has it (models do).
    """

    def __init__(self, agent, board: Board, include_distances: Optional[bool] = None):
        self.agent = agent
        self.board = board
        if include_distances is None:
            include_distances = getattr(agent, 'include_distances', True)
        self.include_distances = include_distances

    def move(self):
        state = state_from_board(self.board)
        mask = action_mask(state)
        obs = encode_observation(state, include_distances=self.include_distances)
        index = int(self.agent.act(obs, mask))
        if not mask[index]:
            legal = np.flatnonzero(mask)
            print('WARNING: agent chose illegal action %i, playing %i instead' % (index, legal[0]))
            index = int(legal[0])
        return to_engine_action(self.board, index_to_action(state, index)), 0

    def flush_cache(self) -> None:
        pass


def launch_gui(agents: Optional[Dict[str, Callable[[], object]]] = None) -> int:
    """Open the game window. ``agents`` maps a menu label to a factory returning an Agent, e.g.
    ``{'RL: mlp': lambda: load_agent('mlp'), 'Heuristic bot': HeuristicAgent}``. Each label becomes
    a selectable player type next to Human / Built-in AI. Factories are called when a game starts."""
    from src.main import main as engine_main

    def controller_factory(make_agent):
        return lambda board, pawn: AgentController(make_agent(), board)

    extra = {label: controller_factory(make) for label, make in (agents or {}).items()}
    set_engine_verbose(True)
    return engine_main(extra_players=extra)


__all__ = [
    'ENGINE_DIR', 'Board', 'AI', 'ActionMovePawn', 'ActionPlaceWall', 'Coord', 'NUM_WALLS',
    'set_engine_verbose', 'new_engine_board', 'state_from_board', 'to_engine_action',
    'from_engine_action', 'apply_to_board', 'BuiltinAI', 'AgentController', 'launch_gui',
]
