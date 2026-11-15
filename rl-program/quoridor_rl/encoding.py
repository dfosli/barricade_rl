"""State / action encodings for neural networks.

Everything here is expressed from the point of view of the player to move ("canonical" frame):
the board is flipped vertically for player 1, so the network always sees *itself* starting at
the bottom and heading for row 0. One network can therefore play both sides.

Action space (``NUM_ACTIONS = 140``), canonical frame:

====================  =========================================================
index                 meaning
====================  =========================================================
0..11                 pawn move by ``MOVE_OFFSETS[index]`` = (d_row, d_col)
12..75                horizontal wall at (i, j), index = 12 + i*8 + j
76..139               vertical wall at (i, j),   index = 76 + i*8 + j
====================  =========================================================

Observation vector (``OBS_SIZE = 294``, or ``OBS_SIZE_NO_DIST = 292``), canonical frame:

====================  =========================================================
slice                 content
====================  =========================================================
0..80                 my pawn, one-hot 9x9
81..161               opponent pawn, one-hot 9x9
162..225              horizontal walls 8x8 (1 = wall)
226..289              vertical walls 8x8
290, 291              my walls left / 10, opponent walls left / 10
292, 293              my / opponent shortest path length / DIST_SCALE (optional)
====================  =========================================================
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .rules import (BOARD_SIZE, NUM_WALLS, UNREACHABLE, WALL_GRID, Action, GameState, PawnMove,
                    WallPlacement, distance_map, legal_pawn_moves, legal_walls)

# ----------------------------- action space ----------------------------- #
MOVE_OFFSETS = (
    (-1, 0), (1, 0), (0, -1), (0, 1),      # 0-3  single step: forward, back, left, right
    (-2, 0), (2, 0), (0, -2), (0, 2),      # 4-7  straight jump over the opponent
    (-1, -1), (-1, 1), (1, -1), (1, 1),    # 8-11 diagonal jump around the opponent
)
NUM_MOVE_ACTIONS = len(MOVE_OFFSETS)
NUM_WALL_CELLS = WALL_GRID * WALL_GRID
H_WALL_START = NUM_MOVE_ACTIONS
V_WALL_START = H_WALL_START + NUM_WALL_CELLS
NUM_ACTIONS = V_WALL_START + NUM_WALL_CELLS  # 140
_OFFSET_INDEX = {o: i for i, o in enumerate(MOVE_OFFSETS)}

# ----------------------------- observation ----------------------------- #
DIST_SCALE = 20.0
_CELLS = BOARD_SIZE * BOARD_SIZE
OBS_SLICES = {
    'my_pawn': slice(0, _CELLS),
    'opp_pawn': slice(_CELLS, 2 * _CELLS),
    'h_walls': slice(2 * _CELLS, 2 * _CELLS + NUM_WALL_CELLS),
    'v_walls': slice(2 * _CELLS + NUM_WALL_CELLS, 2 * _CELLS + 2 * NUM_WALL_CELLS),
    'walls_left': slice(2 * _CELLS + 2 * NUM_WALL_CELLS, 2 * _CELLS + 2 * NUM_WALL_CELLS + 2),
    'distances': slice(2 * _CELLS + 2 * NUM_WALL_CELLS + 2, 2 * _CELLS + 2 * NUM_WALL_CELLS + 4),
}
OBS_SIZE = OBS_SLICES['distances'].stop          # 294
OBS_SIZE_NO_DIST = OBS_SLICES['walls_left'].stop  # 292
NUM_PLANES = 8


def obs_size(include_distances: bool = True) -> int:
    return OBS_SIZE if include_distances else OBS_SIZE_NO_DIST


# ----------------------------- canonical frame ----------------------------- #
def flip_cell(cell):
    """Vertical flip of a board cell (row, col) -> (8-row, col). It is its own inverse."""
    return BOARD_SIZE - 1 - cell[0], cell[1]


def canonical_state(state: GameState, player: Optional[int] = None) -> GameState:
    """Return the state seen from ``player`` (default: player to move): that player becomes
    player 0, the board is flipped vertically if needed. Applying it twice gives back the original."""
    p = state.current_player if player is None else player
    if p == 0:
        return state
    return GameState(
        pawns=(flip_cell(state.pawns[1]), flip_cell(state.pawns[0])),
        walls_left=(state.walls_left[1], state.walls_left[0]),
        h_walls=np.ascontiguousarray(state.h_walls[::-1]),
        v_walls=np.ascontiguousarray(state.v_walls[::-1]),
        current_player=1 - state.current_player,
        turn=state.turn,
    )


# ----------------------------- actions ----------------------------- #
def action_to_index(state: GameState, action: Action) -> int:
    """Absolute action of the player to move -> canonical action index (0..139)."""
    flip = state.current_player == 1
    if isinstance(action, PawnMove):
        r, c = state.pawns[state.current_player]
        dr, dc = action.row - r, action.col - c
        return _OFFSET_INDEX[(-dr if flip else dr, dc)]
    i = WALL_GRID - 1 - action.row if flip else action.row
    return (H_WALL_START if action.horizontal else V_WALL_START) + i * WALL_GRID + action.col


def index_to_action(state: GameState, index: int) -> Action:
    """Canonical action index (0..139) of the player to move -> absolute action."""
    index = int(index)
    flip = state.current_player == 1
    if index < NUM_MOVE_ACTIONS:
        dr, dc = MOVE_OFFSETS[index]
        r, c = state.pawns[state.current_player]
        return PawnMove(r - dr if flip else r + dr, c + dc)
    horizontal = index < V_WALL_START
    k = index - (H_WALL_START if horizontal else V_WALL_START)
    i, j = divmod(k, WALL_GRID)
    return WallPlacement(WALL_GRID - 1 - i if flip else i, j, horizontal)


def action_mask(state: GameState) -> np.ndarray:
    """(140,) bool array, True = legal for the player to move (canonical indices)."""
    mask = np.zeros(NUM_ACTIONS, dtype=bool)
    if state.is_terminal:
        return mask
    cs = canonical_state(state)
    r, c = cs.pawns[0]
    for nr, nc in legal_pawn_moves(cs, 0):
        mask[_OFFSET_INDEX[(nr - r, nc - c)]] = True
    h, v = legal_walls(cs, 0)
    mask[H_WALL_START:V_WALL_START] = h.ravel()
    mask[V_WALL_START:] = v.ravel()
    return mask


def describe_action(index: int) -> str:
    """Human readable description of a canonical action index."""
    index = int(index)
    if index < NUM_MOVE_ACTIONS:
        names = ['forward', 'back', 'left', 'right', 'jump forward', 'jump back', 'jump left',
                 'jump right', 'diag fwd-left', 'diag fwd-right', 'diag back-left', 'diag back-right']
        return 'move %s %s' % (names[index], MOVE_OFFSETS[index])
    horizontal = index < V_WALL_START
    i, j = divmod(index - (H_WALL_START if horizontal else V_WALL_START), WALL_GRID)
    return '%s wall at (%i, %i)' % ('horizontal' if horizontal else 'vertical', i, j)


# ----------------------------- observations ----------------------------- #
def encode_observation(state: GameState, player: Optional[int] = None,
                       include_distances: bool = True) -> np.ndarray:
    """Flat float32 vector for an MLP, from ``player``'s view (default: player to move)."""
    cs = canonical_state(state, player)
    obs = np.zeros(obs_size(include_distances), dtype=np.float32)
    (mr, mc), (orow, oc) = cs.pawns
    obs[OBS_SLICES['my_pawn'].start + mr * BOARD_SIZE + mc] = 1.0
    obs[OBS_SLICES['opp_pawn'].start + orow * BOARD_SIZE + oc] = 1.0
    obs[OBS_SLICES['h_walls']] = cs.h_walls.ravel()
    obs[OBS_SLICES['v_walls']] = cs.v_walls.ravel()
    obs[OBS_SLICES['walls_left']] = np.array(cs.walls_left, dtype=np.float32) / NUM_WALLS
    if include_distances:
        d0 = distance_map(cs, 0)[mr, mc]
        d1 = distance_map(cs, 1)[orow, oc]
        obs[OBS_SLICES['distances']] = (d0 / DIST_SCALE, d1 / DIST_SCALE)
    return obs


def encode_planes(state: GameState, player: Optional[int] = None) -> np.ndarray:
    """(8, 9, 9) float32 planes for a CNN (for later experiments), canonical frame:
    my pawn, opp pawn, h walls, v walls (padded to 9x9), my walls left, opp walls left,
    my distance-to-goal map, opp distance-to-goal map (both / DIST_SCALE, unreachable = 0)."""
    cs = canonical_state(state, player)
    planes = np.zeros((NUM_PLANES, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    planes[0][cs.pawns[0]] = 1.0
    planes[1][cs.pawns[1]] = 1.0
    planes[2, :WALL_GRID, :WALL_GRID] = cs.h_walls
    planes[3, :WALL_GRID, :WALL_GRID] = cs.v_walls
    planes[4] = cs.walls_left[0] / NUM_WALLS
    planes[5] = cs.walls_left[1] / NUM_WALLS
    for k, q in ((6, 0), (7, 1)):
        d = distance_map(cs, q).astype(np.float32)
        planes[k] = np.where(d >= UNREACHABLE, 0.0, d / DIST_SCALE)
    return planes


def observation_to_state(obs: np.ndarray) -> GameState:
    """Inverse of :func:`encode_observation`: the canonical GameState (you = player 0, to move).
    Useful for debugging and for rule-based agents that only receive observations."""
    obs = np.asarray(obs)
    me = divmod(int(np.argmax(obs[OBS_SLICES['my_pawn']])), BOARD_SIZE)
    opp = divmod(int(np.argmax(obs[OBS_SLICES['opp_pawn']])), BOARD_SIZE)
    walls = np.rint(obs[OBS_SLICES['walls_left']] * NUM_WALLS).astype(int)
    return GameState(
        pawns=(me, opp),
        walls_left=(int(walls[0]), int(walls[1])),
        h_walls=obs[OBS_SLICES['h_walls']].reshape(WALL_GRID, WALL_GRID) > 0.5,
        v_walls=obs[OBS_SLICES['v_walls']].reshape(WALL_GRID, WALL_GRID) > 0.5,
        current_player=0,
    )


# ----------------------------- left/right mirror (data augmentation) ----------------------------- #
def _mirror_perm() -> np.ndarray:
    perm = np.zeros(NUM_ACTIONS, dtype=np.int64)
    for i, (dr, dc) in enumerate(MOVE_OFFSETS):
        perm[i] = _OFFSET_INDEX[(dr, -dc)]
    for start in (H_WALL_START, V_WALL_START):
        for i in range(WALL_GRID):
            for j in range(WALL_GRID):
                perm[start + i * WALL_GRID + j] = start + i * WALL_GRID + (WALL_GRID - 1 - j)
    return perm


MIRROR_ACTION_PERM = _mirror_perm()  # mirrored index of action a = MIRROR_ACTION_PERM[a] (self-inverse)


def mirror_action(index):
    """Action index in the left-right mirrored position. Works on ints and int arrays."""
    return MIRROR_ACTION_PERM[index]


def mirror_action_vector(values: np.ndarray) -> np.ndarray:
    """Mirror anything indexed by action on its last axis (mask, Q-values, policy, ...)."""
    return values[..., MIRROR_ACTION_PERM]


def mirror_observation(obs: np.ndarray) -> np.ndarray:
    """Left-right mirror of an observation vector (or a batch, last axis = features)."""
    out = np.array(obs, copy=True)
    for key, size in (('my_pawn', BOARD_SIZE), ('opp_pawn', BOARD_SIZE),
                      ('h_walls', WALL_GRID), ('v_walls', WALL_GRID)):
        s = OBS_SLICES[key]
        block = out[..., s].reshape(out.shape[:-1] + (size, size))
        out[..., s] = block[..., ::-1].reshape(out.shape[:-1] + (size * size,))
    return out
