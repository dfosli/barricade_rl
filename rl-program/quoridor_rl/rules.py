"""Fast, pygame-free Quoridor rules that mirror the engine in ``engine/src``.

Coordinates are absolute board coordinates ``(row, col)``, 0..8:

* Player 0 (red) starts at (8, 4) and must reach row 0.
* Player 1 (blue) starts at (0, 4) and must reach row 8.

Walls are anchored on an 8x8 grid, exactly like the engine's ``Wall(coord, horiz)``:

* horizontal wall (i, j) blocks moving between rows i and i+1 in columns j and j+1
* vertical wall   (i, j) blocks moving between cols j and j+1 in rows    i and i+1

Pawn moves follow the engine: when the opponent is adjacent you jump straight over it. Only when
that is blocked (a wall or the board edge directly behind the opponent) may you land on either
side of it instead (L-shaped / diagonal move).
A wall is legal when it does not overlap/cross another wall and both players still have a path
to their goal row (pawns are ignored for the path test, as in the official rules).
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

BOARD_SIZE = 9
WALL_GRID = BOARD_SIZE - 1
NUM_WALLS = 10
START_POSITIONS = ((8, 4), (0, 4))
GOAL_ROWS = (0, 8)
UNREACHABLE = 999

Cell = Tuple[int, int]
# Orthogonal directions as (d_row, d_col): up, down, left, right
DIRECTIONS = ((-1, 0), (1, 0), (0, -1), (0, 1))


@dataclass(frozen=True)
class PawnMove:
    """Move the current player's pawn to the absolute cell (row, col)."""
    row: int
    col: int


@dataclass(frozen=True)
class WallPlacement:
    """Place a wall anchored at (row, col) on the 8x8 wall grid."""
    row: int
    col: int
    horizontal: bool


Action = Union[PawnMove, WallPlacement]


@dataclass
class GameState:
    """Complete game state. Treat it as immutable: :func:`apply_action` returns a new state."""
    pawns: Tuple[Cell, Cell] = START_POSITIONS
    walls_left: Tuple[int, int] = (NUM_WALLS, NUM_WALLS)
    h_walls: np.ndarray = field(default_factory=lambda: np.zeros((WALL_GRID, WALL_GRID), dtype=bool))
    v_walls: np.ndarray = field(default_factory=lambda: np.zeros((WALL_GRID, WALL_GRID), dtype=bool))
    current_player: int = 0
    turn: int = 0  # number of actions played so far
    _edges: Optional[tuple] = field(default=None, repr=False, compare=False)

    @classmethod
    def initial(cls, first_player: int = 0) -> "GameState":
        return cls(current_player=first_player)

    def copy(self) -> "GameState":
        return GameState(self.pawns, self.walls_left, self.h_walls.copy(), self.v_walls.copy(),
                         self.current_player, self.turn)

    @property
    def winner(self) -> Optional[int]:
        """0 or 1 if a pawn has reached its goal row, else None."""
        for p in (0, 1):
            if self.pawns[p][0] == GOAL_ROWS[p]:
                return p
        return None

    @property
    def is_terminal(self) -> bool:
        return self.winner is not None

    @property
    def num_walls_placed(self) -> int:
        return int(self.h_walls.sum() + self.v_walls.sum())

    def edges(self):
        """Blocked-edge tables ``(south, east)`` as nested lists (cached).

        ``south[r][c]`` is True if (r, c) -> (r+1, c) is blocked (8 x 9),
        ``east[r][c]``  is True if (r, c) -> (r, c+1) is blocked (9 x 8).
        """
        if self._edges is None:
            south = np.zeros((WALL_GRID, BOARD_SIZE), dtype=bool)
            south[:, :-1] |= self.h_walls
            south[:, 1:] |= self.h_walls
            east = np.zeros((BOARD_SIZE, WALL_GRID), dtype=bool)
            east[:-1, :] |= self.v_walls
            east[1:, :] |= self.v_walls
            self._edges = (south.tolist(), east.tolist())
        return self._edges

    def __str__(self) -> str:
        return render_ascii(self)


# --------------------------------------------------------------------------- #
#  Low level helpers (operate on edge tables, so hypothetical walls are cheap)  #
# --------------------------------------------------------------------------- #
def _open(south, east, r: int, c: int, dr: int, dc: int) -> bool:
    """True if moving one step from (r, c) by (dr, dc) stays on the board and crosses no wall."""
    if dr == -1:
        return r > 0 and not south[r - 1][c]
    if dr == 1:
        return r < WALL_GRID and not south[r][c]
    if dc == -1:
        return c > 0 and not east[r][c - 1]
    return c < WALL_GRID and not east[r][c]


def _neighbors(south, east, r: int, c: int):
    if r > 0 and not south[r - 1][c]:
        yield r - 1, c
    if r < WALL_GRID and not south[r][c]:
        yield r + 1, c
    if c > 0 and not east[r][c - 1]:
        yield r, c - 1
    if c < WALL_GRID and not east[r][c]:
        yield r, c + 1


def _has_path(south, east, start: Cell, goal_row: int) -> bool:
    if start[0] == goal_row:
        return True
    seen = [[False] * BOARD_SIZE for _ in range(BOARD_SIZE)]
    seen[start[0]][start[1]] = True
    stack = [start]
    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors(south, east, r, c):
            if not seen[nr][nc]:
                if nr == goal_row:
                    return True
                seen[nr][nc] = True
                stack.append((nr, nc))
    return False


def _shortest_path(south, east, start: Cell, goal_row: int) -> Optional[List[Cell]]:
    if start[0] == goal_row:
        return [start]
    parent = {start: None}
    queue = deque([start])
    while queue:
        cell = queue.popleft()
        for nxt in _neighbors(south, east, *cell):
            if nxt in parent:
                continue
            parent[nxt] = cell
            if nxt[0] == goal_row:
                path = [nxt]
                while parent[path[-1]] is not None:
                    path.append(parent[path[-1]])
                return path[::-1]
            queue.append(nxt)
    return None


# --------------------------------------------------------------------------- #
#  Public rule functions                                                        #
# --------------------------------------------------------------------------- #
def distance_map(state: GameState, player: int) -> np.ndarray:
    """(9, 9) int array: number of steps from every cell to ``player``'s goal row,
    ignoring pawns. Unreachable cells hold ``UNREACHABLE``."""
    south, east = state.edges()
    goal = GOAL_ROWS[player]
    dist = [[UNREACHABLE] * BOARD_SIZE for _ in range(BOARD_SIZE)]
    queue = deque()
    for c in range(BOARD_SIZE):
        dist[goal][c] = 0
        queue.append((goal, c))
    while queue:
        r, c = queue.popleft()
        d = dist[r][c] + 1
        for nr, nc in _neighbors(south, east, r, c):
            if dist[nr][nc] > d:
                dist[nr][nc] = d
                queue.append((nr, nc))
    return np.array(dist, dtype=np.int32)


def path_length(state: GameState, player: int) -> int:
    """Shortest number of steps for ``player`` to reach its goal row (pawns ignored)."""
    r, c = state.pawns[player]
    return int(distance_map(state, player)[r, c])


def shortest_path(state: GameState, player: int) -> Optional[List[Cell]]:
    """One shortest path (list of cells, starting at the pawn) to the goal row."""
    south, east = state.edges()
    return _shortest_path(south, east, state.pawns[player], GOAL_ROWS[player])


def legal_pawn_moves(state: GameState, player: Optional[int] = None) -> List[Cell]:
    """Destination cells the pawn of ``player`` (default: current player) can move to."""
    p = state.current_player if player is None else player
    south, east = state.edges()
    r, c = state.pawns[p]
    opp = state.pawns[1 - p]
    moves = []
    for dr, dc in DIRECTIONS:
        if not _open(south, east, r, c, dr, dc):
            continue
        n = (r + dr, c + dc)
        if n != opp:
            moves.append(n)
            continue
        # Opponent is adjacent: jump straight over it if nothing is behind it...
        if _open(south, east, n[0], n[1], dr, dc):
            moves.append((n[0] + dr, n[1] + dc))
            continue
        # ...otherwise (wall or board edge behind it) land on either side of it (L-shaped move)
        for dr2, dc2 in DIRECTIONS:
            if (dr2, dc2) in ((dr, dc), (-dr, -dc)):
                continue
            if _open(south, east, n[0], n[1], dr2, dc2):
                moves.append((n[0] + dr2, n[1] + dc2))
    return moves


def wall_fits(state: GameState) -> Tuple[np.ndarray, np.ndarray]:
    """(h_fit, v_fit) 8x8 bool arrays: walls that do not overlap or cross existing walls
    (no path check)."""
    h, v = state.h_walls, state.v_walls
    h_fit = ~h & ~v
    h_fit[:, 1:] &= ~h[:, :-1]
    h_fit[:, :-1] &= ~h[:, 1:]
    v_fit = ~v & ~h
    v_fit[1:, :] &= ~v[:-1, :]
    v_fit[:-1, :] &= ~v[1:, :]
    return h_fit, v_fit


def _wall_edges(i: int, j: int, horizontal: bool):
    """The two edge-table entries a wall blocks: list of (table, row, col), table 0=south 1=east."""
    if horizontal:
        return ((0, i, j), (0, i, j + 1))
    return ((1, i, j), (1, i + 1, j))


def walls_on_path(path: Sequence[Cell]) -> Tuple[np.ndarray, np.ndarray]:
    """(h, v) 8x8 bool arrays of wall positions that would block at least one step of ``path``.
    Walls outside these arrays cannot change the length of that path."""
    on_path_h = np.zeros((WALL_GRID, WALL_GRID), dtype=bool)
    on_path_v = np.zeros((WALL_GRID, WALL_GRID), dtype=bool)
    for (r1, c1), (r2, c2) in zip(path, path[1:]):
        if r1 != r2:  # vertical step crosses south[min(r)][c] -> horizontal walls (r, c-1), (r, c)
            r = min(r1, r2)
            on_path_h[r, max(c1 - 1, 0):min(c1, WALL_GRID - 1) + 1] = True
        else:  # horizontal step crosses east[r][min(c)] -> vertical walls (r-1, c), (r, c)
            c = min(c1, c2)
            on_path_v[max(r1 - 1, 0):min(r1, WALL_GRID - 1) + 1, c] = True
    return on_path_h, on_path_v


def legal_walls(state: GameState, player: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """(h_legal, v_legal) 8x8 bool arrays of walls ``player`` (default: current) may place."""
    p = state.current_player if player is None else player
    h_legal = np.zeros((WALL_GRID, WALL_GRID), dtype=bool)
    v_legal = np.zeros((WALL_GRID, WALL_GRID), dtype=bool)
    if state.walls_left[p] <= 0:
        return h_legal, v_legal

    h_fit, v_fit = wall_fits(state)
    south, east = state.edges()

    # Pruning 1: a wall can only disconnect a player if it cuts that player's current shortest path.
    on_path_h = np.zeros_like(h_fit)
    on_path_v = np.zeros_like(v_fit)
    for q in (0, 1):
        ph, pv = walls_on_path(_shortest_path(south, east, state.pawns[q], GOAL_ROWS[q]))
        on_path_h |= ph
        on_path_v |= pv

    # Pruning 2: a wall can only close a loop if it touches the border/other walls at >= 2 points.
    touch = np.zeros((BOARD_SIZE + 1, BOARD_SIZE + 1), dtype=bool)
    touch[0, :] = touch[-1, :] = touch[:, 0] = touch[:, -1] = True
    for i, j in zip(*np.nonzero(state.h_walls)):
        touch[i + 1, j:j + 3] = True
    for i, j in zip(*np.nonzero(state.v_walls)):
        touch[i:i + 3, j + 1] = True
    t = touch.astype(np.int8)
    h_touch = t[1:9, 0:8] + t[1:9, 1:9] + t[1:9, 2:10]
    v_touch = t[0:8, 1:9] + t[1:9, 1:9] + t[2:10, 1:9]

    h_check = h_fit & on_path_h & (h_touch >= 2)
    v_check = v_fit & on_path_v & (v_touch >= 2)
    h_legal[:] = h_fit & ~h_check
    v_legal[:] = v_fit & ~v_check

    if h_check.any() or v_check.any():
        tables = ([row[:] for row in south], [row[:] for row in east])
        for horizontal, check, legal in ((True, h_check, h_legal), (False, v_check, v_legal)):
            for i, j in zip(*np.nonzero(check)):
                cells = _wall_edges(i, j, horizontal)
                for k, a, b in cells:
                    tables[k][a][b] = True
                legal[i, j] = all(_has_path(tables[0], tables[1], state.pawns[q], GOAL_ROWS[q]) for q in (0, 1))
                for k, a, b in cells:
                    tables[k][a][b] = False
    return h_legal, v_legal


def legal_actions(state: GameState) -> List[Action]:
    """Every legal action for the current player (pawn moves first)."""
    if state.is_terminal:
        return []
    actions: List[Action] = [PawnMove(r, c) for r, c in legal_pawn_moves(state)]
    h, v = legal_walls(state)
    actions += [WallPlacement(int(i), int(j), True) for i, j in zip(*np.nonzero(h))]
    actions += [WallPlacement(int(i), int(j), False) for i, j in zip(*np.nonzero(v))]
    return actions


def is_legal(state: GameState, action: Action) -> bool:
    if state.is_terminal:
        return False
    if isinstance(action, PawnMove):
        return (action.row, action.col) in legal_pawn_moves(state)
    if not (0 <= action.row < WALL_GRID and 0 <= action.col < WALL_GRID):
        return False
    h, v = legal_walls(state)
    return bool((h if action.horizontal else v)[action.row, action.col])


def apply_action(state: GameState, action: Action, validate: bool = True) -> GameState:
    """Return the state after the current player plays ``action``.

    The turn passes to the other player unless the move wins the game (like the engine).
    """
    if validate and not is_legal(state, action):
        raise ValueError('Illegal action %r for player %i' % (action, state.current_player))
    p = state.current_player
    pawns, walls_left = list(state.pawns), list(state.walls_left)
    h, v = state.h_walls, state.v_walls
    if isinstance(action, PawnMove):
        pawns[p] = (action.row, action.col)
    else:
        if action.horizontal:
            h = h.copy()
            h[action.row, action.col] = True
        else:
            v = v.copy()
            v[action.row, action.col] = True
        walls_left[p] -= 1
    new = GameState(tuple(pawns), tuple(walls_left), h, v, p, state.turn + 1)
    if new.winner is None:
        new.current_player = 1 - p
    return new


def render_ascii(state: GameState) -> str:
    """Text picture of the board. 0/1 = pawns, | and --- = walls, row 0 at the top."""
    south, east = state.edges()
    lines = ['    ' + '   '.join(str(c) for c in range(BOARD_SIZE))]
    for r in range(BOARD_SIZE):
        row = '%i  ' % r
        for c in range(BOARD_SIZE):
            ch = '.'
            for p in (0, 1):
                if state.pawns[p] == (r, c):
                    ch = str(p)
            row += ' %s ' % ch
            if c < WALL_GRID:
                row += '|' if east[r][c] else ' '
        lines.append(row)
        if r < WALL_GRID:
            lines.append('   ' + ' '.join('---' if south[r][c] else '   ' for c in range(BOARD_SIZE)))
    lines.append('walls left: P0=%i  P1=%i   to move: P%i   turn %i'
                 % (state.walls_left[0], state.walls_left[1], state.current_player, state.turn))
    return '\n'.join(lines)
