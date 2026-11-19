import os
from pygame import Color

from .components.coord import Coord

__doc__ = """ Centralizes all global configuration flags """

# Debug FLAG
__DEBUG__ = False

# Frame rate
FRAMERATE = 25

# Config Options
GAME_TITLE = 'Quoridor'
DEFAULT_NUM_PLAYERS = 2

# ---------------------------------------------------------------- Layout
WINDOW_SIZE = (880, 600)
BOARD_MARGIN = 12          # space between window edge and the board panel
CELL_WIDTH = 48
CELL_HEIGHT = 48
CELL_PAD = 12              # gap between cells (walls are drawn in these gaps)
CELL_RADIUS = 8            # rounded cell corners
CELL_BORDER_SIZE = 2
LABEL_SPACE = 24           # room for the a-i / 1-9 coordinate labels
BOARD_RADIUS = 20          # rounded board panel corners
WALL_INSET = 2             # walls are drawn slightly thinner/shorter than the gap
SIDE_PANEL_GAP = 16        # space between the board and the player cards

# Default Number of rows and cols
DEF_ROWS = 9
DEF_COLS = 9

# Number of Walls per player
NUM_WALLS = 10

# ---------------------------------------------------------------- Colors (dark theme)
BG_COLOR = Color(12, 12, 12)                 # window background
PANEL_COLOR = Color(26, 26, 26)              # board panel and player cards
CELL_COLOR = Color(42, 42, 42)               # normal cell
GOAL_TINTS = (Color(58, 42, 42),             # goal row of red  (top):    dark red tint
              Color(37, 45, 60))             # goal row of blue (bottom): dark blue tint
FOCUS_COLOR = Color(205, 205, 205)           # outline of cells you can move to
CELL_HOVER_COLOR = Color(60, 60, 62)         # legal cell under the mouse
LABEL_COLOR = Color(120, 125, 135)           # a-i / 1-9 labels
TEXT_COLOR = Color(225, 225, 225)
MUTED_TEXT_COLOR = Color(140, 140, 145)
DEBUG_TEXT_COLOR = Color(110, 110, 115)      # A.I. distance numbers in debug mode

PAWN_COLORS = (Color(239, 68, 68),           # red  (player 1, starts at the bottom)
               Color(59, 130, 246))          # blue (player 2, starts at the top)
WALL_COLORS = (Color(244, 163, 163),         # pastel red:  walls placed by red
               Color(158, 197, 250))         # pastel blue: walls placed by blue
WALL_PREVIEW_COLORS = (Color(120, 86, 86),   # hover preview before placing a wall
                       Color(80, 100, 130))
WALL_USED_COLOR = Color(55, 55, 58)          # used-up walls in the player cards
PLAYER_NAMES = ('Red', 'Blue')

# Kept for the engine's defaults
WALL_COLOR = Color(160, 160, 160)
FONT_COLOR = TEXT_COLOR
FONT_SIZE = 16
BOARD_BG_COLOR = PANEL_COLOR
BOARD_BRD_COLOR = PANEL_COLOR
BOARD_BRD_SIZE = 0
CELL_BORDER_COLOR = CELL_COLOR
CELL_VALID_COLOR = CELL_COLOR
PAWN_A_COL, PAWN_B_COL = PAWN_COLORS
PAWN_BORDER_COL = None


class DIR:
    """ Directions
    """
    N = 0
    S = 1
    E = 2
    W = 3


DIRS = {DIR.N, DIR.S, DIR.E, DIR.W}  # Available directions
OPPOSITE_DIRS = [DIR.S, DIR.N, DIR.W, DIR.E]  # Reverse direction

# Delta to add to position to move into that direction
DIRS_DELTA = [Coord(-1, 0), Coord(+1, 0), Coord(0, -1), Coord(0, +1)]

# Network port
NETWORK_ENABLED = False  # Set to true to enable network playing
PORT = 8001  # This client port
BASE_PORT = 8000
SERVER_ADDR = 'localhost'
SERVER_URL = 'http://{}:{}'.format(SERVER_ADDR, PORT)

# Default AI playing level
LEVEL = 0

# Infinite
INF = 99

# Highest A.I. level selectable from the menu (search cost grows ~100x per level)
MAX_LEVEL = 4

# Paths are relative to the engine folder, so the game works from any working directory
ENGINE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOUND_MOVE = os.path.join(ENGINE_DIR, 'sound', 'move.wav')  # wooden click, like a chess move
SOUND_VOLUME = 0.8  # 0.0 - 1.0

# Cache
CACHE_ENABLED = False
CACHE_DIR = os.path.join(ENGINE_DIR, '__cache')
# v2: caches written before the board-state fix (ai.memo) contain wrong entries, so they are not reused
CACHE_AI_FNAME = os.path.join(CACHE_DIR, 'ai_v2.memo')
CACHE_DIST_FNAME = os.path.join(CACHE_DIR, 'dist.memo')
