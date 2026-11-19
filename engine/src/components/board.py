from typing import Set, List, Union
import pygame

from ..logger import log
from .. import config as cfg

from ..ai.action import ActionMovePawn, ActionPlaceWall
from ..ai.ai import AI

from .drawable import Drawable
from .pawn import Pawn
from .cell import Cell
from .wall import Wall
from .coord import Coord

from ..config import DIR


class Board(Drawable):
    """ Quoridor board.
    This object contains te state of the game.
    """

    def __init__(self,
                 screen: pygame.Surface,
                 rows=cfg.DEF_ROWS,
                 cols=cfg.DEF_COLS,
                 cell_padding=cfg.CELL_PAD,
                 color=cfg.BOARD_BG_COLOR,
                 border_color=cfg.BOARD_BRD_COLOR,
                 border_size=cfg.BOARD_BRD_SIZE,
                 ai_players=(1,),
                 ai_level=None,
                 headless=False):
        """
        ai_players: indices of the pawns moved by the built-in A.I. (0 = red/bottom, 1 = blue/top).
        ai_level:   search depth of the built-in A.I. (defaults to cfg.LEVEL).
        headless:   if True nothing is ever drawn (used for fast simulation / RL training).
        """
        Drawable.__init__(self, screen=screen, color=color, border_color=border_color, border_size=border_size)
        self.headless = headless
        self.player_labels = None  # e.g. ['Human', 'Built-in AI'], shown on the player cards
        self.highlighted = set()   # cells outlined as legal moves for a human player
        self.aborted = False  # Set by abort() when the game is left before it ends
        self._state = None
        self.rows: int = rows
        self.cols: int = cols
        self.cell_pad = cell_padding
        self.mouse_wall = None  # Wall painted on mouse move
        self.player: int = 0  # Current player 0 or 1
        self.board: List[List[Cell]] = []
        self.computing = False  # True if a non-human player is moving

        for i in range(rows):
            self.board.append([])
            for j in range(cols):
                self.board[-1].append(Cell(screen, self, coord=Coord(i, j)))

        self.pawns: List[Pawn] = []
        self.pawns += [Pawn(screen=screen,
                            board=self,
                            color=cfg.PAWN_A_COL,
                            border_color=cfg.PAWN_BORDER_COL,
                            coord=Coord(rows - 1, cols >> 1),  # Centered
                            # URL = SERVER_URL + ':%i' % (BASE_PORT + PAWNS)
                            )]
        self.pawns += [Pawn(screen=screen,
                            board=self,
                            color=cfg.PAWN_B_COL,
                            border_color=cfg.PAWN_BORDER_COL,
                            coord=Coord(0, col=cols >> 1)  # Centered
                            )]

        self.regenerate_board(cfg.CELL_COLOR, cfg.CELL_BORDER_COLOR)
        self.num_players = cfg.DEFAULT_NUM_PLAYERS
        self.walls: Set[Wall] = set()  # Walls placed on board
        self._state = None  # The state cached while creating the pawns is incomplete
        self.draw_players_info()
        level = cfg.LEVEL if ai_level is None else ai_level
        self._AI = [AI(self.pawns[i], level=level) for i in ai_players]

    def abort(self) -> None:
        """ Stops this game: an A.I. thread still thinking finishes its search silently
        (nothing is drawn anymore) and makes no further moves.
        """
        self.aborted = True
        self.headless = True

    @property
    def player(self) -> int:
        """ Index of the player to move """
        return self._player

    @player.setter
    def player(self, value: int) -> None:
        self._player = value
        self._state = None  # The serialized state starts with the current player

    def regenerate_board(self, c_color, cb_color, c_width=cfg.CELL_WIDTH, c_height=cfg.CELL_HEIGHT):
        """ Regenerate board colors and get_cell positions.
        Must be called on initialization or whenever a screen attribute
        changes (eg. color, board size, etc)
        """
        y = cfg.BOARD_MARGIN + self.cell_pad
        for i in range(self.rows):
            x = cfg.BOARD_MARGIN + self.cell_pad
            for j in range(self.cols):
                cell = self.board[i][j]
                cell.x, cell.y = x, y
                cell.color = c_color
                cell.border_color = cb_color
                cell.height = c_height
                cell.width = c_width
                cell.pawn = None
                x += c_width + self.cell_pad
            y += c_height + self.cell_pad

        for pawn in self.pawns:
            pawn.cell = self.get_cell(pawn.coord)

    def draw(self):
        """ Draws a squared n x n board, defaults
        to the standard 9 x 9
        """
        if self.headless:
            return

        human_turn = not self.finished and not self.computing and not self.current_player.AI
        self.highlighted = set(self.current_player.valid_moves) if human_turn else set()

        pygame.draw.rect(self.screen, cfg.BG_COLOR, self.panel_rect.inflate(4, 4))
        pygame.draw.rect(self.screen, cfg.PANEL_COLOR, self.panel_rect, border_radius=cfg.BOARD_RADIUS)

        for row in self:
            for cell in row:
                cell.draw()

        # Coordinates like a chess board: columns a-i, rows 1-9 counted from red's side (bottom)
        label_y = self.board[-1][0].rect.bottom + (self.cell_pad + cfg.LABEL_SPACE) // 2
        label_x = self.board[0][-1].rect.right + (self.cell_pad + cfg.LABEL_SPACE) // 2
        for j in range(self.cols):
            self.msg(self.board[0][j].rect.centerx, label_y, 'abcdefghijklmnopqrstuvwxyz'[j],
                     cfg.LABEL_COLOR, 22, anchor='center')
        for i in range(self.rows):
            self.msg(label_x, self.board[i][0].rect.centery, str(self.rows - i), cfg.LABEL_COLOR, 22,
                     anchor='center')

        if cfg.__DEBUG__:
            for p in self.pawns:
                if p.AI:
                    p.distances.draw()
                    break

        for wall in self.walls:
            wall.draw()

    @property
    def panel_rect(self) -> pygame.Rect:
        """ The rounded board panel: cells + gaps + room for the coordinate labels """
        w = self.cols * (cfg.CELL_WIDTH + self.cell_pad) + self.cell_pad + cfg.LABEL_SPACE
        h = self.rows * (cfg.CELL_HEIGHT + self.cell_pad) + self.cell_pad + cfg.LABEL_SPACE
        return pygame.Rect(cfg.BOARD_MARGIN, cfg.BOARD_MARGIN, w, h)

    def get_cell(self, coord: Coord) -> Cell:
        """ Returns board get_cell at the given the coord
        """
        return self.board[coord.row][coord.col]

    def set_cell(self, coord: Coord, value: Cell):
        """ Updates a cell in the board with the new Cell
        instance at the given coord
        """
        self.board[coord.row][coord.col] = value

    def __getitem__(self, i: int) -> List[Cell]:
        return self.board[i]

    def in_range(self, coord: Coord) -> bool:
        """ Returns whether te given coordinate are within the board or not
        """
        return 0 <= coord.col < self.cols and 0 <= coord.row < self.rows

    def putWall(self, wall: Wall) -> None:
        """ Puts the given wall on the board.
        The cells are updated accordingly
        """
        if wall in self.walls:
            return  # If already put, nothing to do

        self.walls.add(wall)
        i, j = wall.coord

        if wall.horiz:
            self.board[i][j].set_path(DIR.S, False)
            self.board[i][j + 1].set_path(DIR.S, False)
        else:
            self.board[i][j].set_path(DIR.W, False)
            self.board[i + 1][j].set_path(DIR.W, False)

        self._state = None

    def removeWall(self, wall: Wall) -> None:
        """ Removes a wall from the board.
        The cells are updated accordingly
        """
        if wall not in self.walls:
            return  # Already removed, nothing to do

        self.walls.remove(wall)
        i, j = wall.coord

        if wall.horiz:
            self.board[i][j].set_path(DIR.S, True)
            self.board[i][j + 1].set_path(DIR.S, True)
        else:
            self.board[i][j].set_path(DIR.W, True)
            self.board[i + 1][j].set_path(DIR.W, True)

        self._state = None

    def onMouseClick(self, x, y):
        """ Dispatch mouse click Event
        """
        cell = self.which_cell(x, y)
        if cell is not None:
            pawn = self.current_player
            if not pawn.can_move(cell.coord):
                return

            self.do_action(ActionMovePawn(pawn.coord, cell.coord))
            cell.set_focus(False)

            if not self.finished:
                self.next_player()
            self.draw()
            self.draw_players_info()
            return

        wall = self.wall(x, y)
        if not wall:
            return

        if self.can_put_wall(wall):
            self.do_action(ActionPlaceWall(wall))
            self.mouse_wall = None
            self.next_player()
            self.draw()
            self.draw_players_info()

    def onMouseMotion(self, x, y):
        """ Get mouse motion event and acts accordingly
        """
        if not self.rect.collidepoint(x, y):
            return

        for row in self.board:
            for cell in row:
                cell.onMouseMotion(x, y)

        if self.which_cell(x, y):
            if self.mouse_wall:
                self.mouse_wall = None
                self.draw()

            return  # The focus was on a get_cell, we're done

        if not self.current_player.walls:
            return  # The current player has run out of walls. We're done

        wall = self.wall(x, y)
        if not wall:
            return

        if self.can_put_wall(wall):
            self.mouse_wall = wall
            self.draw()
            wall.draw()

    def can_put_wall(self, wall) -> bool:
        """ Returns whether the given wall can be put
        on the board.
        """
        if not self.current_player.walls:
            return False

        # Check if any wall has already got that place...
        for w in self.walls:
            if wall.collides(w):
                return False

        result = True
        self.putWall(wall)

        for pawn in self.pawns:
            if not pawn.can_reach_goal():
                result = False
                break

        self.removeWall(wall)
        return result

    def wall(self, x, y) -> Union[Wall, None]:
        """ Factory which returns which wall is below mouse cursor at x, y coords.
        Returns None if no wall matches x, y coords
        """
        if not self.rect.collidepoint(x, y):
            return None

        # Wall: Guess which top-left get_cell is it
        j = (x - self.x) // (self.board[0][0].width + self.cell_pad)
        i = (y - self.y) // (self.board[0][0].height + self.cell_pad)
        cell = self.board[i][j]

        # Wall: Guess if it is horizontal or vertical
        horiz = x < (cell.x + cell.width)
        if horiz:
            if j > 7:
                j = 7
        else:
            if i > 7:
                i = 7

        if i > 7 or j > 7:
            return None

        return self.new_wall(Coord(i, j), horiz, cfg.WALL_PREVIEW_COLORS[self.player])

    def new_wall(self, coord: Coord, horiz: bool, color: pygame.Color = None) -> Wall:
        """ Wall factory. Creates a new wall
        """
        if color is None:
            color = self.board[0][0].wall_color
        return Wall(self.screen, self, color, coord, horiz)

    @property
    def x(self):
        """ Absolute left coordinate
        """
        return self.board[0][0].x

    @property
    def y(self):
        """ Absolute left coordinate
        """
        return self.board[0][0].y

    @property
    def width(self):
        return (self.cell_pad + self.board[0][0].width) * self.cols

    @property
    def height(self):
        return (self.cell_pad + self.board[0][0].height) * self.rows

    @property
    def rect(self):
        return pygame.Rect(self.x, self.y, self.width, self.height)

    def next_player(self):
        """ Switches to next player
        """
        self.player = (self.player + 1) % self.num_players
        self.update_pawns_distances()

    def update_pawns_distances(self):
        for pawn in self.pawns:
            pawn.distances.update()

    def which_cell(self, x, y):
        """ Returns an instance of the get_cell for which (x, y) screen coord
        matches. Otherwise, returns None if no get_cell is at (x, y) screen
        coords.
        """
        for row in self.board:
            for cell in row:
                if cell.rect.collidepoint(x, y):
                    return cell

        return None

    @property
    def current_player(self):
        """ Returns current player's pawn
        """
        return self.pawns[self.player]

    # ------------------------------------------------------------------ side panel
    CARD_HEIGHT = 128

    def _card_rect(self, player_num: int) -> pygame.Rect:
        x = self.panel_rect.right + cfg.SIDE_PANEL_GAP
        w = self.screen.get_width() - x - cfg.BOARD_MARGIN
        y = cfg.BOARD_MARGIN + player_num * (self.CARD_HEIGHT + cfg.BOARD_MARGIN)
        return pygame.Rect(x, y, w, self.CARD_HEIGHT)

    def draw_player_info(self, player_num):
        """ Draws the card of one player: color, who controls it, walls left, status """
        if self.headless:
            return

        pawn = self.pawns[player_num]
        card = self._card_rect(player_num)
        color = cfg.PAWN_COLORS[player_num]
        winner = self.finished and pawn.coord in pawn.goals
        active = not self.finished and self.player == player_num
        thinking = active and self.computing and pawn.percent is not None

        pygame.draw.rect(self.screen, cfg.BG_COLOR, card.inflate(6, 6))
        pygame.draw.rect(self.screen, cfg.PANEL_COLOR, card, border_radius=14)
        if winner:
            pygame.draw.rect(self.screen, color, card, width=3, border_radius=14)
        elif active:
            pygame.draw.rect(self.screen, cfg.FOCUS_COLOR, card, width=2, border_radius=14)

        # Header: disc, name, controller ("Human", "Built-in AI", "RL: mlp", ...)
        pygame.draw.circle(self.screen, color, (card.x + 32, card.y + 36), 14)
        self.msg(card.x + 58, card.y + 18, cfg.PLAYER_NAMES[player_num], cfg.TEXT_COLOR, 32)
        if self.player_labels:
            self.msg(card.x + 58, card.y + 44, self.player_labels[player_num], cfg.MUTED_TEXT_COLOR, 20)

        if winner:
            status, status_color = 'wins!', color
        elif thinking:
            status, status_color = 'thinking %2.0f%%' % (100 * pawn.percent), cfg.MUTED_TEXT_COLOR
        elif active:
            status, status_color = 'to move', cfg.TEXT_COLOR
        else:
            status, status_color = '', cfg.TEXT_COLOR
        if status:
            self.msg(card.right - 16, card.y + 22, status, status_color, 22, anchor='topright')

        # Walls left: one small bar per wall, in the player's wall color
        self.msg(card.x + 16, card.y + 88, 'Walls', cfg.MUTED_TEXT_COLOR, 20, anchor='midleft')
        bar_x = card.x + 70
        for k in range(cfg.NUM_WALLS):
            c = cfg.WALL_COLORS[player_num] if k < pawn.walls else cfg.WALL_USED_COLOR
            pygame.draw.rect(self.screen, c, (bar_x + k * 14, card.y + 77, 7, 22), border_radius=3)
        self.msg(card.right - 16, card.y + 88, str(pawn.walls), cfg.TEXT_COLOR, 26, anchor='midright')

        # Thin progress bar while the A.I. is thinking
        if thinking:
            bar = pygame.Rect(card.x + 16, card.bottom - 14, card.width - 32, 4)
            pygame.draw.rect(self.screen, cfg.WALL_USED_COLOR, bar, border_radius=2)
            bar.width = int(bar.width * pawn.percent)
            pygame.draw.rect(self.screen, color, bar, border_radius=2)

        self.draw_footer()

    def draw_footer(self):
        """ Text under the player cards: help while playing, the result when finished """
        if self.headless:
            return
        first = self._card_rect(0)
        top = self._card_rect(1).bottom + cfg.BOARD_MARGIN
        area = pygame.Rect(first.x - 3, top, first.width + 6, self.screen.get_height() - top)
        pygame.draw.rect(self.screen, cfg.BG_COLOR, area)
        x, y = first.x + 4, top + 12
        if self.finished:
            winner = next(i for i, p in enumerate(self.pawns) if p.coord in p.goals)
            self.msg(x, y, '%s wins!' % cfg.PLAYER_NAMES[winner], cfg.PAWN_COLORS[winner], 44)
            self.msg(x, y + 48, 'Press any key or click', cfg.MUTED_TEXT_COLOR, 22)
            self.msg(x, y + 70, 'to return to the menu', cfg.MUTED_TEXT_COLOR, 22)
        else:
            for k, line in enumerate(('Click an outlined square to move.',
                                      'Click between squares to place a wall.',
                                      'ESC: back to the menu')):
                self.msg(x, y + 24 * k, line, cfg.MUTED_TEXT_COLOR, 20)

    _fonts = {}

    def msg(self, x, y, str_, color=cfg.TEXT_COLOR, fsize=cfg.FONT_SIZE, anchor='topleft'):
        """ Draws text; (x, y) is the point given by ``anchor`` ('topleft', 'center', 'midright', ...) """
        font = Board._fonts.get(fsize)
        if font is None:
            font = Board._fonts[fsize] = pygame.font.Font(None, fsize)
        surf = font.render(str_, True, color)
        self.screen.blit(surf, surf.get_rect(**{anchor: (x, y)}))

    def draw_players_info(self):
        """ Calls the above function for every player.
        """
        for i in range(len(self.pawns)):
            self.draw_player_info(i)

    def do_action(self, action: Union[ActionPlaceWall, ActionMovePawn]):
        """ Performs a playing action: move a pawn or place a barrier.
        Transmit the action to the network, to inform other players.
        """
        player_id = self.current_player.id

        if isinstance(action, ActionPlaceWall):
            wdir = 'horizontal' if action.horiz else 'vertical'
            log('Player %i places %s wall at (%i, %i)' % (player_id, wdir, action.coord.col, action.coord.row))
            self.putWall(self.new_wall(action.coord, action.horiz, cfg.WALL_COLORS[self.player]))
            self.current_player.walls -= 1
        else:
            log('Player %i moves to (%i, %i)' % (player_id, action.dest.row, action.dest.col))
            self.current_player.move_to(action.dest)
            self._state = None
        self.play_move_sound()

    _move_sound = None  # loaded once, shared by all boards

    def play_move_sound(self):
        """ Short wooden click on every move or wall, for humans and A.I.s alike (silent when headless) """
        if self.headless:
            return
        try:
            if Board._move_sound is None:
                if not pygame.mixer.get_init():
                    pygame.mixer.init()
                Board._move_sound = pygame.mixer.Sound(cfg.SOUND_MOVE)
                Board._move_sound.set_volume(cfg.SOUND_VOLUME)
            Board._move_sound.play()
        except pygame.error:
            pass  # No audio device / file: play silently

    def computer_move(self):
        """ Performs computer moves for every non-human player
        """
        while self.current_player.AI and not self.finished and not self.aborted:
            self.draw()
            self.draw_players_info()
            action, x = self.current_player.AI.move()
            if self.aborted:
                break
            self.do_action(action)

            if self.finished:
                break

            self.next_player()

        self.computing = False  # before drawing, so a human's legal moves get outlined
        self.draw()
        self.draw_players_info()

    @property
    def finished(self):
        """ Returns whether the match has finished or not.
        """
        return any(pawn.coord in pawn.goals for pawn in self.pawns)

    @property
    def state(self):
        """ Status serialization in a t-uple
        """
        if self._state is not None:
            return self._state

        result = str(self.player)  # current player
        result += ''.join(p.state for p in self.pawns)
        result += ''.join(self.board[i][j].state for j in range(self.cols - 1) for i in range(self.rows - 1))
        self._state = result

        return result
