import os
import pygame
from pygame.locals import *
from pygame import Color
import threading
import time
from dataclasses import dataclass, field, replace
from typing import Callable, Dict, List, Optional, Tuple

from .logger import log, LogLevel
from . import config as cfg
from . import core

from .components.board import Board


# Player types always available in the menu
HUMAN = 'Human'
BUILTIN_AI = 'Built-in AI'
PLAYER_LABELS = ('Red (bottom)', 'Blue (top)')

# A controller factory receives (board, pawn) and returns an object with a .move() -> (Action, value)
# method, exactly like the built-in AI class. Used to plug in e.g. a trained RL agent.
ControllerFactory = Callable[[Board, object], object]


QUIT_APP = 'quit'  # window closed
BACK_TO_MENU = 'menu'  # ESC, or any key / click after the game has finished


def dispatch(events, board: Board) -> Optional[str]:
    """ Handles the in-game events. Returns QUIT_APP, BACK_TO_MENU or None (keep playing) """
    for event in events:
        if event.type == QUIT:
            return QUIT_APP

        if event.type == KEYDOWN:
            if event.key == K_ESCAPE or board.finished:
                return BACK_TO_MENU

        if board.finished and event.type == MOUSEBUTTONDOWN:
            return BACK_TO_MENU

        if board.computing or board.finished:
            continue

        if event.type == MOUSEBUTTONDOWN:
            x, y = pygame.mouse.get_pos()
            board.onMouseClick(x, y)

        if event.type == MOUSEMOTION:
            x, y = pygame.mouse.get_pos()
            board.onMouseMotion(x, y)

    return None


# ----------------------- Welcome / UI  ----------------------- #
@dataclass
class GameSettings:
    """ Everything chosen in the welcome screen """
    players: List[str] = field(default_factory=lambda: [HUMAN, BUILTIN_AI])
    level: int = cfg.LEVEL
    debug: bool = cfg.__DEBUG__
    cache: bool = cfg.CACHE_ENABLED


def draw_rounded_button(surface: pygame.Surface,
                        rect: pygame.Rect,
                        text: str,
                        font: pygame.font.Font,
                        base_color: Color,
                        hover_color: Color,
                        border_color: Color,
                        hovered: bool,
                        text_color: Color = Color(255, 255, 255),
                        radius: int = 10) -> None:
    """Draw a rounded button and centered text."""
    color = hover_color if hovered else base_color
    pygame.draw.rect(surface, color, rect, border_radius=radius)
    pygame.draw.rect(surface, border_color, rect, width=2, border_radius=radius)

    text_surf = font.render(text, True, text_color)
    text_rect = text_surf.get_rect(center=rect.center)
    surface.blit(text_surf, text_rect)


def fit_text(text: str, font: pygame.font.Font, max_width: int) -> str:
    """Shortens '< long name >' to '< long na.. >' until it fits in max_width pixels."""
    if font.size(text)[0] <= max_width:
        return text
    inner = text[2:-2]
    while inner and font.size('< %s.. >' % inner)[0] > max_width:
        inner = inner[:-1]
    return '< %s.. >' % inner


def point_in_rect(point: Tuple[int, int], rect: pygame.Rect) -> bool:
    x, y = point
    return rect.left <= x <= rect.right and rect.top <= y <= rect.bottom


def _menu_background(size: Tuple[int, int], panel: pygame.Rect) -> pygame.Surface:
    """ Dark background with a rounded panel behind the settings, computed once """
    bg = pygame.Surface(size)
    bg.fill(cfg.BG_COLOR)
    pygame.draw.rect(bg, cfg.PANEL_COLOR, panel, border_radius=cfg.BOARD_RADIUS)
    return bg


def show_welcome_screen(screen: pygame.Surface, player_options: List[str],
                        initial: Optional[GameSettings] = None) -> Optional[GameSettings]:
    """
    Display the welcome / settings screen, pre-filled with ``initial`` (e.g. the last game's settings).
    Returns the chosen GameSettings, or None if the user exits.
    """
    settings = GameSettings() if initial is None else replace(initial, players=list(initial.players))
    settings.players = [p if p in player_options else HUMAN for p in settings.players]
    screen_w, screen_h = screen.get_size()
    cx = screen_w // 2

    title_font = pygame.font.Font(None, 72)
    subtitle_font = pygame.font.Font(None, 28)
    label_font = pygame.font.Font(None, 30)
    hint_font = pygame.font.Font(None, 20)
    btn_font = pygame.font.Font(None, 28)

    dark = cfg.TEXT_COLOR              # main text
    soft = cfg.MUTED_TEXT_COLOR        # hints
    border = Color(70, 70, 74)
    white = Color(245, 245, 245)
    wood, wood_hover = cfg.CELL_COLOR, cfg.CELL_HOVER_COLOR       # neutral controls
    on_col, on_hover = cfg.PAWN_COLORS[1], Color(96, 155, 250)    # "On" and Start: blue
    off_col, off_hover = cfg.CELL_COLOR, cfg.CELL_HOVER_COLOR
    exit_col, exit_hover = cfg.PAWN_COLORS[0], Color(245, 110, 110)

    title_surf = title_font.render('Welcome to Quoridor', True, dark).convert_alpha()
    subtitle_surf = subtitle_font.render('Choose your settings and press Start', True, soft).convert_alpha()
    title_rect = title_surf.get_rect(center=(cx, 80))
    subtitle_rect = subtitle_surf.get_rect(center=(cx, 128))

    # --- Layout: one row per setting ---
    label_x = cx - 260
    ctrl_x, ctrl_w, ctrl_h = cx + 10, 240, 40
    row_y = [175 + i * 60 for i in range(5)]
    player_rects = [pygame.Rect(ctrl_x, row_y[i], ctrl_w, ctrl_h) for i in range(2)]
    minus_rect = pygame.Rect(ctrl_x, row_y[2], 48, ctrl_h)
    plus_rect = pygame.Rect(ctrl_x + ctrl_w - 48, row_y[2], 48, ctrl_h)
    level_rect = pygame.Rect(ctrl_x + 48, row_y[2], ctrl_w - 96, ctrl_h)
    debug_rect = pygame.Rect(ctrl_x, row_y[3], ctrl_w, ctrl_h)
    cache_rect = pygame.Rect(ctrl_x, row_y[4], ctrl_w, ctrl_h)

    btn_w, btn_h, gap = 180, 48, 24
    start_rect = pygame.Rect(cx - btn_w - gap // 2, 500, btn_w, btn_h)
    exit_rect = pygame.Rect(cx + gap // 2, 500, btn_w, btn_h)
    background = _menu_background((screen_w, screen_h), pygame.Rect(cx - 300, 158, 600, 314))

    def label(text: str, y: int, hint: str = None):
        surf = label_font.render(text, True, dark)
        screen.blit(surf, surf.get_rect(midleft=(label_x, y + ctrl_h // 2 - (7 if hint else 0))))
        if hint:
            h = hint_font.render(hint, True, soft)
            screen.blit(h, h.get_rect(midleft=(label_x, y + ctrl_h // 2 + 14)))

    def cycle(i: int, step: int):
        options = player_options
        settings.players[i] = options[(options.index(settings.players[i]) + step) % len(options)]

    clock = pygame.time.Clock()
    start_time = time.time()

    while True:
        clock.tick(cfg.FRAMERATE)
        mouse = pygame.mouse.get_pos()
        alpha = min(255, int(255 * (time.time() - start_time) / 0.5))  # title fade-in

        # --- Draw ---
        screen.blit(background, (0, 0))
        title_surf.set_alpha(alpha)
        subtitle_surf.set_alpha(alpha)
        screen.blit(title_surf, title_rect)
        screen.blit(subtitle_surf, subtitle_rect)

        uses_ai = BUILTIN_AI in settings.players
        for i in range(2):
            label(PLAYER_LABELS[i], row_y[i], 'click to change (right-click: back)')
            pygame.draw.circle(screen, cfg.PAWN_COLORS[i], (label_x - 18, row_y[i] + ctrl_h // 2 - 7), 8)
            text = fit_text('< %s >' % settings.players[i], btn_font, ctrl_w - 16)
            draw_rounded_button(screen, player_rects[i], text, btn_font,
                                wood, wood_hover, border, player_rects[i].collidepoint(mouse), white, radius=8)

        label('Built-in AI level', row_y[2],
              '0 = easy, higher = stronger but slower' if uses_ai else '(no built-in AI selected)')
        for rect, txt in ((minus_rect, '-'), (plus_rect, '+')):
            draw_rounded_button(screen, rect, txt, btn_font, wood, wood_hover, border,
                                rect.collidepoint(mouse), white, radius=8)
        lvl = label_font.render(str(settings.level), True, dark if uses_ai else soft)
        screen.blit(lvl, lvl.get_rect(center=level_rect.center))

        label('Debug mode', row_y[3], 'shows A.I. distances and logs')
        label('Persistent A.I. cache', row_y[4], 'remembers A.I. searches between runs')
        for rect, value in ((debug_rect, settings.debug), (cache_rect, settings.cache)):
            base, hover = (on_col, on_hover) if value else (off_col, off_hover)
            draw_rounded_button(screen, rect, 'On' if value else 'Off', btn_font, base, hover, border,
                                rect.collidepoint(mouse), white, radius=8)

        draw_rounded_button(screen, start_rect, 'Start', btn_font, on_col, on_hover, border,
                            start_rect.collidepoint(mouse), white, radius=8)
        draw_rounded_button(screen, exit_rect, 'Exit', btn_font, exit_col, exit_hover, border,
                            exit_rect.collidepoint(mouse), white, radius=8)
        pygame.display.update()

        # --- Events ---
        for event in pygame.event.get():
            if event.type == QUIT:
                return None
            if event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    return None
                if event.key in (K_RETURN, K_KP_ENTER, K_SPACE):
                    return settings
                if event.key in (K_LEFT, K_MINUS, K_KP_MINUS):
                    settings.level = max(0, settings.level - 1)
                if event.key in (K_RIGHT, K_PLUS, K_KP_PLUS, K_EQUALS):
                    settings.level = min(cfg.MAX_LEVEL, settings.level + 1)
            if event.type == MOUSEBUTTONDOWN and event.button in (1, 3):
                pos = event.pos
                step = 1 if event.button == 1 else -1
                for i in range(2):
                    if player_rects[i].collidepoint(pos):
                        cycle(i, step)
                if event.button != 1:
                    continue
                if minus_rect.collidepoint(pos):
                    settings.level = max(0, settings.level - 1)
                elif plus_rect.collidepoint(pos):
                    settings.level = min(cfg.MAX_LEVEL, settings.level + 1)
                elif debug_rect.collidepoint(pos):
                    settings.debug = not settings.debug
                elif cache_rect.collidepoint(pos):
                    settings.cache = not settings.cache
                elif start_rect.collidepoint(pos):
                    return settings
                elif exit_rect.collidepoint(pos):
                    return None


# ----------------------- End welcome helpers ----------------------- #


def play_game(screen: pygame.Surface, settings: GameSettings,
              extra_players: Dict[str, ControllerFactory]) -> str:
    """ Plays one game with the given settings. Returns QUIT_APP or BACK_TO_MENU. """
    core.init()
    cfg.LEVEL = settings.level
    cfg.__DEBUG__ = settings.debug
    cfg.CACHE_ENABLED = settings.cache
    log('Settings: %s' % settings)

    if cfg.CACHE_ENABLED:
        if not os.path.exists(cfg.CACHE_DIR):
            log('Cache directory {} not found. Creating it...'.format(cfg.CACHE_DIR))
            os.makedirs(cfg.CACHE_DIR, exist_ok=True)

        if not os.path.isdir(cfg.CACHE_DIR):
            log('Could not create cache directory {}. Caching disabled'.format(cfg.CACHE_DIR), LogLevel.ERROR)
            cfg.CACHE_ENABLED = False

    screen.fill(cfg.BG_COLOR)
    builtin = [i for i, kind in enumerate(settings.players) if kind == BUILTIN_AI]
    board = core.BOARD = Board(screen, ai_players=builtin, ai_level=settings.level)
    for i, kind in enumerate(settings.players):
        if kind in extra_players:
            board.pawns[i].AI = extra_players[kind](board, board.pawns[i])
            log('Player %i is moved by %s' % (i, kind), LogLevel.INFO)

    board.player_labels = list(settings.players)
    board.draw()
    board.draw_players_info()
    log('Game started')

    clock = pygame.time.Clock()
    thread = None
    result = None
    while result is None:
        clock.tick(cfg.FRAMERATE)
        pygame.display.flip()

        if not board.computing and not board.finished:
            if board.current_player.AI:
                board.computing = True
                thread = threading.Thread(target=board.computer_move, daemon=True)
                thread.start()

        result = dispatch(pygame.event.get(), board)

    # Leaving the game: an A.I. still thinking finishes silently in the background
    board.abort()
    ai_running = thread is not None and thread.is_alive()

    if cfg.CACHE_ENABLED:
        if ai_running:
            log('A.I. still thinking, its cache is not saved this time', LogLevel.WARN)
        else:
            for pawn in board.pawns:
                if pawn.AI is not None and hasattr(pawn.AI, 'flush_cache'):
                    pawn.AI.flush_cache()

    log('Memoized nodes: %i' % core.MEMOIZED_NODES)
    log('Memoized nodes hits: %i' % core.MEMOIZED_NODES_HITS)
    for pawn in board.pawns:
        log('Memoized distances for [%i]: %i' % (pawn.id, pawn.distances.MEMO_COUNT))
        log('Memoized distances hits for [%i]: %i' % (pawn.id, pawn.distances.MEMO_HITS))

    return result


def main(extra_players: Optional[Dict[str, ControllerFactory]] = None) -> int:
    """ Runs the game window: menu -> game -> menu -> ... until the user exits.

    extra_players: optional extra player types for the menu, e.g. {'RL agent': factory}.
                   factory(board, pawn) must return a controller with a .move() method
                   returning (Action, value), like the built-in AI.
    """
    extra_players = dict(extra_players or {})

    log('Quoridor AI game')
    log('Initializing system...')

    pygame.init()
    pygame.display.set_mode(cfg.WINDOW_SIZE)
    pygame.display.set_caption(cfg.GAME_TITLE)
    screen = pygame.display.get_surface()
    log('System initialized OK')

    settings = None
    while True:
        settings = show_welcome_screen(screen, [HUMAN, BUILTIN_AI] + list(extra_players), settings)
        if settings is None:
            log('User exited from the menu.')
            break

        if play_game(screen, settings, extra_players) == QUIT_APP:
            break

    pygame.quit()
    log('Exiting. Bye!')
    return 0


if __name__ == '__main__':
    main()
