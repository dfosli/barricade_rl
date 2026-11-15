"""Workspace entry point for the Quoridor game.

Opens the game window. Level, debug mode and cache are chosen in the menu. Each seat can be
Human, Built-in AI, Heuristic bot, or any saved model: every checkpoints/<run>.pt appears as
"RL: <run>" (newest first). Train models with train.py.
"""
from models import list_checkpoints, load_agent
from quoridor_rl import HeuristicAgent
from quoridor_rl.engine_bridge import launch_gui


def menu_agents() -> dict:
    """Player types offered in the menu besides Human / Built-in AI: {label: factory()}."""
    agents = {}
    for path in list_checkpoints():
        agents['RL: %s' % path.stem] = lambda p=path: load_agent(p)
    agents['Heuristic bot'] = HeuristicAgent
    return agents


if __name__ == '__main__':
    raise SystemExit(launch_gui(menu_agents()))
