"""Sanity tests for the RL toolkit. Run:  python rl-program/tests/test_quoridor_rl.py  (or pytest)."""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quoridor_rl import (NUM_ACTIONS, OBS_SIZE, GameState, GreedyPathAgent, QuoridorEnv, RandomAgent,  # noqa: E402
                         action_mask, action_to_index, apply_action, canonical_state, encode_observation,
                         evaluate, index_to_action, legal_actions, legal_pawn_moves, legal_walls,
                         mirror_action, mirror_action_vector, mirror_observation, observation_to_state,
                         play_game)
from quoridor_rl.rules import GOAL_ROWS, _has_path, wall_fits  # noqa: E402


def random_states(n_games=30, seed=0, move_probability=0.5):
    """Yield states from random games (mix of moves and walls)."""
    agent = RandomAgent(seed=seed, move_probability=move_probability)
    for _ in range(n_games):
        state = GameState.initial()
        while not state.is_terminal and state.turn < 120:
            yield state
            mask = action_mask(state)
            state = apply_action(state, index_to_action(state, agent.act(None, mask)), validate=False)


def brute_force_walls(state):
    """Legal walls without any pruning."""
    h_fit, v_fit = wall_fits(state)
    out = []
    for horizontal, fit in ((True, h_fit), (False, v_fit)):
        legal = np.zeros_like(fit)
        for i, j in zip(*np.nonzero(fit)):
            h, v = state.h_walls.copy(), state.v_walls.copy()
            (h if horizontal else v)[i, j] = True
            s = GameState(state.pawns, state.walls_left, h, v)
            south, east = s.edges()
            legal[i, j] = all(_has_path(south, east, s.pawns[q], GOAL_ROWS[q]) for q in (0, 1))
        out.append(legal)
    return out


def test_legal_walls_match_brute_force():
    for state in random_states(15, seed=1):
        if state.walls_left[state.current_player] == 0:
            continue
        h, v = legal_walls(state)
        bh, bv = brute_force_walls(state)
        assert (h == bh).all() and (v == bv).all(), str(state)


def test_index_roundtrip_and_canonical():
    for state in random_states(10, seed=2):
        mask = action_mask(state)
        assert mask.shape == (NUM_ACTIONS,) and mask.any()
        assert len(legal_actions(state)) == mask.sum()
        for a in np.flatnonzero(mask):
            assert action_to_index(state, index_to_action(state, a)) == a
        cs = canonical_state(canonical_state(state, 1), 1)
        assert cs.pawns == state.pawns and (cs.h_walls == state.h_walls).all()


def test_observation_roundtrip_and_mirror():
    for state in random_states(5, seed=3):
        obs = encode_observation(state)
        assert obs.shape == (OBS_SIZE,)
        back = observation_to_state(obs)
        cs = canonical_state(state)
        assert back.pawns == cs.pawns and (back.h_walls == cs.h_walls).all() and (back.v_walls == cs.v_walls).all()
        # Mirroring the observation and the mask must stay consistent
        mask = action_mask(state)
        mobs = mirror_observation(obs)
        mstate = observation_to_state(mobs)
        assert (action_mask(mstate) == mirror_action_vector(mask)).all()
        assert (mirror_action(mirror_action(np.arange(NUM_ACTIONS))) == np.arange(NUM_ACTIONS)).all()


def test_jump_rule():
    """Straight jump when free; sideways (L-shaped) only with a wall / edge behind the opponent."""
    from quoridor_rl import WallPlacement
    s = GameState(pawns=((5, 4), (4, 4)))
    assert set(legal_pawn_moves(s)) == {(6, 4), (5, 3), (5, 5), (3, 4)}
    s = apply_action(s, WallPlacement(3, 4, True))  # wall behind the opponent (between rows 3 and 4)
    s.current_player = 0
    s._edges = None
    assert set(legal_pawn_moves(s)) == {(6, 4), (5, 3), (5, 5), (4, 3), (4, 5)}
    s = GameState(pawns=((1, 4), (0, 4)))  # board edge behind the opponent
    assert set(legal_pawn_moves(s)) == {(2, 4), (1, 3), (1, 5), (0, 3), (0, 5)}


def test_rules_match_engine():
    """Pawn moves must match the engine exactly; wall legality: engine-legal walls must be legal
    for us (the engine's path test is slightly stricter because it treats pawns as obstacles)."""
    from quoridor_rl.engine_bridge import apply_to_board, new_engine_board, state_from_board

    agent = RandomAgent(seed=4, move_probability=0.5)
    checked = 0
    for _ in range(4):
        board = new_engine_board()
        state = GameState.initial()
        while not state.is_terminal and state.turn < 80:
            synced = state_from_board(board)
            assert synced.pawns == state.pawns and (synced.h_walls == state.h_walls).all()
            assert synced.walls_left == state.walls_left and synced.current_player == state.current_player
            engine_moves = {(c.row, c.col) for c in board.current_player.valid_moves}
            assert engine_moves == set(legal_pawn_moves(state)), str(state)
            if state.turn % 5 == 0 and state.walls_left[state.current_player]:
                h, v = legal_walls(state)
                for i in range(8):
                    for j in range(8):
                        for horizontal, ours in ((True, h), (False, v)):
                            from quoridor_rl.engine_bridge import Coord
                            if board.can_put_wall(board.new_wall(Coord(i, j), horizontal)):
                                assert ours[i, j]
                checked += 1
            action = index_to_action(state, agent.act(None, action_mask(state)))
            state = apply_action(state, action, validate=False)
            apply_to_board(board, action)
    assert checked > 0


def test_env_vs_random_and_selfplay():
    env = QuoridorEnv(opponent='random', agent_player='random', seed=0)
    agent = RandomAgent(seed=5)
    for _ in range(5):
        obs, info = env.reset()
        done = False
        while not done:
            assert info['action_mask'].any()
            obs, r, term, trunc, info = env.step(agent.act(obs, info['action_mask']))
            done = term or trunc
        assert r in (1.0, -1.0) or trunc

    env = QuoridorEnv(opponent=None, seed=0)
    obs, info = env.reset()
    done = False
    while not done:
        obs, r, term, trunc, info = env.step(agent.act(obs, info['action_mask']))
        done = term or trunc
    if term:
        assert r == 1.0 and info['winner'] == info['mover']


def test_env_vs_builtin():
    env = QuoridorEnv(opponent='builtin', opponent_level=0, agent_player=1, max_turns=80)
    agent = GreedyPathAgent(seed=0)
    obs, info = env.reset()
    assert env.state.turn == 1  # the AI moved first
    done = False
    while not done:
        obs, r, term, trunc, info = env.step(agent.act(obs, info['action_mask']))
        done = term or trunc
        synced = __import__('quoridor_rl.engine_bridge', fromlist=['x']).state_from_board(env._builtin.board)
        assert synced.pawns == env.state.pawns and (synced.v_walls == env.state.v_walls).all()


def test_heuristic_agent_is_exact_one_ply():
    """HeuristicAgent's wall pruning must not change its choice vs. scoring every action."""
    from quoridor_rl import HeuristicAgent, path_length
    agent = HeuristicAgent(seed=0)
    for state in random_states(4, seed=6):
        mask = action_mask(state)
        a = agent.act(encode_observation(state), mask)
        cs = canonical_state(state)
        act = index_to_action(cs, a)
        if a < 12 and act.row == 0:
            continue  # winning move
        scores = {}
        for b in np.flatnonzero(mask):
            ns = apply_action(cs, index_to_action(cs, b), validate=False)
            scores[b] = path_length(ns, 0) - path_length(ns, 1)
        assert scores[a] == min(scores.values())


def test_evaluate_and_play_game():
    rec = play_game(GreedyPathAgent(), RandomAgent(seed=0))
    assert rec.winner in (0, 1, None)
    res = evaluate(GreedyPathAgent(), opponent='random', n_games=6)
    assert res['wins'] + res['losses'] + res['draws'] == 6


def test_all_registered_models():
    """Every model in models/: builds, outputs (B, 140), acts legally, survives save/load."""
    import tempfile
    import torch
    from models import TorchAgent, available_models, create_model, load_checkpoint, save_checkpoint
    assert {'mlp', 'cnn'} <= set(available_models())
    env = QuoridorEnv(opponent='random', seed=0)
    for name in available_models():
        net = create_model(name)
        x = torch.randn(3, net.obs_size)
        assert tuple(net.action_values(x).shape) == (3, NUM_ACTIONS), name
        env.include_distances = net.include_distances
        obs, info = env.reset()
        agent = TorchAgent(net)
        for _ in range(5):
            a = agent.act(obs, info['action_mask'])
            assert info['action_mask'][a]
            obs, r, term, trunc, info = env.step(a)
            if term or trunc:
                break
        with tempfile.TemporaryDirectory() as d:
            path = save_checkpoint(net, run_name=str(Path(d) / ('%s.pt' % name)), episode=7)
            loaded, extra = load_checkpoint(path)
            assert extra['episode'] == 7 and loaded.config == net.config
            assert torch.allclose(loaded.action_values(x), net.eval().action_values(x))


def benchmark():
    env = QuoridorEnv(opponent='random', seed=0)
    agent = RandomAgent(seed=0, move_probability=0.5)
    obs, info = env.reset()
    steps, t0 = 0, time.time()
    while time.time() - t0 < 3:
        obs, r, term, trunc, info = env.step(agent.act(obs, info['action_mask']))
        steps += 1
        if term or trunc:
            obs, info = env.reset()
    print('env vs random: %.0f steps/s (each step = agent + opponent move)' % (steps / (time.time() - t0)))


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            t = time.time()
            fn()
            print('PASS %-40s %.1fs' % (name, time.time() - t))
    benchmark()
