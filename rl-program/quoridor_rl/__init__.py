"""Quoridor RL toolkit.

Most used names are re-exported here:

    from quoridor_rl import QuoridorEnv, RandomAgent, evaluate, NUM_ACTIONS, OBS_SIZE

The engine bridge (pygame) is NOT imported here, so training without the built-in AI never
touches pygame. Import it explicitly: ``from quoridor_rl.engine_bridge import launch_gui``.
"""
from .agents import Agent, FunctionAgent, GreedyPathAgent, HeuristicAgent, RandomAgent, as_agent
from .encoding import (DIST_SCALE, H_WALL_START, MOVE_OFFSETS, NUM_ACTIONS, NUM_MOVE_ACTIONS, OBS_SIZE,
                       OBS_SIZE_NO_DIST, OBS_SLICES, V_WALL_START, action_mask, action_to_index,
                       canonical_state, describe_action, encode_observation, encode_planes,
                       index_to_action, mirror_action, mirror_action_vector, mirror_observation,
                       obs_size, observation_to_state)
from .env import QuoridorEnv
from .evaluate import GameRecord, evaluate, play_game
from .masking import epsilon_greedy, masked_argmax, masked_fill, masked_softmax, sample_masked
from .rules import (Action, GameState, PawnMove, WallPlacement, apply_action, distance_map, is_legal, walls_on_path,
                    legal_actions, legal_pawn_moves, legal_walls, path_length, render_ascii,
                    shortest_path)
