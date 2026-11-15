# rl-program – Quoridor RL toolkit

```
rl-program/
  main.py              open the game window (menu: Human / Built-in AI / Heuristic bot / every saved model)
  train.py             training template – plumbing done, add your algorithm at the TODOs
  models/              one file per network, found automatically
    base.py            registry, QuoridorModel base class, TorchAgent, checkpoints
    mlp.py, cnn.py     example models ('mlp', 'cnn')
    _template.py       copy this to add a model
  checkpoints/         <run>.pt files written by train.py (git-ignored)
  quoridor_rl/
    rules.py           fast pygame-free game rules (GameState, legal moves/walls, distances)
    encoding.py        observation vector (294), action space (140), canonical flip, mirroring
    env.py             QuoridorEnv – gym-style reset()/step()
    agents.py          Agent interface + RandomAgent, GreedyPathAgent, HeuristicAgent
    masking.py         masked argmax / softmax / epsilon-greedy (numpy + torch)
    evaluate.py        play_game, evaluate (win rates)
    engine_bridge.py   link to engine/: built-in AI, headless boards, GUI launcher
  tests/test_quoridor_rl.py   python rl-program/tests/test_quoridor_rl.py
```

Run from the repo root: `python rl-program/main.py`, `python rl-program/train.py`.

## Conventions

* **Canonical frame**: observations, masks and action indices are always from the point of view of
  the player to move; for player 1 the board is flipped vertically. One network plays both sides.
* **Actions (140)**: `0-11` pawn moves by `MOVE_OFFSETS` (step / straight jump / diagonal jump),
  `12-75` horizontal walls `12 + i*8 + j`, `76-139` vertical walls `76 + i*8 + j`.
* **Observation (294 floats)**: my pawn one-hot (81), opponent one-hot (81), h-walls (64), v-walls (64),
  walls left /10 (2), shortest-path lengths /20 (2, optional → 292).
* **Agent** = any object with `act(obs, mask) -> int`.
* **Rewards**: +1 win, -1 loss, 0 otherwise (+ optional `distance_reward` shaping). Games longer
  than `max_turns` are truncated (draw).

## Quick start

```python
from quoridor_rl import QuoridorEnv, RandomAgent, evaluate

env = QuoridorEnv(opponent='heuristic', agent_player='random')
obs, info = env.reset()
done = False
while not done:
    action = my_agent.act(obs, info['action_mask'])
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
print(env.render())

print(evaluate(my_agent, opponent='builtin', n_games=10))
```

## Function reference

### Environment – `quoridor_rl.env`
| | |
|---|---|
| `QuoridorEnv(opponent, opponent_level, agent_player, max_turns, include_distances, distance_reward, seed)` | `opponent`: `None` (self-play), `'random'`, `'greedy'`, `'heuristic'`, `'builtin'`, or any Agent / function |
| `env.reset(seed=None) -> obs, info` | new game; opponent moves first if `agent_player == 1` |
| `env.step(action) -> obs, reward, terminated, truncated, info` | `info`: `action_mask`, `current_player`, `agent_player`, `mover`, `winner`, `turn`, `walls_left`, `last_opponent_action` |
| `env.action_mask()`, `env.legal_actions()`, `env.sample_action()` | legal-action helpers |
| `env.render()` | ASCII board |
| `env.state` | the underlying `GameState` |
| `env.observation_size`, `env.num_actions` | 294 (or 292), 140 |

### Encoding – `quoridor_rl.encoding`
| | |
|---|---|
| `encode_observation(state, player=None, include_distances=True)` | flat float32 vector for an MLP |
| `encode_planes(state, player=None)` | (8, 9, 9) planes for a CNN later |
| `observation_to_state(obs)` | inverse of `encode_observation` (canonical GameState) |
| `action_mask(state)` | (140,) bool mask for the player to move |
| `index_to_action(state, idx)` / `action_to_index(state, action)` | canonical index ↔ absolute `PawnMove` / `WallPlacement` |
| `describe_action(idx)` | e.g. `'horizontal wall at (3, 4)'` |
| `canonical_state(state, player=None)` | state as seen by `player` (becomes player 0) |
| `mirror_observation(obs)`, `mirror_action(idx)`, `mirror_action_vector(v)` | left-right mirror – free data augmentation |
| constants | `NUM_ACTIONS`, `NUM_MOVE_ACTIONS`, `H_WALL_START`, `V_WALL_START`, `MOVE_OFFSETS`, `OBS_SIZE`, `OBS_SIZE_NO_DIST`, `OBS_SLICES`, `DIST_SCALE` |

### Rules – `quoridor_rl.rules` (pure Python, no pygame)
| | |
|---|---|
| `GameState.initial()`, `.copy()`, `.winner`, `.is_terminal`, `.current_player`, `.pawns`, `.walls_left`, `.h_walls`, `.v_walls`, `.turn` | game state |
| `apply_action(state, action, validate=True)` | new state after an action (for search / MCTS) |
| `legal_actions(state)`, `is_legal(state, action)` | |
| `legal_pawn_moves(state, player=None)` | destination cells (engine jump rules) |
| `legal_walls(state, player=None)` | (h, v) 8x8 bool arrays |
| `wall_fits(state)` | walls that don't overlap (no path check) |
| `distance_map(state, player)` | (9, 9) steps to goal from every cell |
| `path_length(state, player)`, `shortest_path(state, player)` | |
| `walls_on_path(path)` | walls that would block a given path |
| `render_ascii(state)` | text board |

### Agents – `quoridor_rl.agents`
| | |
|---|---|
| `RandomAgent(seed, move_probability=None)` | uniform random legal action |
| `GreedyPathAgent(epsilon=0)` | walks the shortest path, never walls |
| `HeuristicAgent(epsilon=0)` | 1-ply: minimise my_dist − opp_dist; similar to built-in level 0 but ~25x faster |
| `FunctionAgent(fn)`, `as_agent(obj)` | wrap a function as an Agent |

### Masking – `quoridor_rl.masking`
`masked_fill(values, mask)`, `masked_argmax(values, mask)`, `masked_softmax(logits, mask)`,
`sample_masked(logits, mask)`, `epsilon_greedy(values, mask, eps)` – numpy or torch.

### Evaluation – `quoridor_rl.evaluate`
| | |
|---|---|
| `evaluate(agent, opponent, n_games, opponent_level)` | win rate, alternating sides |
| `play_game(agent0, agent1, max_turns, verbose)` | one game between two agents (e.g. two models) → `GameRecord` |

### Engine bridge – `quoridor_rl.engine_bridge` (imports pygame)
| | |
|---|---|
| `launch_gui({label: agent_factory, ...})` | open the game with these agents as extra player types |
| `BuiltinAI(level)` | engine's minimax AI as an opponent (~0.5 s per move at level 0) |
| `AgentController(agent, board)` | makes an Agent usable as an engine AI |
| `new_engine_board(ai_players, ai_level)` | headless engine Board |
| `state_from_board(board)`, `apply_to_board(board, action)` | sync engine ↔ GameState |
| `to_engine_action(board, action)`, `from_engine_action(a)` | convert action types |

### Models – `models/`
Add a model: copy `models/_template.py` to `models/<name>.py`, rename the class and
`@register_model('<name>')`. Nothing else to change – it is picked up by `train.py` and, once a
checkpoint exists, by the game menu as `RL: <run>`.

| | |
|---|---|
| `@register_model('name')` | make a `QuoridorModel` subclass selectable by name |
| `QuoridorModel` | base class: `forward(obs) -> (B, 140)`, override `action_values(obs)` if forward returns more; attributes `obs_size`, `n_actions`, `include_distances`, `description` |
| `available_models()`, `get_model_class(name)` | what is registered |
| `create_model(name, **kwargs)` | build a model (constructor args are stored for checkpoints) |
| `TorchAgent(net, mode='greedy'\|'sample', epsilon)` | model -> Agent |
| `save_checkpoint(net, run_name=None, **extra)` | -> `checkpoints/<run>.pt` (run defaults to model name) |
| `load_checkpoint(run) -> (net, extra)`, `load_model(run)`, `load_agent(run)` | load by run name or path |
| `list_checkpoints()` | saved runs, newest first |
| `vector_to_planes(obs)` | (B, 294) -> (B, 8, 9, 9) for convolutional models |

### Training – `train.py`
```
python rl-program/train.py --list                      # models + checkpoints
python rl-program/train.py --model cnn --run cnn_v1    # train a model under a run name
python rl-program/train.py --run cnn_v1 --resume       # continue a run
```
Defaults live in `CONFIG` at the top (`model`, `model_kwargs`, `episodes`, ...).
`collect_episode(env, agent)`, `train(cfg, resume)`.

## Notes

* Speed: ~1700 env steps/s vs random/heuristic-free opponents; `'heuristic'` ≈ 20 ms/move,
  `'builtin'` ≈ 0.5 s/move – train against `'heuristic'`, evaluate against `'builtin'`.
* Wall legality follows the official rule (pawns ignored in the path test). The engine also
  treats pawns as obstacles, so very rarely it forbids a wall that is legal here.
* In self-play (`opponent=None`) the reward goes to the player who just moved; give the loser's
  last transition -1 yourself.
