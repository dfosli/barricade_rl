# Barricade RL

Training reinforcement learning (RL) agents to play **Quoridor**, also known as **Barricade**.

Each player races a pawn to the opposite side of a 9x9 board. On every turn you either move your
pawn or place a wall to lengthen your opponent's path, but you may never block it completely.

## Motivation

- **Emerging strategies:** see which strategies trained models discover on their own: when they
  place walls, how they defend, and whether they find tactics a human would not.
- **Beat me:** train a model that can beat me in a game.

## Status

🚧 **In development.** The game engine, training environment and a first training loop (plain
gradient descent on game outcomes, but it sucks) work. The project will be updated with better models and
different training techniques soon.

## Project structure

```
engine/          the playable game (pygame), with the original minimax AI
rl-program/      the RL side
  main.py        start the game: play against built-in AI or any trained model
  train.py       train a model
  models/        neural network models (add a new file to add a model)
  quoridor_rl/   fast game rules, encodings, training environment, evaluation
  tests/         tests for rules, environment and models
```

More details about the training toolkit and all of its functions are in
[rl-program/README.md](rl-program/README.md).

## Getting started

Requires Python 3.10+.

```bash
pip install -r requirements.txt

# Play: choose Human / Built-in AI / trained models for each side in the menu
python rl-program/main.py

# Train a model (see CONFIG at the top of train.py for settings)
python rl-program/train.py --model mlp --run my_first_model

# List available models and saved checkpoints
python rl-program/train.py --list

# Run the tests
python rl-program/tests/test_quoridor_rl.py
```

Trained models are saved in `rl-program/checkpoints/` and show up automatically in the game menu
as `RL: <run name>`.

## Rules used

Standard Quoridor rules for two players, with 10 walls each. When you jump over an adjacent
opponent you must jump straight. You may only jump diagonally (to the side of the opponent) when
a wall or the board edge is directly behind the opponent.

## Credits

The game engine is based on
[harshitap1305/Quoridor-Game](https://github.com/harshitap1305/Quoridor-Game/tree/main),
with modifications to fit this project:

- a settings menu (players, AI level, debug mode) and a play-again option
- headless mode for fast simulation during training
- support for plugging trained agents in as players
- the jump rule above
- fixes to the built-in AI's state caching
