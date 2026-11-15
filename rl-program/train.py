"""Training: plain gradient descent on Monte Carlo returns.

    python rl-program/train.py                         # trains CONFIG['model']
    python rl-program/train.py --model cnn --run cnn_v1
    python rl-program/train.py --run cnn_v1 --resume   # continue from checkpoints/cnn_v1.pt
    python rl-program/train.py --list                  # show available models and checkpoints

Loop: play a game -> label each of the agent's moves with the game's (discounted) outcome G ->
loss = mean (Q(state, action) - G)^2 -> one gradient descent step. The network learns to predict
how good each action is; the agent then plays the action with the highest prediction.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from models import (TorchAgent, available_models, create_model, get_model_class, list_checkpoints,
                    load_checkpoint, save_checkpoint)
from quoridor_rl import GreedyPathAgent, HeuristicAgent, QuoridorEnv, RandomAgent, evaluate

CONFIG = {
    'model': 'mlp',               # any name from available_models()
    'model_kwargs': {},           # constructor arguments, e.g. {'hidden': (512, 512)}
    'run_name': None,             # checkpoint name (default: model name) -> checkpoints/<run>.pt
    'episodes': 400,
    'eval_every': 50,
    'eval_games': 20,
    'epsilon': 0.2,               # exploration of the learning agent
    'opponent': 'random',         # training opponent: 'random' (easiest), 'greedy', 'heuristic' (hardest)
    'opponent_epsilon': 0.3,      # for 'greedy'/'heuristic': fraction of random moves (makes them weaker)
    'max_turns': 200,
    'distance_reward': 0.0,       # try e.g. 0.05 for dense shaping
    # --- gradient descent ---
    'learning_rate': 0.01,        # step size: weights -= learning_rate * gradient
    'gamma': 0.99,                # discount: a win k moves later is worth gamma**k now
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'seed': 0,
}


def collect_episode(env: QuoridorEnv, agent) -> list:
    """Play one game; return the agent's transitions as dicts.

    For masked DQN targets you need next_mask (all False when the game is over).
    """
    transitions = []
    obs, info = env.reset()
    done = False
    while not done:
        mask = info['action_mask']
        action = agent.act(obs, mask)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        transitions.append({
            'obs': obs, 'mask': mask, 'action': action, 'reward': reward,
            'next_obs': next_obs, 'next_mask': info['action_mask'],
            'terminated': terminated, 'truncated': truncated,
        })
        obs = next_obs
    return transitions


def compute_returns(transitions: list, gamma: float) -> np.ndarray:
    """Monte Carlo return of every step: G_t = r_t + gamma * G_(t+1), computed backwards.

    With only win/loss rewards this is +-gamma**(steps until the end): the target says
    "this action led to a win/loss, k moves later". A truncated game (draw) gives 0.
    """
    returns = np.zeros(len(transitions), dtype=np.float32)
    g = 0.0
    for t in reversed(range(len(transitions))):
        g = transitions[t]['reward'] + gamma * g
        returns[t] = g
    return returns


def gradient_descent_step(net, transitions: list, returns: np.ndarray, learning_rate: float,
                          device: str) -> float:
    """One plain gradient descent step on the episode. Returns the loss.

    loss = mean over the game's steps of (Q(state, chosen action) - G)^2
    """
    obs = torch.as_tensor(np.stack([t['obs'] for t in transitions]), dtype=torch.float32, device=device)
    actions = torch.as_tensor([t['action'] for t in transitions], dtype=torch.long, device=device)
    targets = torch.as_tensor(returns, device=device)

    # 1. Forward pass: the network's score for the action that was actually played
    q_all = net.action_values(obs)                                  # (steps, 140)
    q_taken = q_all.gather(1, actions.unsqueeze(1)).squeeze(1)      # (steps,)

    # 2. Loss: how far the predictions are from what really happened
    loss = ((q_taken - targets) ** 2).mean()

    # 3. Backward pass: d(loss)/d(weight) for every weight, stored in weight.grad
    net.zero_grad()
    loss.backward()

    # 4. Update: move every weight a small step opposite to its gradient
    #    (this is exactly what torch.optim.SGD(net.parameters(), lr=learning_rate).step() does)
    with torch.no_grad():
        for weight in net.parameters():
            weight -= learning_rate * weight.grad

    return loss.item()


def train(cfg=CONFIG, resume: bool = False):
    torch.manual_seed(cfg['seed'])
    device = cfg['device']
    run_name = cfg['run_name'] or cfg['model']

    start_episode = 0
    if resume:
        net, extra = load_checkpoint(run_name, device)
        net.train()
        start_episode = extra.get('episode', 0)
        print('Resumed %s (%s) from episode %i' % (run_name, net.model_name, start_episode))
    else:
        net = create_model(cfg['model'], **cfg['model_kwargs']).to(device)
    print('Model %s %s: %i parameters -> checkpoints/%s.pt'
          % (net.model_name, net.config, net.num_parameters(), run_name))

    opponents = {
        'random': lambda: RandomAgent(seed=cfg['seed']),
        'greedy': lambda: GreedyPathAgent(epsilon=cfg['opponent_epsilon'], seed=cfg['seed']),
        'heuristic': lambda: HeuristicAgent(epsilon=cfg['opponent_epsilon'], seed=cfg['seed']),
    }
    env = QuoridorEnv(opponent=opponents[cfg['opponent']](),
                      agent_player='random', max_turns=cfg['max_turns'],
                      include_distances=net.include_distances,
                      distance_reward=cfg['distance_reward'], seed=cfg['seed'])
    agent = TorchAgent(net, device=device, epsilon=cfg['epsilon'], seed=cfg['seed'])
    net.train()

    t0 = time.time()
    returns, losses = [], []
    for episode in range(start_episode + 1, start_episode + cfg['episodes'] + 1):
        # Play one game with the current network (+ epsilon exploration) ...
        transitions = collect_episode(env, agent)
        returns.append(sum(t['reward'] for t in transitions))

        # ... then learn from it: label every step with how the game went, one gradient descent step
        targets = compute_returns(transitions, cfg['gamma'])
        losses.append(gradient_descent_step(net, transitions, targets, cfg['learning_rate'], device))

        if episode % cfg['eval_every'] == 0:
            greedy = TorchAgent(net, device=device)
            res_r = evaluate(greedy, opponent='random', n_games=cfg['eval_games'])
            res_h = evaluate(greedy, opponent='heuristic', n_games=cfg['eval_games'])
            print('ep %5i | loss %.4f | avg return %+.2f | win vs random %.0f%% | vs heuristic %.0f%% | %.0fs'
                  % (episode, np.mean(losses[-cfg['eval_every']:]), np.mean(returns[-cfg['eval_every']:]),
                     100 * res_r['win_rate'], 100 * res_h['win_rate'], time.time() - t0))
            save_checkpoint(net, run_name, episode=episode, win_rate_heuristic=res_h['win_rate'])


def print_overview():
    print('Models:')
    for name in available_models():
        print('  %-12s %s' % (name, get_model_class(name).description))
    print('Checkpoints:')
    for path in list_checkpoints() or []:
        print('  %s' % path.stem)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a Quoridor model')
    parser.add_argument('--model', help='model name (default: CONFIG["model"])')
    parser.add_argument('--run', help='checkpoint / run name (default: model name)')
    parser.add_argument('--episodes', type=int)
    parser.add_argument('--resume', action='store_true', help='continue from checkpoints/<run>.pt')
    parser.add_argument('--list', action='store_true', help='list models and checkpoints, then exit')
    args = parser.parse_args()

    if args.list:
        print_overview()
    else:
        cfg = dict(CONFIG)
        if args.model:
            cfg['model'] = args.model
        if args.run:
            cfg['run_name'] = args.run
        if args.episodes:
            cfg['episodes'] = args.episodes
        train(cfg, resume=args.resume)
