"""TEMPLATE for a new model (this file itself is ignored because its name starts with '_').

1. Copy it to models/<something>.py
2. Rename the class and the name in @register_model(...)
3. Train it:   python rl-program/train.py --model my_model --run my_model_v1
   Play it:    python rl-program/main.py  ->  choose "RL: my_model_v1" in the menu

Rules for a model:
  * subclass QuoridorModel and call super().__init__() first
  * constructor arguments must be plain values (numbers, tuples, strings): they are stored in the
    checkpoint and passed again when the model is loaded
  * forward(obs) gets (B, self.obs_size) floats and returns (B, 140) action scores
    (use vector_to_planes(obs) if you want (B, C, 9, 9) board planes)
  * if forward returns more than the scores (e.g. actor-critic (logits, value)), override
    action_values(obs) to return only the (B, 140) scores
  * set include_distances = False to train without the two path-length features (292 inputs)
"""
import torch
import torch.nn as nn

from .base import QuoridorModel, register_model


@register_model('test_mlp')
class MLP(QuoridorModel):
    description = 'test mlp with 2 hidden layers of 5 each'

    def __init__(self):
        super().__init__()
        layers, size = [], self.obs_size
        hidden = 512
        self.net = nn.Sequential(
            nn.Linear(size, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, self.n_actions)
        )

    def forward(self, obs: torch.Tensor):
        return self.net(obs)
