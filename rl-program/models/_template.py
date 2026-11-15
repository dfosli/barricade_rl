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


@register_model('my_model')
class MyModel(QuoridorModel):
    description = 'Describe your model here'

    def __init__(self, hidden: int = 128):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(self.obs_size, hidden), nn.ReLU())
        self.policy = nn.Linear(hidden, self.n_actions)
        self.value = nn.Linear(hidden, 1)

    def forward(self, obs: torch.Tensor):
        x = self.body(obs)
        return self.policy(x), self.value(x)

    def action_values(self, obs: torch.Tensor) -> torch.Tensor:
        return self(obs)[0]
