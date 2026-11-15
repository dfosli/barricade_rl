from typing import Sequence

import torch
import torch.nn as nn

from .base import QuoridorModel, register_model


@register_model('mlp')
class QuoridorMLP(QuoridorModel):
    description = 'Fully connected network on the 294-value observation'

    def __init__(self, hidden: Sequence[int] = (256, 256)):
        super().__init__()
        layers, size = [], self.obs_size
        for h in hidden:
            layers += [nn.Linear(size, h), nn.ReLU()]
            size = h
        layers.append(nn.Linear(size, self.n_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)
