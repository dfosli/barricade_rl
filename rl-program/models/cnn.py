import torch
import torch.nn as nn

from .base import QuoridorModel, register_model, vector_to_planes


@register_model('cnn')
class QuoridorCNN(QuoridorModel):
    description = 'Small convolutional network on the 9x9 board planes (example of a second model)'

    def __init__(self, channels: int = 64, n_conv: int = 3, hidden: int = 256):
        super().__init__()
        in_ch = 4 + (4 if self.include_distances else 2)
        convs = []
        for _ in range(n_conv):
            convs += [nn.Conv2d(in_ch, channels, kernel_size=3, padding=1), nn.ReLU()]
            in_ch = channels
        self.conv = nn.Sequential(*convs)
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(channels * 81, hidden), nn.ReLU(),
                                  nn.Linear(hidden, self.n_actions))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.head(self.conv(vector_to_planes(obs)))
