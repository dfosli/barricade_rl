"""Model registry, base class, agent wrapper and checkpoints.

Adding a model = one new file in this folder (see ``_template.py``):

    from .base import QuoridorModel, register_model

    @register_model('my_model')
    class MyModel(QuoridorModel):
        def __init__(self, hidden=128):
            super().__init__()
            ...
        def forward(self, obs):          # (B, obs_size) -> (B, 140)
            ...

It is found automatically; then ``python rl-program/train.py --model my_model`` trains it and
every saved checkpoint shows up as a player in the game menu.
"""
from __future__ import annotations

import inspect
from pathlib import Path
from typing import Dict, List, Optional, Type, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from quoridor_rl import NUM_ACTIONS, OBS_SLICES, masked_fill, obs_size

CHECKPOINT_DIR = Path(__file__).resolve().parents[1] / 'checkpoints'

_REGISTRY: Dict[str, Type['QuoridorModel']] = {}


# --------------------------------------------------------------------------- registry
def register_model(name: Optional[str] = None):
    """Class decorator: makes the model selectable by ``name`` (default: lower-case class name)."""
    def decorator(cls):
        key = name or cls.__name__.lower()
        if key in _REGISTRY and _REGISTRY[key].__qualname__ != cls.__qualname__:
            raise ValueError('Model name %r is already used by %s' % (key, _REGISTRY[key].__name__))
        cls.model_name = key
        _REGISTRY[key] = cls
        return cls
    return decorator


def available_models() -> List[str]:
    return sorted(_REGISTRY)


def get_model_class(name: str) -> Type['QuoridorModel']:
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError('Unknown model %r. Available: %s' % (name, ', '.join(available_models()))) from None


def create_model(name: str, **kwargs) -> 'QuoridorModel':
    """Build a registered model. Its full constructor arguments (defaults included) are stored in
    ``model.config`` so a checkpoint rebuilds exactly the same network even if defaults change later."""
    cls = get_model_class(name)
    model = cls(**kwargs)
    sig = inspect.signature(cls.__init__)
    bound = sig.bind(None, **kwargs)  # None stands in for self
    bound.apply_defaults()
    config = {}
    for key, value in list(bound.arguments.items())[1:]:
        if sig.parameters[key].kind is inspect.Parameter.VAR_KEYWORD:
            config.update(value)
        else:
            config[key] = value
    model.config = config
    return model


# --------------------------------------------------------------------------- base class
class QuoridorModel(nn.Module):
    """Base class for all models.

    Contract:
      * ``forward(obs)`` takes a float tensor (B, obs_size) from ``encode_observation``.
      * ``action_values(obs)`` must return (B, 140) scores used to pick actions (Q-values or logits).
        By default it is ``forward``; override it if forward returns more (e.g. (logits, value)).
      * ``include_distances``: which observation format the model expects (294 vs 292 values).
    """
    model_name: str = None
    description: str = ''
    include_distances: bool = True

    def __init__(self):
        super().__init__()
        self.config: dict = {}

    @property
    def obs_size(self) -> int:
        return obs_size(self.include_distances)

    @property
    def n_actions(self) -> int:
        return NUM_ACTIONS

    def action_values(self, obs: torch.Tensor) -> torch.Tensor:
        return self(obs)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def vector_to_planes(obs: torch.Tensor) -> torch.Tensor:
    """(B, 292|294) observation -> (B, C, 9, 9) planes for convolutional models.

    Channels: my pawn, opp pawn, h walls, v walls (8x8 padded to 9x9), then one constant plane
    per scalar (walls left x2, and distances x2 if present). C = 6 or 8.
    """
    b = obs.shape[0]
    me = obs[:, OBS_SLICES['my_pawn']].view(b, 9, 9)
    opp = obs[:, OBS_SLICES['opp_pawn']].view(b, 9, 9)
    h = F.pad(obs[:, OBS_SLICES['h_walls']].view(b, 8, 8), (0, 1, 0, 1))
    v = F.pad(obs[:, OBS_SLICES['v_walls']].view(b, 8, 8), (0, 1, 0, 1))
    scalars = obs[:, OBS_SLICES['walls_left'].start:]
    planes = scalars[:, :, None, None].expand(b, scalars.shape[1], 9, 9)
    return torch.cat([torch.stack([me, opp, h, v], dim=1), planes], dim=1)


# --------------------------------------------------------------------------- agent
class TorchAgent:
    """Agent (act(obs, mask) -> int) around a model.

    mode='greedy': masked argmax (DQN / evaluation); mode='sample': sample from masked softmax.
    epsilon: probability of a random legal action (exploration).
    """

    def __init__(self, net: QuoridorModel, device: str = 'cpu', mode: str = 'greedy', epsilon: float = 0.0,
                 seed: int = None):
        self.net = net
        self.device = device
        self.mode = mode
        self.epsilon = epsilon
        self.rng = np.random.default_rng(seed)

    @property
    def include_distances(self) -> bool:
        return getattr(self.net, 'include_distances', True)

    @torch.no_grad()
    def act(self, obs: np.ndarray, mask: np.ndarray) -> int:
        if self.epsilon and self.rng.random() < self.epsilon:
            return int(self.rng.choice(np.flatnonzero(mask)))
        x = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        out = masked_fill(self.net.action_values(x)[0], mask)
        if self.mode == 'sample':
            return int(torch.distributions.Categorical(logits=out).sample())
        return int(out.argmax())


# --------------------------------------------------------------------------- checkpoints
def checkpoint_path(run_name: Union[str, Path]) -> Path:
    """A run name ('mlp', 'cnn_big') -> checkpoints/<run>.pt. Existing paths are returned as-is."""
    p = Path(run_name)
    if p.suffix == '.pt' or p.exists():
        return p
    return CHECKPOINT_DIR / ('%s.pt' % run_name)


def save_checkpoint(model: QuoridorModel, run_name: Optional[str] = None, **extra) -> Path:
    """Save to checkpoints/<run_name>.pt (default run name = model name). ``extra`` is stored too
    (episode, optimizer state, stats, ...)."""
    path = checkpoint_path(run_name or model.model_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'model_name': model.model_name, 'config': model.config,
                'state_dict': model.state_dict(), 'extra': extra}, path)
    return path


def load_checkpoint(run_name: Union[str, Path], device: str = 'cpu'):
    """Returns (model, extra_dict). The model class must still be registered under the same name."""
    data = torch.load(checkpoint_path(run_name), map_location=device, weights_only=False)
    model = create_model(data['model_name'], **data['config']).to(device)
    model.load_state_dict(data['state_dict'])
    model.eval()
    return model, data.get('extra', {})


def load_model(run_name: Union[str, Path], device: str = 'cpu') -> QuoridorModel:
    return load_checkpoint(run_name, device)[0]


def load_agent(run_name: Union[str, Path], device: str = 'cpu', **agent_kwargs) -> TorchAgent:
    """Checkpoint -> ready-to-play TorchAgent (greedy by default)."""
    return TorchAgent(load_model(run_name, device), device=device, **agent_kwargs)


def list_checkpoints() -> List[Path]:
    """Saved checkpoints, most recently modified first."""
    if not CHECKPOINT_DIR.exists():
        return []
    return sorted(CHECKPOINT_DIR.glob('*.pt'), key=lambda p: p.stat().st_mtime, reverse=True)
