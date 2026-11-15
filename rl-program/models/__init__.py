"""All models. Every ``*.py`` file in this folder (except ``base.py`` and files starting with ``_``)
is imported automatically, so a model registered with ``@register_model('name')`` in a new file is
available everywhere without touching any other code.

    from models import available_models, create_model, load_agent
    net = create_model('mlp', hidden=(512, 512))
"""
import importlib
import pkgutil
import traceback

from .base import (CHECKPOINT_DIR, QuoridorModel, TorchAgent, available_models, checkpoint_path,
                   create_model, get_model_class, list_checkpoints, load_agent, load_checkpoint,
                   load_model, register_model, save_checkpoint, vector_to_planes)

for _module in pkgutil.iter_modules(__path__):
    if _module.name == 'base' or _module.name.startswith('_'):
        continue
    try:
        importlib.import_module('%s.%s' % (__name__, _module.name))
    except Exception:  # a broken model file should not break the others
        print('WARNING: could not load models/%s.py:\n%s' % (_module.name, traceback.format_exc()))
