"""Workspace entry point for the Quoridor game."""

import sys
from pathlib import Path

ENGINE_DIRECTORY = Path(__file__).resolve().parent.parent / "engine"
sys.path.insert(0, str(ENGINE_DIRECTORY))

from src.main import main


level = 0 #0-99, for current ai (will be replaced)
debug_on = False
caching_on = False
tags = []

if level or debug_on or caching_on:
    if level:
        tags.append("--level")
        tags.append(str(level))
    if debug_on:
        tags.append("--debug")
    if caching_on:
        tags.append("--cache")
print(tags)


if __name__ == "__main__":
    if tags:
        sys.argv[1:1] = tags

    raise SystemExit(main())