"""
Secrets loader — reads .env from the project root and populates os.environ.
Call load() once at startup; all other modules read from os.environ normally.
"""

from __future__ import annotations

import os
from pathlib import Path


def load(env_path: str | Path | None = None) -> dict[str, str]:
    """
    Load key=value pairs from a .env file into os.environ (without overwriting
    already-set variables so real env vars always win).

    Returns the dict of keys that were newly loaded.
    """
    path = Path(env_path) if env_path else _find_env()
    if path is None or not path.exists():
        return {}

    loaded: dict[str, str] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key   = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and value and key not in os.environ:
            os.environ[key] = value
            loaded[key] = value

    return loaded


def require(*keys: str) -> dict[str, str]:
    """
    Return {key: value} for each key.
    Raises EnvironmentError listing any missing or empty keys.
    """
    load()
    missing = [k for k in keys if not os.environ.get(k)]
    if missing:
        raise EnvironmentError(
            f"Missing required secrets: {', '.join(missing)}\n"
            f"Set them in .env or as environment variables."
        )
    return {k: os.environ[k] for k in keys}


def get(key: str, default: str = "") -> str:
    load()
    return os.environ.get(key, default)


def _find_env() -> Path | None:
    """Walk up from cwd looking for a .env file."""
    here = Path.cwd()
    for parent in [here, *here.parents]:
        candidate = parent / ".env"
        if candidate.exists():
            return candidate
    return None
