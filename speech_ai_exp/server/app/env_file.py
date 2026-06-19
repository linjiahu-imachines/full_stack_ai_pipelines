from __future__ import annotations

import os
from pathlib import Path


def load_env_file(path: Path, *, override: bool = False) -> bool:
    """Load KEY=VALUE lines into os.environ. Returns True if file was read."""
    if not path.is_file():
        return False
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if not key:
            continue
        if override or key not in os.environ:
            os.environ[key] = value
    return True


def load_server_env(server_root: Path) -> Path | None:
    """Load server/.env.local if present (secrets — not committed to git)."""
    path = (server_root / ".env.local").resolve()
    if load_env_file(path):
        return path
    return None
