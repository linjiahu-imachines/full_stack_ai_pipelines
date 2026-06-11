from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    staged_config: Path = Path("../project/configs/demo_staged_kokoro.yaml")
    sessions_dir: Path = Path("data/sessions")
    static_dir: Path = Path("static")

    @classmethod
    def from_env(cls) -> ServerConfig:
        root = Path(__file__).resolve().parents[1]
        cfg = cls(
            host=os.environ.get("CHAT_HOST", "0.0.0.0"),
            port=int(os.environ.get("CHAT_PORT", "8000")),
            staged_config=Path(
                os.environ.get(
                    "STAGED_CONFIG",
                    str(root / ".." / "project" / "configs" / "demo_staged_kokoro.yaml"),
                )
            ),
            sessions_dir=Path(os.environ.get("SESSIONS_DIR", str(root / "data" / "sessions"))),
            static_dir=Path(os.environ.get("STATIC_DIR", str(root / "static"))),
        )
        cfg.staged_config = cfg.staged_config.expanduser().resolve()
        cfg.sessions_dir = (
            cfg.sessions_dir if cfg.sessions_dir.is_absolute() else (root / cfg.sessions_dir)
        ).resolve()
        cfg.static_dir = (
            cfg.static_dir if cfg.static_dir.is_absolute() else (root / cfg.static_dir)
        ).resolve()
        return cfg
