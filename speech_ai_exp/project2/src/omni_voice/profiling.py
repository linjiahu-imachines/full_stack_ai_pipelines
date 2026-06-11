from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class OmniTurnProfile:
    """Wall-clock profile for one voice-to-voice turn (batch mode)."""

    stack: str
    backend: str
    audio_path: str
    audio_duration_s: float
    e2e_wall_s: float
    ttfa_s: float
    output_wav_path: str | None
    transcript: str = ""
    reply_text: str = ""
    inner_text: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


def write_profile_json(profile: OmniTurnProfile, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(profile.to_json_dict(), indent=2), encoding="utf-8")
