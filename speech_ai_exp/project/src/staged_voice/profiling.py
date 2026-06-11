from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class StageProfile:
    """Wall-clock timing for one pipeline turn (batch / non-overlapped)."""

    audio_path: str
    audio_duration_s: float
    asr_s: float
    asr_rtf: float
    llm_ttft_s: float
    llm_generation_s: float
    llm_chars: int
    llm_tokens_per_s_est: float
    tts_s: float
    transcript: str
    reply_text: str
    output_wav_path: str | None
    meta: dict[str, Any] = field(default_factory=dict)

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


def write_profile_json(profile: StageProfile, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(profile.to_json_dict(), indent=2), encoding="utf-8")


def append_profile_jsonl(profile: StageProfile, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(profile.to_json_dict(), ensure_ascii=False) + "\n")
