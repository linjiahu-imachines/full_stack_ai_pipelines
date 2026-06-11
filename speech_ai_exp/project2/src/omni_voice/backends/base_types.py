from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol


class OmniRunResult:
    __slots__ = (
        "output_wav_path",
        "audio_duration_s",
        "inference_s",
        "ttfa_s",
        "inner_text",
        "meta",
    )

    def __init__(
        self,
        *,
        output_wav_path: Path,
        audio_duration_s: float,
        inference_s: float,
        ttfa_s: float,
        inner_text: str = "",
        meta: dict[str, Any] | None = None,
    ) -> None:
        self.output_wav_path = output_wav_path
        self.audio_duration_s = audio_duration_s
        self.inference_s = inference_s
        self.ttfa_s = ttfa_s
        self.inner_text = inner_text
        self.meta = meta or {}


class OmniBackend(Protocol):
    def run_turn(self, wav_path: Path, out_wav_path: Path) -> OmniRunResult:
        """Speech in → speech out for one batch turn."""
