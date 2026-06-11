from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any


class EspeakNgTTS:
    """Wrap `espeak-ng` for dependency-light local synthesis (quality is utilitarian)."""

    def __init__(self, *, voice: str = "en-us", rate_wpm: int = 175) -> None:
        self._binary = shutil.which("espeak-ng") or shutil.which("espeak")
        if not self._binary:
            raise EnvironmentError(
                "espeak-ng not found on PATH. Install with e.g. `sudo apt install espeak-ng`."
            )
        self._voice = voice
        self._rate_wpm = int(rate_wpm)

    def synthesize_file(self, text: str, out_wav_path: Path) -> dict[str, Any]:
        out_wav_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            self._binary,
            "-v",
            self._voice,
            "-s",
            str(self._rate_wpm),
            "-w",
            str(out_wav_path),
            text,
        ]
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        return {"tts_engine": Path(self._binary).name, "tts_voice": self._voice}
