from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any


def synthesize_espeak(
    text: str,
    out_wav_path: Path,
    *,
    voice: str = "en-us",
    rate_wpm: int = 165,
) -> dict[str, Any]:
    binary = shutil.which("espeak-ng") or shutil.which("espeak")
    if not binary:
        raise EnvironmentError(
            "espeak-ng not found on PATH. Install: sudo apt install espeak-ng"
        )
    out_wav_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [binary, "-v", voice, "-s", str(rate_wpm), "-w", str(out_wav_path), text],
        check=True,
        capture_output=True,
        text=True,
    )
    return {"tts_engine": Path(binary).name, "tts_voice": voice, "rate_wpm": rate_wpm}
