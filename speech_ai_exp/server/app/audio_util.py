from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


def prepare_wav_for_asr(src: Path, dest_wav: Path) -> Path:
    """Copy or convert input audio to mono WAV for faster-whisper."""
    dest_wav.parent.mkdir(parents=True, exist_ok=True)
    if src.suffix.lower() == ".wav":
        shutil.copy2(src, dest_wav)
        return dest_wav

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError(
            f"Browser sent {src.suffix} audio; install ffmpeg to convert. "
            "Example: sudo apt install ffmpeg"
        )
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-i",
            str(src),
            "-ac",
            "1",
            "-ar",
            "22050",
            str(dest_wav),
        ],
        check=True,
        capture_output=True,
    )
    return dest_wav
