#!/usr/bin/env python3
"""Dev launcher — prefer `pip install -e .` and `staged-voice-run` on PATH."""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))
    from staged_voice.cli import main as cli_main

    return cli_main()


if __name__ == "__main__":
    raise SystemExit(main())
