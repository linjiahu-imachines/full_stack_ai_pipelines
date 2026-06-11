"""
Legacy hook (superseded). On Debian, ``/usr/lib/python3.12/sitecustomize.py`` is
imported before a venv ``sitecustomize.py``, so this file is not used.

``setup_sglang_env.sh`` installs ``_sglang_exploration_rope.pth`` + ``pth_rope_bootstrap.py`` instead.
"""

from __future__ import annotations

import os
import sys


def _main() -> None:
    root = (os.environ.get("SGLANG_EXPLORATION_ROOT") or "").strip()
    if not root:
        return
    d = os.path.join(root, "sglang_test")
    if d not in sys.path:
        sys.path.insert(0, d)
    try:
        import rope_torch_fallback

        rope_torch_fallback.apply()
    except Exception as e:
        print(f"sitecustomize: RoPE fallback hook failed: {e}", file=sys.stderr)


_main()
