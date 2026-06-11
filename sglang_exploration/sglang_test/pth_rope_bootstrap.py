"""
Loaded via a ``*.pth`` line in the venv (see ``setup_sglang_env.sh``).

Runs after ``site`` for every interpreter in this venv, including SGLang worker
processes, without relying on ``sitecustomize.py`` (Debian ships
``/usr/lib/python3.12/sitecustomize.py`` which shadows a venv ``sitecustomize``).
"""

from __future__ import annotations

import os
import sys
from importlib.util import module_from_spec, spec_from_file_location


def _main() -> None:
    if not (os.environ.get("SGLANG_EXPLORATION_ROOT") or "").strip():
        return
    root = os.environ["SGLANG_EXPLORATION_ROOT"].strip()
    mod_path = os.path.join(root, "sglang_test", "rope_torch_fallback.py")
    try:
        spec = spec_from_file_location("rope_torch_fallback", mod_path)
        if spec is None or spec.loader is None:
            return
        mod = module_from_spec(spec)
        spec.loader.exec_module(mod)
        mod.apply()
    except Exception as e:
        print(f"pth_rope_bootstrap: failed: {e}", file=sys.stderr)


_main()
