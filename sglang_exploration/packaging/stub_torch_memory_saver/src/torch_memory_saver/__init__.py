"""Minimal stub API expected by SGLang's ``torch_memory_saver_adapter``."""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from typing import Any

__version__ = "0.0.9"


class _NoopTorchMemorySaver:
    @contextmanager
    def region(self, *_args: Any, **_kwargs: Any):
        yield

    def cuda_graph(self, **_kwargs: Any):
        return nullcontext()

    def disable(self) -> None:
        return None

    def pause(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def resume(self, *_args: Any, **_kwargs: Any) -> None:
        return None


def configure_subprocess() -> None:
    return None


torch_memory_saver = _NoopTorchMemorySaver()
