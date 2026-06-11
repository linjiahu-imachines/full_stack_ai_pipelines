from __future__ import annotations

import threading
from collections.abc import Iterator
from typing import Any

from staged_voice.backends.base_types import ChatMessage


class HFCausalLM:
    def __init__(
        self,
        model_name: str,
        *,
        device: str = "auto",
        torch_dtype: str = "auto",
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

        self._torch = torch
        self._TextIteratorStreamer = TextIteratorStreamer

        dtype = self._resolve_dtype(torch, torch_dtype)
        self._tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self._model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map=device if device != "cpu" else None,
            trust_remote_code=True,
        )
        if device == "cpu":
            self._model = self._model.to("cpu")
        self._model.eval()

    @staticmethod
    def _resolve_dtype(torch: Any, name: str) -> Any:
        if name == "bf16":
            return torch.bfloat16
        if name == "fp16":
            return torch.float16
        if name == "fp32":
            return torch.float32
        return "auto"

    def iter_chat_messages(
        self,
        *,
        messages: list[ChatMessage],
        system_prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> Iterator[str]:
        full: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
        for m in messages:
            role = m["role"]
            if role not in ("user", "assistant"):
                continue
            full.append({"role": role, "content": m["content"]})

        if hasattr(self._tokenizer, "apply_chat_template"):
            prompt = self._tokenizer.apply_chat_template(
                full,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            parts = [f"System:\n{system_prompt}\n"]
            for m in full[1:]:
                parts.append(f"{m['role'].capitalize()}:\n{m['content']}\n")
            parts.append("Assistant:\n")
            prompt = "\n".join(parts)

        inputs = self._tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

        streamer = self._TextIteratorStreamer(
            self._tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )
        gen_kwargs: dict[str, Any] = {
            **inputs,
            "streamer": streamer,
            "max_new_tokens": max_tokens,
            "do_sample": temperature > 0,
            "temperature": temperature if temperature > 0 else None,
        }
        if gen_kwargs.get("temperature") is None:
            gen_kwargs.pop("temperature", None)

        thread = threading.Thread(
            target=self._model.generate,
            kwargs=gen_kwargs,
            daemon=True,
        )
        thread.start()
        for text in streamer:
            if text:
                yield text
        thread.join()
