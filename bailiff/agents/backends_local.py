"""Offline adapters backed by transformers or llama.cpp."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


class LocalBackendError(RuntimeError):
    """Raised when a local backend cannot be initialized."""


@dataclass
class LocalTransformersBackend:
    """Hugging Face transformers-backed generation."""

    model_name_or_path: str
    device: Optional[str] = None
    max_new_tokens: int = 256
    temperature: float = 0.2
    top_p: float = 0.95

    def __post_init__(self) -> None:
        try:  # pragma: no cover - optional dependency
            import torch  # type: ignore
            from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise LocalBackendError("transformers and torch are required for LocalTransformersBackend") from exc

        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        self._model = AutoModelForCausalLM.from_pretrained(self.model_name_or_path)
        target_device = self.device
        if target_device is None:
            target_device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model.to(target_device)
        self._defaults = {
            "max_new_tokens": int(self.max_new_tokens),
            "temperature": float(self.temperature),
            "top_p": float(self.top_p),
        }

    def __call__(self, prompt: str, **kwargs: object) -> str:
        params = self._merge_params(kwargs)
        encoded = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        with self._torch.no_grad():
            output = self._model.generate(
                **encoded,
                max_new_tokens=int(params["max_new_tokens"]),
                temperature=float(params["temperature"]),
                top_p=float(params["top_p"]),
            )
        completion = output[0][encoded["input_ids"].shape[-1] :]
        text = self._tokenizer.decode(completion, skip_special_tokens=True)
        return text.strip()

    def _merge_params(self, overrides: Dict[str, object]) -> Dict[str, object]:
        params = dict(self._defaults)
        for key in ("max_new_tokens", "temperature", "top_p"):
            if key in overrides:
                params[key] = overrides[key]
        return params


@dataclass
class LlamaCppBackend:
    """llama.cpp-backed offline text generation."""

    model_path: str
    n_ctx: int = 2048
    n_threads: Optional[int] = None
    max_tokens: int = 256
    temperature: float = 0.2
    top_p: float = 0.95

    def __post_init__(self) -> None:
        try:  # pragma: no cover - optional dependency
            from llama_cpp import Llama  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise LocalBackendError("llama-cpp-python is required for LlamaCppBackend") from exc
        self._defaults = {
            "max_tokens": int(self.max_tokens),
            "temperature": float(self.temperature),
            "top_p": float(self.top_p),
        }
        self._llama = Llama(
            model_path=self.model_path,
            n_ctx=self.n_ctx,
            n_threads=self.n_threads,
        )

    def __call__(self, prompt: str, **kwargs: object) -> str:
        params = dict(self._defaults)
        for key in ("max_tokens", "temperature", "top_p"):
            if key in kwargs:
                params[key] = kwargs[key]
        response = self._llama(
            prompt,
            max_tokens=int(params["max_tokens"]),
            temperature=float(params["temperature"]),
            top_p=float(params["top_p"]),
            echo=False,
        )
        choices = response.get("choices", [])
        if not choices:
            return ""
        choice = choices[0]
        if "text" in choice:
            return str(choice["text"]).strip()
        message = choice.get("message", {})
        return str(message.get("content", "")).strip()
