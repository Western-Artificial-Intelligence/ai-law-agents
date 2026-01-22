"""Offline adapters backed by transformers or llama.cpp with token budget tracking."""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Dict, Optional

from ..core.token_budget import check_run_allowed, register_token_usage


class LocalBackendError(RuntimeError):
    """Raised when a local backend cannot be initialized."""


@dataclass
class LocalTransformersBackend:
    """Hugging Face transformers-backed generation with token tracking."""

    model_name_or_path: str
    device: Optional[str] = None
    load_in_4bit: bool = False
    max_new_tokens: int = 256
    temperature: float = 0.2
    top_p: float = 0.95
    enforce_budget: bool = True

    def __post_init__(self) -> None:
        try:  # pragma: no cover - optional dependency
            import torch  # type: ignore
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise LocalBackendError("transformers and torch are required for LocalTransformersBackend") from exc

        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        
        quantization_config = None
        if self.load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
            )

        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path,
            quantization_config=quantization_config,
            device_map="auto" if self.load_in_4bit else None,
        )
        
        if not self.load_in_4bit:
            target_device = self.device
            if target_device is None:
                target_device = "cuda" if torch.cuda.is_available() else "cpu"
            self._model.to(target_device)
            
        self._defaults = {
            "max_new_tokens": int(self.max_new_tokens),
            "temperature": float(self.temperature),
            "top_p": float(self.top_p),
        }
        self._run_id = str(uuid.uuid4())

    def _merge_params(self, kwargs: Dict[str, object]) -> Dict[str, object]:
        """Apply defaults and merge with provided kwargs."""
        params = self._defaults.copy()
        params.update(kwargs)
        return params

    def __call__(self, prompt: str, **kwargs: object) -> str:
        params = self._merge_params(kwargs)
        
        if self.enforce_budget:
            # Get accurate token count from tokenizer
            input_tokens = len(self._tokenizer.encode(prompt))
            output_tokens = params["max_new_tokens"]
            total_tokens = input_tokens + output_tokens
            
            # Check budget
            allowed, reason = check_run_allowed(
                run_id=self._run_id,
                model_identifier=self.model_name_or_path,
                api_key_id="local",
                estimated_tokens=total_tokens
            )
            
            if not allowed:
                raise RuntimeError(f"Token budget exceeded: {reason}")
        
        encoded = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        with self._torch.no_grad():
            output = self._model.generate(
                **encoded,
                max_new_tokens=int(params["max_new_tokens"]),
                temperature=float(params["temperature"]),
                top_p=float(params["top_p"]),
            )
        
        # Decode and strip prompt
        decoded = self._tokenizer.decode(output[0], skip_special_tokens=True)
        result = decoded[len(self._tokenizer.decode(encoded["input_ids"][0], skip_special_tokens=True)) :]
        
        if self.enforce_budget:
            # Log actual token usage
            input_tokens = len(encoded["input_ids"][0])
            output_tokens = len(output[0]) - len(encoded["input_ids"][0])
            
            register_token_usage(
                run_id=self._run_id,
                model_identifier=self.model_name_or_path,
                api_key_id="local",
                tokens_prompt=input_tokens,
                tokens_completion=output_tokens
            )
        
        return result


@dataclass
class LlamaCppBackend:
    """llama.cpp via python bindings with token tracking."""

    model_path: str
    n_ctx: int = 2048
    n_threads: int = 4
    n_batch: int = 512
    temperature: float = 0.2
    top_p: float = 0.95
    enforce_budget: bool = True
    
    def __post_init__(self) -> None:
        try:  # pragma: no cover - optional dependency
            from llama_cpp import Llama  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise LocalBackendError("llama-cpp-python is required for LlamaCppBackend") from exc
        
        self._Llama = Llama
        self._defaults = {
            "max_tokens": 256,
            "temperature": float(self.temperature),
            "top_p": float(self.top_p),
        }
        self._llama = self._Llama(
            model_path=str(self.model_path),
            n_ctx=int(self.n_ctx),
            n_threads=int(self.n_threads),
            n_batch=int(self.n_batch),
        )
        self._run_id = str(uuid.uuid4())
    
    def _merge_params(self, kwargs: Dict[str, object]) -> Dict[str, object]:
        """Apply defaults and merge with provided kwargs."""
        params = self._defaults.copy()
        params.update(kwargs)
        return params
    
    def __call__(self, prompt: str, **kwargs: object) -> str:
        params = self._merge_params(kwargs)
        
        if self.enforce_budget:
            # Get accurate token count when possible
            try:
                input_tokens = len(self._llama.tokenize(prompt.encode('utf-8')))
            except:
                # Fall back to rough estimate
                input_tokens = len(prompt) // 4
                
            output_tokens = params.get("max_tokens", 256)
            total_tokens = input_tokens + output_tokens
            
            # Check budget
            allowed, reason = check_run_allowed(
                run_id=self._run_id,
                model_identifier=self.model_path,
                api_key_id="local",
                estimated_tokens=total_tokens
            )
            
            if not allowed:
                raise RuntimeError(f"Token budget exceeded: {reason}")
        
        output = self._llama(
            prompt=prompt,
            max_tokens=int(params["max_tokens"]),
            temperature=float(params["temperature"]),
            top_p=float(params["top_p"]),
            echo=False,
        )
        
        if self.enforce_budget:
            # Record actual token usage when possible
            completion_text = output.get("choices", [{}])[0].get("text", "")
            
            try:
                completion_tokens = len(self._llama.tokenize(completion_text.encode('utf-8')))
            except:
                # Fall back to rough estimate
                completion_tokens = len(completion_text) // 4
            
            register_token_usage(
                run_id=self._run_id,
                model_identifier=self.model_path,
                api_key_id="local",
                tokens_prompt=input_tokens,
                tokens_completion=completion_tokens
            )
        
        return output.get("choices", [{}])[0].get("text", "")
