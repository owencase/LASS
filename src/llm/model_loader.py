"""Lazy, offline-only loader for the project-local EXAONE model."""

from __future__ import annotations

import gc
from pathlib import Path
from threading import RLock
from typing import Any, Mapping, Sequence, Tuple

from .model_manager import LocalModelManager
from src.runtime import release_memory, resolve_device


class LocalEXAONEClient:
    """Run EXAONE 4.0 1.2B from local MLX or Transformers safetensors.

    No server process is required and ``local_files_only`` prevents accidental
    network access while loading the tokenizer or model.
    """

    def __init__(
        self,
        model_dir: Path,
        model_id: str = "LGAI-EXAONE/EXAONE-4.0-1.2B",
        device: str = "auto",
        max_new_tokens: int = 384,
        temperature: float = 0.1,
        max_input_tokens: int = 2048,
        backend: str = "transformers",
    ) -> None:
        self.model_dir = model_dir.expanduser().resolve()
        self.model_id = model_id
        self.requested_device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.max_input_tokens = max_input_tokens
        self.backend = backend.strip().lower()
        if self.backend not in {"transformers", "mlx"}:
            raise ValueError(f"Unsupported LLM backend: {backend}")
        self.manager = LocalModelManager(model_id, self.model_dir)
        self._model: Any = None
        self._tokenizer: Any = None
        self._torch: Any = None
        self._device = "cpu"
        self._mlx_generate: Any = None
        self._mlx_sampler: Any = None
        self._lock = RLock()

    def _runtime(self) -> Tuple[Any, Any, Any]:
        with self._lock:
            if self._model is not None:
                return self._torch, self._tokenizer, self._model

            self.manager.require_ready()
            try:
                import torch
                import transformers
                from packaging.version import Version
                from transformers import AutoModelForCausalLM, AutoTokenizer
            except ImportError as exc:
                raise RuntimeError(
                    "Local EXAONE inference requires torch and transformers. "
                    "Install dependencies with `pip install -r requirements.txt`."
                ) from exc

            self._device = self._resolve_device(torch)
            dtype = self._resolve_dtype(torch, self._device)
            self._tokenizer = AutoTokenizer.from_pretrained(
                str(self.model_dir),
                local_files_only=True,
                trust_remote_code=False,
            )
            dtype_argument = (
                {"dtype": dtype}
                if Version(transformers.__version__) >= Version("4.56")
                else {"torch_dtype": dtype}
            )
            self._model = AutoModelForCausalLM.from_pretrained(
                str(self.model_dir),
                local_files_only=True,
                trust_remote_code=False,
                low_cpu_mem_usage=True,
                **dtype_argument,
            )
            self._model.to(self._device)
            self._model.eval()
            self._torch = torch
            return self._torch, self._tokenizer, self._model

    def _mlx_runtime(self) -> Tuple[Any, Any, Any, Any, Any]:
        """Lazy-load the 4-bit Apple-Silicon runtime from local files only."""
        with self._lock:
            if self._model is not None:
                return (
                    self._torch,
                    self._tokenizer,
                    self._model,
                    self._mlx_generate,
                    self._mlx_sampler,
                )
            self.manager.require_ready()
            try:
                import mlx.core as mx
                from mlx_lm import generate, load
                from mlx_lm.sample_utils import make_sampler
            except ImportError as exc:
                raise RuntimeError(
                    "MLX 4-bit inference requires mlx-lm on Apple Silicon. "
                    "Install dependencies with `pip install -r requirements.txt`."
                ) from exc
            try:
                model, tokenizer = load(str(self.model_dir), lazy=True)
            except Exception as exc:
                raise RuntimeError(f"MLX 로컬 모델을 불러올 수 없습니다: {exc}") from exc
            self._torch = mx
            self._tokenizer = tokenizer
            self._model = model
            self._mlx_generate = generate
            self._mlx_sampler = make_sampler
            self._device = "mlx"
            return mx, tokenizer, model, generate, make_sampler

    def _resolve_device(self, torch: Any) -> str:
        return resolve_device(torch, self.requested_device)

    def unload(self) -> None:
        with self._lock:
            mx = self._torch if self.backend == "mlx" else None
            self._model = self._tokenizer = self._torch = None
            self._mlx_generate = self._mlx_sampler = None
            gc.collect()
            if mx is not None:
                try:
                    mx.clear_cache()
                except Exception:
                    pass
            release_memory()

    def _bounded_input(self, tokenizer: Any, conversation: list) -> Any:
        """Bound the KV cache while retaining system instructions and latest question."""
        conversation = [dict(message) for message in conversation]
        while True:
            ids = tokenizer.apply_chat_template(
                conversation, tokenize=True, add_generation_prompt=True,
                return_tensors="pt", enable_thinking=False,
            )
            excess = ids.shape[-1] - self.max_input_tokens
            if excess <= 0:
                return ids
            if len(conversation) > 2:
                del conversation[1:-1]
                continue
            # Keep both the start of retrieved context and the question at its end.
            target = max(conversation, key=lambda item: len(item["content"]))
            tokens = tokenizer.encode(target["content"], add_special_tokens=False)
            budget = len(tokens) - excess - 32
            if budget < 32:
                raise ValueError("입력 지시문이 너무 깁니다. 질문이나 메모를 줄여 주세요.")
            tail = min(512, budget // 2)
            target["content"] = tokenizer.decode(tokens[:budget-tail]) + "\n…\n" + tokenizer.decode(tokens[-tail:])

    def _bounded_mlx_input(self, tokenizer: Any, conversation: list) -> list[int]:
        """Build the same bounded chat prompt without allocating PyTorch tensors."""
        conversation = [dict(message) for message in conversation]
        while True:
            template_options = {
                "tokenize": True,
                "add_generation_prompt": True,
                "enable_thinking": False,
            }
            try:
                ids = tokenizer.apply_chat_template(conversation, **template_options)
            except TypeError:
                template_options.pop("enable_thinking")
                ids = tokenizer.apply_chat_template(conversation, **template_options)
            if hasattr(ids, "tolist"):
                ids = ids.tolist()
            if ids and isinstance(ids[0], list):
                ids = ids[0]
            excess = len(ids) - self.max_input_tokens
            if excess <= 0:
                return list(ids)
            if len(conversation) > 2:
                del conversation[1:-1]
                continue
            target = max(conversation, key=lambda item: len(item["content"]))
            tokens = tokenizer.encode(target["content"], add_special_tokens=False)
            budget = len(tokens) - excess - 32
            if budget < 32:
                raise ValueError("입력 지시문이 너무 깁니다. 질문이나 메모를 줄여 주세요.")
            tail = min(512, budget // 2)
            target["content"] = (
                tokenizer.decode(tokens[: budget - tail])
                + "\n…\n"
                + tokenizer.decode(tokens[-tail:])
            )

    @staticmethod
    def _resolve_dtype(torch: Any, device: str) -> Any:
        if device == "cuda":
            return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        if device == "mps":
            return torch.float16
        return torch.float32

    @property
    def active_device(self) -> str:
        return self._device

    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        max_new_tokens: int | None = None,
    ) -> str:
        return self.chat_messages(
            system_prompt,
            [{"role": "user", "content": user_prompt}],
            max_new_tokens=max_new_tokens,
        )

    def chat_messages(
        self,
        system_prompt: str,
        messages: Sequence[Mapping[str, str]],
        *,
        max_new_tokens: int | None = None,
    ) -> str:
        conversation = [{"role": "system", "content": system_prompt.strip()}]
        for message in messages:
            role = str(message.get("role", ""))
            content = str(message.get("content", "")).strip()
            if role not in {"user", "assistant"} or not content:
                continue
            conversation.append({"role": role, "content": content})
        if len(conversation) == 1 or conversation[-1]["role"] != "user":
            raise ValueError("Chat messages must end with a non-empty user message")
        if self.backend == "mlx":
            return self._chat_mlx(conversation, max_new_tokens=max_new_tokens)

        torch, tokenizer, model = self._runtime()
        input_ids = self._bounded_input(tokenizer, conversation).to(model.device)
        generation_options = {
            "max_new_tokens": self.max_new_tokens if max_new_tokens is None else max(
                1, min(int(max_new_tokens), self.max_new_tokens)
            ),
            "do_sample": self.temperature > 0,
            "pad_token_id": tokenizer.eos_token_id,
        }
        if self.temperature > 0:
            generation_options.update({"temperature": self.temperature, "top_p": 0.95})
        generated = None
        try:
            with self._lock, torch.inference_mode():
                generated = model.generate(
                    input_ids=input_ids,
                    attention_mask=torch.ones_like(input_ids),
                    **generation_options,
                )
            content = tokenizer.decode(
                generated[0, input_ids.shape[-1] :],
                skip_special_tokens=True,
            ).strip()
        finally:
            del input_ids
            if generated is not None:
                del generated
            release_memory()
        if not content:
            raise RuntimeError("EXAONE returned an empty response")
        return content

    def _chat_mlx(
        self,
        conversation: list[dict[str, str]],
        *,
        max_new_tokens: int | None,
    ) -> str:
        mx, tokenizer, model, generate, make_sampler = self._mlx_runtime()
        prompt_tokens = self._bounded_mlx_input(tokenizer, conversation)
        output_limit = self.max_new_tokens if max_new_tokens is None else max(
            1, min(int(max_new_tokens), self.max_new_tokens)
        )
        sampler = make_sampler(
            temp=self.temperature,
            top_p=0.95 if self.temperature > 0 else 0.0,
        )
        try:
            with self._lock:
                content = generate(
                    model,
                    tokenizer,
                    prompt=prompt_tokens,
                    max_tokens=output_limit,
                    verbose=False,
                    sampler=sampler,
                    prefill_step_size=512,
                    kv_bits=8,
                    quantized_kv_start=0,
                ).strip()
        finally:
            del prompt_tokens
            gc.collect()
            mx.clear_cache()
        if not content:
            raise RuntimeError("EXAONE returned an empty response")
        return content
