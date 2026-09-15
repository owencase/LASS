"""Lazy-loaded local embeddings using the existing Transformers runtime."""

from __future__ import annotations

from itertools import islice
from typing import Any, Iterable, Iterator, List, Sequence, Tuple
from src.runtime import release_memory, resolve_device


class LocalEmbedder:
    def __init__(self, model_name: str = "BAAI/bge-m3", device: str | None = None) -> None:
        self.model_name = model_name
        self.device = device
        self._model: Any = None
        self._tokenizer: Any = None
        self._torch: Any = None
        self._active_device = "cpu"

    def _get_model(self) -> Tuple[Any, Any, Any]:
        if self._model is None:
            try:
                import torch
                import transformers
                from packaging.version import Version
                from transformers import AutoModel, AutoTokenizer
            except ImportError as exc:
                raise RuntimeError(
                    "torch and transformers are required for embeddings. "
                    "Install dependencies with `pip install -r requirements.txt`."
                ) from exc
            self._active_device = self._resolve_device(torch)
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            dtype = torch.float16 if self._active_device != "cpu" else torch.float32
            dtype_argument = (
                {"dtype": dtype}
                if Version(transformers.__version__) >= Version("4.56")
                else {"torch_dtype": dtype}
            )
            self._model = AutoModel.from_pretrained(
                self.model_name,
                low_cpu_mem_usage=True,
                **dtype_argument,
            )
            self._model.to(self._active_device)
            self._model.eval()
            self._torch = torch
        return self._torch, self._tokenizer, self._model

    def _resolve_device(self, torch: Any) -> str:
        return resolve_device(torch, self.device or "auto")

    def unload(self) -> None:
        self._model = self._tokenizer = self._torch = None
        release_memory()

    def _prepared_text(self, text: str, kind: str) -> str:
        if "multilingual-e5" in self.model_name.lower():
            return f"{'query' if kind == 'query' else 'passage'}: {text}"
        return text

    def iter_embeddings(
        self, texts: Iterable[str], *, kind: str = "passage"
    ) -> Iterator[List[float]]:
        """Yield tiny batches so a long document never creates a vector list."""
        torch, tokenizer, model = self._get_model()
        iterator = iter(texts)
        while True:
            batch = list(islice(iterator, 2))
            if not batch:
                break
            encoded = tokenizer(
                [self._prepared_text(text, kind) for text in batch],
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            encoded = {key: value.to(self._active_device) for key, value in encoded.items()}
            with torch.inference_mode():
                output = model(**encoded).last_hidden_state
                mask = encoded["attention_mask"].unsqueeze(-1).to(output.dtype)
                pooled = (output * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
                normalized = torch.nn.functional.normalize(pooled, p=2, dim=1)
            values = normalized.cpu().float().tolist()
            del encoded, output, mask, pooled, normalized
            for value in values:
                yield value

    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        return list(self.iter_embeddings(texts)) if texts else []

    def embed_query(self, text: str) -> List[float]:
        if not text.strip():
            raise ValueError("Query text must not be empty")
        return next(self.iter_embeddings([text], kind="query"))
