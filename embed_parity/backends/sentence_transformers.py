from __future__ import annotations

from typing import Any, Sequence

import numpy as np


class SentenceTransformersBackend:
    name = "SentenceTransformers"

    def __init__(
        self,
        model_id: str,
        *,
        revision: str | None = None,
        device: str | None = None,
        dtype: str | None = None,
        normalize: bool | None = None,
    ) -> None:
        try:
            import sentence_transformers
            import torch
            import transformers
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - depends on optional package
            raise RuntimeError(
                "SentenceTransformers is not installed; install embed-parity[sentence-transformers]"
            ) from exc

        model_kwargs: dict[str, Any] = {}
        if dtype:
            torch_dtype = getattr(torch, dtype, None)
            if torch_dtype is None:
                raise ValueError(f"unknown torch dtype: {dtype}")
            model_kwargs["torch_dtype"] = torch_dtype
        kwargs: dict[str, Any] = {"revision": revision, "device": device}
        if model_kwargs:
            kwargs["model_kwargs"] = model_kwargs
        self.model = SentenceTransformer(model_id, **kwargs)
        self.model_id = model_id
        self.normalize = normalize
        self._torch = torch
        self._versions = {
            "sentence_transformers_version": sentence_transformers.__version__,
            "transformers_version": transformers.__version__,
            "torch_version": torch.__version__,
        }

    @property
    def tokenizer(self):
        return self.model.tokenizer

    @property
    def metadata(self) -> dict[str, Any]:
        first_parameter = next(self.model.parameters(), None)
        revision = None
        auto_model = getattr(self.model, "_first_module", lambda: None)()
        config = getattr(getattr(auto_model, "auto_model", None), "config", None)
        commit_hash = getattr(config, "_commit_hash", None)
        if commit_hash:
            revision = commit_hash
        return {
            "runtime": self.name,
            "model_id": self.model_id,
            "resolved_revision": revision,
            **self._versions,
            "device": str(self.model.device),
            "dtype": str(first_parameter.dtype).removeprefix("torch.") if first_parameter is not None else None,
            "embedding_dimension": self.model.get_sentence_embedding_dimension(),
            "normalization_setting": self.normalize if self.normalize is not None else "model_default",
            "normalization_module_present": any(
                module.__class__.__name__ == "Normalize" for module in self.model.modules()
            ),
            "tokenizer_max_length": getattr(self.tokenizer, "model_max_length", None),
        }

    def encode(self, texts: Sequence[str], batch_size: int) -> np.ndarray:
        kwargs: dict[str, Any] = {
            "batch_size": batch_size,
            "convert_to_numpy": True,
            "show_progress_bar": False,
        }
        if self.normalize is not None:
            kwargs["normalize_embeddings"] = self.normalize
        return np.asarray(self.model.encode(list(texts), **kwargs), dtype=np.float64)

    def text_at_token_length(self, target: int) -> str:
        """Create deterministic text with exactly target tokenizer tokens, if possible."""
        tokenizer = self.tokenizer
        special = tokenizer.num_special_tokens_to_add(pair=False)
        content_length = max(0, target - special)
        seed = tokenizer.encode(
            "Parity probes repeat deterministic language, numbers 123, Unicode cafe.",
            add_special_tokens=False,
            truncation=False,
        )
        if not seed:
            raise RuntimeError("tokenizer produced no tokens for length probe seed")
        ids = [seed[i % len(seed)] for i in range(content_length)]
        text = tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        actual = len(tokenizer.encode(text, add_special_tokens=True, truncation=False))
        # Decode/encode is not always bijective. Adjust deterministically using one-token pieces.
        piece = tokenizer.decode([seed[0]], skip_special_tokens=True).strip() or "x"
        attempts = 0
        while actual < target and attempts < target + 16:
            text += " " + piece
            actual = len(tokenizer.encode(text, add_special_tokens=True, truncation=False))
            attempts += 1
        while actual > target and text and attempts < target * 2 + 32:
            text = text[:-1]
            actual = len(tokenizer.encode(text, add_special_tokens=True, truncation=False))
            attempts += 1
        if actual != target:
            raise RuntimeError(f"could not construct exactly {target} tokens (got {actual})")
        return text
