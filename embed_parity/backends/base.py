from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

import numpy as np


class EmbeddingBackend(Protocol):
    """The intentionally small contract used by the comparison engine."""

    name: str

    @property
    def metadata(self) -> dict[str, Any]: ...

    def encode(self, texts: Sequence[str], batch_size: int) -> np.ndarray: ...


class RetrievalEmbeddingBackend(Protocol):
    """Optional role-aware paths used by asymmetric retrieval models."""

    def encode_query(self, texts: Sequence[str], batch_size: int) -> np.ndarray: ...

    def encode_document(self, texts: Sequence[str], batch_size: int) -> np.ndarray: ...


class ConcurrentEmbeddingBackend(Protocol):
    """Optional capability for production-style independent request checks."""

    def encode_concurrently(self, text: str, count: int) -> np.ndarray: ...
