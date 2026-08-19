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
