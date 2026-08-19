from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import Any

import numpy as np


def raw_embedding(text: str, dimension: int = 24, *, salt: str = "") -> np.ndarray:
    values = []
    counter = 0
    while len(values) < dimension:
        digest = hashlib.sha256(f"{salt}|{text}|{counter}".encode()).digest()
        values.extend((byte - 127.5) / 127.5 for byte in digest)
        counter += 1
    vector = np.asarray(values[:dimension], dtype=np.float64)
    return vector * (1.0 + (len(text) % 7) / 5.0)


class FakeBackend:
    name = "fake"

    def __init__(
        self,
        *,
        dimension: int = 24,
        normalize: bool = True,
        noise: float = 0.0,
        perturb: float = 0.0,
        truncate: int | None = None,
        batch_sensitive: bool = False,
        concurrency_sensitive: bool = False,
        prefix: str | None = None,
        nan: bool = False,
        salt: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.dimension = dimension
        self.normalize = normalize
        self.noise = noise
        self.perturb = perturb
        self.truncate = truncate
        self.batch_sensitive = batch_sensitive
        self.concurrency_sensitive = concurrency_sensitive
        self.prefix = prefix
        self.nan = nan
        self.salt = salt
        self.extra_metadata = metadata or {}

    @property
    def metadata(self) -> dict[str, Any]:
        return {"runtime": self.name, "embedding_dimension": self.dimension, **self.extra_metadata}

    def encode(self, texts: Sequence[str], batch_size: int) -> np.ndarray:
        rows = []
        for index, original in enumerate(texts):
            text = original
            if self.prefix and not text.startswith(self.prefix):
                text = self.prefix + text
            if self.truncate is not None:
                text = " ".join(text.split()[: self.truncate])
            vector = raw_embedding(text, self.dimension, salt=self.salt)
            if self.perturb:
                vector[::2] += self.perturb
            if self.noise:
                rng = np.random.default_rng(index + 42)
                vector += rng.normal(0, self.noise, self.dimension)
            if self.batch_sensitive and batch_size >= 8:
                vector[::3] += 0.5
            if self.normalize:
                vector /= np.linalg.norm(vector)
            rows.append(vector)
        result = np.stack(rows)
        if self.nan:
            result[0, 0] = np.nan
        return result

    def encode_concurrently(self, text: str, count: int) -> np.ndarray:
        result = self.encode([text] * count, batch_size=1)
        if self.concurrency_sensitive:
            result[1::2, ::3] += 1.0
            if self.normalize:
                result[1::2] /= np.linalg.norm(result[1::2], axis=1, keepdims=True)
        return result
