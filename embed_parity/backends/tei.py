from __future__ import annotations

from typing import Any, Sequence

import httpx
import numpy as np


class TEIBackend:
    name = "TEI"

    def __init__(
        self,
        url: str,
        *,
        timeout: float = 120.0,
        client: httpx.Client | None = None,
    ) -> None:
        self.url = url.rstrip("/")
        self.client = client or httpx.Client(timeout=timeout)
        self._owns_client = client is None
        self._metadata = self._fetch_metadata()

    def _fetch_metadata(self) -> dict[str, Any]:
        data: dict[str, Any] = {"runtime": self.name, "url": self.url}
        # TEI versions expose different metadata surfaces. Absence is not an error.
        for path in ("/info", "/"):
            try:
                response = self.client.get(self.url + path)
                if response.is_success and "application/json" in response.headers.get("content-type", ""):
                    payload = response.json()
                    if isinstance(payload, dict):
                        data["server_metadata"] = payload
                        self._promote_metadata(data, payload)
                        break
            except (httpx.HTTPError, ValueError):
                continue
        return data

    @staticmethod
    def _promote_metadata(target: dict[str, Any], payload: dict[str, Any]) -> None:
        """Promote only recognized TEI metadata without guessing absent values."""
        aliases = {
            "model_id": ("model_id", "modelId", "model_name"),
            "resolved_revision": ("revision", "model_sha", "sha"),
            "dtype": ("dtype",),
            "embedding_dimension": ("embedding_dimension", "dimension", "dim"),
            "tokenizer_max_length": ("max_input_length", "max_seq_length", "max_length"),
        }
        containers = [payload]
        for key in ("model", "config", "model_config"):
            nested = payload.get(key)
            if isinstance(nested, dict):
                containers.append(nested)
        for output_key, input_keys in aliases.items():
            for container in containers:
                value = next((container[key] for key in input_keys if key in container), None)
                if isinstance(value, (str, int, float, bool)) and value != "":
                    target[output_key] = value
                    break

    @property
    def metadata(self) -> dict[str, Any]:
        return dict(self._metadata)

    def encode(self, texts: Sequence[str], batch_size: int) -> np.ndarray:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        rows: list[list[float]] = []
        for start in range(0, len(texts), batch_size):
            chunk = list(texts[start : start + batch_size])
            response = self.client.post(
                self.url + "/embed",
                json={"inputs": chunk, "truncate": True},
            )
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, list):
                raise RuntimeError("TEI /embed returned a non-list response")
            # A one-input response is still expected to be a matrix in TEI.
            if payload and isinstance(payload[0], (int, float)):
                payload = [payload]
            if len(payload) != len(chunk):
                raise RuntimeError(
                    f"TEI /embed returned {len(payload)} vectors for {len(chunk)} inputs"
                )
            rows.extend(payload)
        try:
            return np.asarray(rows, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise RuntimeError("TEI returned malformed or ragged embeddings") from exc

    def close(self) -> None:
        if self._owns_client:
            self.client.close()

    def __enter__(self) -> "TEIBackend":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
