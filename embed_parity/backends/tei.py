from __future__ import annotations

import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from typing import Any

import httpx
import numpy as np


class TEIBackend:
    name = "TEI"
    # TEI rejects empty or whitespace-only inputs during request validation.
    supports_empty_inputs = False

    def __init__(
        self,
        url: str,
        *,
        timeout: float = 120.0,
        api_key: str | None = None,
        headers: dict[str, str] | None = None,
        truncate: bool = True,
        retries: int = 2,
        retry_backoff: float = 0.25,
        client: httpx.Client | None = None,
    ) -> None:
        self.url = url.rstrip("/")
        request_headers = dict(headers or {})
        if api_key:
            request_headers.setdefault("Authorization", f"Bearer {api_key}")
        self.client = client or httpx.Client(timeout=timeout, headers=request_headers)
        self._owns_client = client is None
        self.truncate = truncate
        self.retries = retries
        self.retry_backoff = retry_backoff
        self._metadata = self._fetch_metadata()

    def _fetch_metadata(self) -> dict[str, Any]:
        data: dict[str, Any] = {"runtime": self.name, "url": self.url}
        # TEI versions expose different metadata surfaces. Absence is not an error.
        for path in ("/info", "/"):
            try:
                response = self.client.get(self.url + path)
                if response.is_success and "application/json" in response.headers.get(
                    "content-type", ""
                ):
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
            "dtype": ("dtype", "model_dtype"),
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
        model_type = payload.get("model_type")
        embedding = model_type.get("embedding") if isinstance(model_type, dict) else None
        pooling = embedding.get("pooling") if isinstance(embedding, dict) else None
        if isinstance(pooling, str) and pooling:
            target["pooling"] = pooling

    @property
    def metadata(self) -> dict[str, Any]:
        return {**self._metadata, "request_truncation": self.truncate}

    def wait_until_ready(self, timeout: float) -> None:
        """Wait for TEI's health endpoint and fail with a useful timeout message."""
        deadline = time.monotonic() + timeout
        last_error: str | None = None
        while time.monotonic() < deadline:
            try:
                response = self.client.get(self.url + "/health")
                if response.is_success:
                    self._metadata = self._fetch_metadata()
                    return
                last_error = f"HTTP {response.status_code}: {response.text[:200]}"
            except httpx.RequestError as exc:
                last_error = str(exc)
            time.sleep(min(0.25, max(0.0, deadline - time.monotonic())))
        raise RuntimeError(
            f"TEI did not become ready within {timeout:g}s ({last_error or 'no response'})"
        )

    def _embed_request(self, chunk: list[str]) -> httpx.Response:
        retryable_statuses = {408, 425, 429, 500, 502, 503, 504}
        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                response = self.client.post(
                    self.url + "/embed",
                    json={"inputs": chunk, "truncate": self.truncate},
                )
                if response.status_code not in retryable_statuses or attempt == self.retries:
                    return response
            except httpx.RequestError as exc:
                last_error = exc
                if attempt == self.retries:
                    break
            time.sleep(self.retry_backoff * (2**attempt))
        raise RuntimeError(
            f"TEI /embed request failed after {self.retries + 1} attempts: {last_error}"
        ) from last_error

    def encode(self, texts: Sequence[str], batch_size: int) -> np.ndarray:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        rows: list[list[float]] = []
        for start in range(0, len(texts), batch_size):
            chunk = list(texts[start : start + batch_size])
            response = self._embed_request(chunk)
            if not response.is_success:
                previews = [repr(text[:80]) for text in chunk[:3]]
                raise RuntimeError(
                    f"TEI /embed failed with HTTP {response.status_code} for inputs "
                    f"{start}..{start + len(chunk) - 1} ({', '.join(previews)}): "
                    f"{response.text[:500]}"
                )
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

    def encode_query(self, texts: Sequence[str], batch_size: int) -> np.ndarray:
        """TEI's /embed route uses the server's configured default prompt."""
        return self.encode(texts, batch_size)

    def encode_document(self, texts: Sequence[str], batch_size: int) -> np.ndarray:
        """TEI does not expose a per-request query/document selector on /embed."""
        return self.encode(texts, batch_size)

    def encode_concurrently(self, text: str, count: int) -> np.ndarray:
        """Send independent one-input requests at the same time.

        This exercises TEI router coalescing, which is distinct from sending one
        client request whose ``inputs`` field already contains a list.
        """
        if count < 2:
            raise ValueError("concurrent request count must be at least two")
        start = Barrier(count)

        def request_one(_: int) -> list[float]:
            start.wait()
            response = self._embed_request([text])
            if not response.is_success:
                raise RuntimeError(
                    f"TEI concurrent /embed failed with HTTP {response.status_code} "
                    f"for {text[:80]!r}: {response.text[:500]}"
                )
            payload = response.json()
            if not isinstance(payload, list):
                raise RuntimeError("TEI concurrent /embed returned a non-list response")
            if payload and isinstance(payload[0], (int, float)):
                payload = [payload]
            if len(payload) != 1:
                raise RuntimeError("TEI concurrent /embed returned an invalid output count")
            return payload[0]

        with ThreadPoolExecutor(max_workers=count) as executor:
            rows = list(executor.map(request_one, range(count)))
        try:
            vectors: np.ndarray = np.asarray(rows, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise RuntimeError("TEI concurrent /embed returned malformed embeddings") from exc
        if vectors.ndim != 2:
            raise RuntimeError("TEI concurrent /embed returned malformed embeddings")
        return vectors

    def close(self) -> None:
        if self._owns_client:
            self.client.close()

    def __enter__(self) -> TEIBackend:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
