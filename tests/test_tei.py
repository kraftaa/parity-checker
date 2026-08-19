from __future__ import annotations

import json

import httpx
import numpy as np
import pytest

from embed_parity.backends.tei import TEIBackend


def test_tei_metadata_promotion_and_request_batching():
    request_sizes = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET" and request.url.path == "/info":
            return httpx.Response(
                200,
                json={
                    "model_id": "org/model",
                    "model_sha": "abcdef012345",
                    "dtype": "float16",
                    "model_type": {"embedding": {"pooling": "mean"}},
                    "max_input_length": 512,
                    "dimension": 3,
                },
            )
        payload = json.loads(request.content)
        assert payload["truncate"] is True
        request_sizes.append(len(payload["inputs"]))
        return httpx.Response(
            200, json=[[float(len(text)), 1.0, 2.0] for text in payload["inputs"]]
        )

    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://tei.test")
    backend = TEIBackend("http://tei.test/", client=client)
    vectors = backend.encode(["a", "bb", "ccc", "dddd", "eeeee"], batch_size=2)
    assert request_sizes == [2, 2, 1]
    assert vectors.shape == (5, 3)
    assert np.array_equal(vectors[:, 0], [1, 2, 3, 4, 5])
    assert backend.metadata["resolved_revision"] == "abcdef012345"
    assert backend.metadata["tokenizer_max_length"] == 512
    assert backend.metadata["pooling"] == "mean"


def test_tei_rejects_wrong_output_count():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(404)
        return httpx.Response(200, json=[[1.0, 2.0]])

    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://tei.test")
    backend = TEIBackend("http://tei.test", client=client)
    with pytest.raises(RuntimeError, match="1 vectors for 2 inputs"):
        backend.encode(["a", "b"], batch_size=2)


def test_tei_rejects_ragged_embeddings():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(404)
        return httpx.Response(200, json=[[1.0], [1.0, 2.0]])

    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://tei.test")
    backend = TEIBackend("http://tei.test", client=client)
    with pytest.raises(RuntimeError, match="malformed or ragged"):
        backend.encode(["a", "b"], batch_size=2)


def test_tei_retries_transient_failures_and_controls_truncation():
    posts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal posts
        if request.method == "GET":
            return httpx.Response(404)
        posts += 1
        payload = json.loads(request.content)
        assert payload["truncate"] is False
        if posts == 1:
            return httpx.Response(503, text="warming up")
        return httpx.Response(200, json=[[1.0, 2.0]])

    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://tei.test")
    backend = TEIBackend(
        "http://tei.test", client=client, truncate=False, retries=1, retry_backoff=0
    )
    assert backend.encode(["hello"], batch_size=1).shape == (1, 2)
    assert posts == 2


def test_tei_wait_until_ready_refreshes_metadata():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200)
        if request.url.path == "/info":
            return httpx.Response(200, json={"model_id": "org/model"})
        return httpx.Response(404)

    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://tei.test")
    backend = TEIBackend("http://tei.test", client=client)
    backend.wait_until_ready(1)
    assert backend.metadata["model_id"] == "org/model"
