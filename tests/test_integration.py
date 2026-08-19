"""Opt-in tests for live TEI deployments.

Set each environment variable to a TEI URL serving the matching model, then run
`pytest -m integration`. These are skipped in ordinary unit-test runs.
"""

from __future__ import annotations

import os

import pytest

from embed_parity.backends.sentence_transformers import SentenceTransformersBackend
from embed_parity.backends.tei import TEIBackend
from embed_parity.compare import compare_backends
from embed_parity.probes import built_in_probes

MODELS = [
    (
        "BAAI/bge-small-en-v1.5",
        "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
        "TEI_BGE_SMALL_URL",
    ),
    (
        "intfloat/e5-small-v2",
        "ffb93f3bd4047442299a41ebb6fa998a38507c52",
        "TEI_E5_SMALL_URL",
    ),
    (
        "sentence-transformers/all-MiniLM-L6-v2",
        "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
        "TEI_MINILM_URL",
    ),
]


@pytest.mark.integration
@pytest.mark.parametrize(("model_id", "revision", "url_variable"), MODELS)
def test_live_runtime_parity(model_id: str, revision: str, url_variable: str):
    url = os.environ.get(url_variable)
    if not url:
        pytest.skip(f"{url_variable} is not set")
    reference = SentenceTransformersBackend(model_id, revision=revision)
    report = compare_backends(
        model_id,
        reference,
        TEIBackend(url),
        built_in_probes(),
        batch_sizes=(1, 8, 32),
        length_factory=reference.text_at_token_length,
    )
    assert report["passed"], report["diagnostics"]
