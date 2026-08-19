from __future__ import annotations

import numpy as np

from embed_parity.compare import Thresholds, compare_backends
from embed_parity.probes import Probe, built_in_probes
from tests.fakes import FakeBackend

PROBES = built_in_probes()[:24]


def run(reference=None, candidate=None, **kwargs):
    return compare_backends(
        kwargs.pop("model_id", "test/model"),
        reference or FakeBackend(),
        candidate or FakeBackend(),
        kwargs.pop("probes", PROBES),
        batch_sizes=kwargs.pop("batch_sizes", (1, 8)),
        **kwargs,
    )


def codes(report):
    return {item["code"] for item in report["diagnostics"]}


def test_probe_corpus_is_deterministic_and_has_100_items():
    assert built_in_probes() == built_in_probes()
    assert len(built_in_probes()) == 100
    assert len({probe.id for probe in built_in_probes()}) == 100


def test_empty_probes_are_recorded_and_skipped_when_unsupported():
    candidate = FakeBackend()
    candidate.supports_empty_inputs = False
    report = run(candidate=candidate, probes=built_in_probes()[:8], batch_sizes=(1,))
    assert report["requested_probe_count"] == 8
    assert report["probe_count"] == 5
    assert len(report["skipped_probes"]) == 3
    assert report["passed"]


def test_exact_equivalent_embeddings_pass():
    report = run()
    assert report["passed"]
    assert report["vector"]["minimum"] > 0.999999
    assert report["neighbors"]["top_10_overlap"] == 1.0


def test_non_normalized_candidate_is_diagnosed():
    report = run(candidate=FakeBackend(normalize=False))
    assert not report["passed"]
    assert not report["structural"]["norm_compatible"]
    assert "normalization_mismatch" in codes(report)


def test_dimension_mismatch_fails_clearly():
    report = run(candidate=FakeBackend(dimension=16))
    assert not report["passed"]
    assert report["vector"] is None
    assert "dimension_mismatch" in codes(report)


def test_deterministic_small_noise_passes():
    assert run(candidate=FakeBackend(noise=1e-7))["passed"]


def test_large_vector_perturbation_fails():
    report = run(candidate=FakeBackend(perturb=1.5))
    assert not report["passed"]
    assert "embedding_space_mismatch" in codes(report)


def test_candidate_truncation_boundary_is_minimized():
    def factory(n):
        return " ".join(f"token{i}" for i in range(n))

    report = run(
        candidate=FakeBackend(truncate=256),
        length_factory=factory,
        lengths=(32, 128, 256, 257, 384, 512),
    )
    assert "length_boundary" in codes(report)
    minimal = report["length_analysis"]["minimal_reproducer"]
    assert minimal["tokens"] == 257
    assert minimal["previous_cosine"] > 0.999


def test_batch_dependent_candidate_is_diagnosed():
    report = run(candidate=FakeBackend(batch_sensitive=True))
    assert "batch_sensitive_candidate" in codes(report)
    assert not report["batch_consistency"]["candidate"]["passed"]


def test_concurrent_server_batching_passes_when_embeddings_are_stable():
    report = run(concurrent_requests=4, concurrency_trials=2)
    server_batch = report["server_batch_consistency"]
    assert server_batch["passed"]
    assert server_batch["responses_compared"] == 8
    assert server_batch["minimum_cosine"] > 0.999999


def test_concurrent_server_batching_failure_is_diagnosed():
    report = run(
        candidate=FakeBackend(concurrency_sensitive=True),
        concurrent_requests=4,
        concurrency_trials=2,
    )
    server_batch = report["server_batch_consistency"]
    assert not report["passed"]
    assert not server_batch["passed"]
    assert server_batch["divergent_responses"] == 4
    assert "server_batch_sensitive_candidate" in codes(report)


def test_backend_without_concurrency_capability_is_skipped_cleanly():
    class EncodeOnlyBackend:
        name = "encode-only"
        metadata = {"runtime": "encode-only"}

        def encode(self, texts, batch_size):
            return FakeBackend().encode(texts, batch_size)

    report = run(candidate=EncodeOnlyBackend(), concurrent_requests=4)
    assert report["server_batch_consistency"] is None
    assert report["passed"]


def test_query_prefix_difference_is_diagnosed():
    queries = [
        Probe(f"q{i}", "query", text)
        for i, text in enumerate(("best tea", "reset password", "nearby trains"))
    ]
    prefix = "query: "
    report = run(
        model_id="intfloat/e5-small-v2",
        probes=queries,
        candidate=FakeBackend(prefix=prefix),
        batch_sizes=(1,),
    )
    assert "query_prefix_mismatch" in codes(report)
    assert (
        report["prefix_analysis"]["query"]["scores"]["reference_prefixed_to_candidate_raw"] > 0.999
    )


def test_document_prefix_difference_is_diagnosed():
    documents = [
        Probe(f"d{i}", "document", text)
        for i, text in enumerate(("tea guide", "password instructions", "train timetable"))
    ]
    report = run(
        model_id="intfloat/e5-small-v2",
        probes=documents,
        candidate=FakeBackend(prefix="passage: "),
        batch_sizes=(1,),
    )
    assert "query_prefix_mismatch" in codes(report)
    assert report["prefix_analysis"]["document"]["possible_mismatch"]


def test_nan_output_fails_structural_checks():
    report = run(candidate=FakeBackend(nan=True))
    assert not report["passed"]
    assert "non_finite" in codes(report)


def test_completely_incompatible_space_fails_geometry_and_vectors():
    report = run(candidate=FakeBackend(salt="other-space"))
    assert not report["passed"]
    assert report["geometry"]["spearman"] < 0.5
    assert "embedding_space_mismatch" in codes(report)


def test_thresholds_are_configurable():
    strict = Thresholds(vector_min=1.0)
    assert not run(candidate=FakeBackend(noise=1e-7), thresholds=strict)["passed"]


def test_invalid_threshold_is_rejected():
    with np.testing.assert_raises_regex(ValueError, "top5_overlap"):
        Thresholds(top5_overlap=1.01)


def test_known_revision_mismatch_fails_even_when_vectors_match():
    reference = FakeBackend(metadata={"model_id": "org/model", "resolved_revision": "abcdef012345"})
    candidate = FakeBackend(metadata={"model_id": "org/model", "resolved_revision": "999999999999"})
    report = run(reference=reference, candidate=candidate, model_id="org/model")
    assert not report["passed"]
    assert "revision_mismatch" in codes(report)


def test_short_and_full_revision_hashes_match():
    reference = FakeBackend(metadata={"model_id": "org/model", "resolved_revision": "abcdef012345"})
    candidate = FakeBackend(metadata={"model_id": "org/model", "resolved_revision": "abcdef0"})
    report = run(reference=reference, candidate=candidate, model_id="org/model")
    assert report["metadata_compatibility"]["revision_match"] is True


def test_dtype_difference_is_observational_not_an_automatic_failure():
    reference = FakeBackend(metadata={"dtype": "float32"})
    candidate = FakeBackend(metadata={"dtype": "float16"})
    report = run(reference=reference, candidate=candidate)
    assert report["passed"]
    assert "dtype_difference" in codes(report)


def test_known_pooling_mismatch_is_diagnosed_and_fails():
    reference = FakeBackend(metadata={"pooling": "mean"})
    candidate = FakeBackend(metadata={"pooling": "cls"})
    report = run(reference=reference, candidate=candidate)
    assert not report["passed"]
    assert "pooling_mismatch" in codes(report)
