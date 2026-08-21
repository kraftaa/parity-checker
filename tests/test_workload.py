from __future__ import annotations

import copy
import json
import math
from collections.abc import Sequence

import numpy as np
import pytest

from embed_parity.workload import (
    WorkloadRecord,
    add_baseline_comparison,
    compare_workload,
    load_workload_jsonl,
    sample_workload,
)
from embed_parity.workload_report import render_workload_html, render_workload_text
from tests.fakes import FakeBackend


def records() -> list[WorkloadRecord]:
    return [
        WorkloadRecord("q-1", "query", "postgres lock timeout"),
        WorkloadRecord("q-2", "query", "redis cache eviction"),
        *[
            WorkloadRecord(f"d-{index}", "document", f"document text {index}")
            for index in range(1, 13)
        ],
    ]


def test_load_workload_rejects_unknown_roles(tmp_path):
    path = tmp_path / "workload.jsonl"
    path.write_text('{"id":"x","type":"passage","text":"text"}\n')
    with pytest.raises(ValueError, match="unknown type.*query or document"):
        load_workload_jsonl(path)


def test_load_workload_rejects_duplicate_ids(tmp_path):
    path = tmp_path / "workload.jsonl"
    path.write_text(
        "\n".join(
            (
                '{"id":"same","type":"query","text":"query"}',
                '{"id":"same","type":"document","text":"document"}',
            )
        )
    )
    with pytest.raises(ValueError, match="duplicate workload id"):
        load_workload_jsonl(path)


def test_sampling_is_deterministic_and_reported():
    first, details = sample_workload(records(), max_documents=4, max_queries=1, seed=7)
    second, second_details = sample_workload(records(), max_documents=4, max_queries=1, seed=7)
    assert first == second
    assert details == second_details
    assert details["sampling_applied"]
    assert details["documents_available"] == 12
    assert details["documents_sampled"] == 4
    assert details["queries_sampled"] == 1


def test_matching_workload_has_identical_retrieval():
    report = compare_workload(
        "test/model",
        FakeBackend(),
        FakeBackend(),
        records(),
        batch_size=4,
        max_documents=None,
        max_queries=None,
    )
    assert report["passed"]
    assert report["report_type"] == "workload_parity"
    assert report["retrieval"]["summary"]["top_1_agreement"] == 1.0
    assert report["retrieval"]["summary"]["top_5_overlap"] == 1.0
    assert report["retrieval"]["summary"]["rbo_at_10"] == pytest.approx(1.0)
    assert report["retrieval"]["most_affected"] == []
    assert len(report["vector"]["records"]) == len(records())


class AngleBackend:
    name = "angle"

    def __init__(self, query_angle: float) -> None:
        self.query_angle = query_angle

    @property
    def metadata(self):
        return {"runtime": self.name, "embedding_dimension": 2}

    @staticmethod
    def _vector(angle: float) -> np.ndarray:
        return np.asarray([math.cos(angle), math.sin(angle)], dtype=np.float64)

    def encode_query(self, texts: Sequence[str], batch_size: int) -> np.ndarray:
        return np.stack([self._vector(self.query_angle) for _ in texts])

    def encode_document(self, texts: Sequence[str], batch_size: int) -> np.ndarray:
        angles = (-0.05, -0.02, 0.0, 0.02, 0.05, 0.08)
        return np.stack([self._vector(angles[int(text)]) for text in texts])

    def encode(self, texts: Sequence[str], batch_size: int) -> np.ndarray:
        raise AssertionError("workload analysis must use role-aware encoding")


def test_structurally_valid_vectors_can_change_retrieval():
    workload = [
        WorkloadRecord("q", "query", "dense neighborhood"),
        *[WorkloadRecord(f"d-{index}", "document", str(index)) for index in range(6)],
    ]
    report = compare_workload(
        "test/model",
        AngleBackend(0.0),
        AngleBackend(0.10),
        workload,
        max_documents=None,
        max_queries=None,
    )
    query = report["retrieval"]["queries"][0]
    assert query["vector_cosine"] > 0.99
    assert query["structurally_valid"]
    assert query["top_5_overlap"] == 0.8
    assert query["silent_risk"]
    assert not report["passed"]
    assert "STRUCTURALLY VALID, RETRIEVAL CHANGED" in render_workload_text(report)


def test_baseline_reports_percentage_point_change():
    baseline = compare_workload(
        "test/model", FakeBackend(), FakeBackend(), records(), max_documents=None, max_queries=None
    )
    current = copy.deepcopy(baseline)
    current["retrieval"]["summary"]["top_1_agreement"] = 0.75
    add_baseline_comparison(current, baseline)
    comparison = current["baseline_comparison"]
    assert comparison["metrics"]["top_1_agreement"]["change_percentage_points"] == -25.0
    assert comparison["retrieval_parity_regression"]


def test_static_html_contains_summary_and_escapes_workload_text():
    workload = records()
    workload[0] = WorkloadRecord("q-1", "query", "<script>alert(1)</script>")
    report = compare_workload(
        "test/model",
        FakeBackend(),
        FakeBackend(),
        workload,
        max_documents=None,
        max_queries=None,
    )
    artifact = render_workload_html(report)
    assert "Top-1 agreement" in artifact
    assert "Runtime and model metadata" in artifact
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in artifact
    assert "<script>alert(1)</script>" not in artifact
    assert "<script src=" not in artifact


def test_report_is_json_serializable_after_non_finite_sanitization(tmp_path):
    report = compare_workload(
        "test/model", FakeBackend(), FakeBackend(), records(), max_documents=None, max_queries=None
    )
    path = tmp_path / "report.json"
    from embed_parity.report import write_json

    write_json(report, path)
    payload = json.loads(path.read_text())
    assert payload["schema_version"] == 4
