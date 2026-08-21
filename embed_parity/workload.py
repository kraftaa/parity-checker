from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np

from . import __version__
from .backends.base import EmbeddingBackend, RetrievalEmbeddingBackend
from .compare import DEFAULT_THRESHOLDS, Thresholds

WorkloadRole = Literal["query", "document"]


@dataclass(frozen=True)
class WorkloadRecord:
    id: str
    type: WorkloadRole
    text: str


def load_workload_jsonl(path: str | Path) -> list[WorkloadRecord]:
    records: list[WorkloadRecord] = []
    seen: set[str] = set()
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on line {line_number}: {exc.msg}") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"line {line_number} must be a JSON object")
            record_id = payload.get("id")
            role = payload.get("type")
            text = payload.get("text")
            if not isinstance(record_id, str) or not record_id.strip():
                raise ValueError(f"line {line_number} has an invalid id")
            if record_id in seen:
                raise ValueError(f"duplicate workload id {record_id!r} on line {line_number}")
            if role not in ("query", "document"):
                raise ValueError(
                    f"line {line_number} has unknown type {role!r}; expected query or document"
                )
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"line {line_number} has empty or invalid text")
            seen.add(record_id)
            records.append(WorkloadRecord(record_id, cast(WorkloadRole, role), text))
    if not records:
        raise ValueError("workload input contains no records")
    if not any(record.type == "query" for record in records):
        raise ValueError("workload input must contain at least one query")
    if not any(record.type == "document" for record in records):
        raise ValueError("workload input must contain at least one document")
    return records


def _fingerprint(records: Sequence[WorkloadRecord]) -> str:
    canonical = json.dumps(
        [[record.id, record.type, record.text] for record in records],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode()
    return "sha256:" + hashlib.sha256(canonical).hexdigest()


def sample_workload(
    records: Sequence[WorkloadRecord],
    *,
    max_documents: int | None,
    max_queries: int | None,
    seed: int,
) -> tuple[list[WorkloadRecord], dict[str, Any]]:
    if max_documents is not None and max_documents < 1:
        raise ValueError("max_documents must be positive")
    if max_queries is not None and max_queries < 1:
        raise ValueError("max_queries must be positive")
    by_role = {
        role: [record for record in records if record.type == role]
        for role in ("query", "document")
    }
    rng = random.Random(seed)

    def choose(items: list[WorkloadRecord], maximum: int | None) -> list[WorkloadRecord]:
        if maximum is None or len(items) <= maximum:
            return items
        indices = sorted(rng.sample(range(len(items)), maximum))
        return [items[index] for index in indices]

    queries = choose(by_role["query"], max_queries)
    documents = choose(by_role["document"], max_documents)
    sampled_ids = {record.id for record in [*queries, *documents]}
    sampled = [record for record in records if record.id in sampled_ids]
    details = {
        "queries_available": len(by_role["query"]),
        "queries_sampled": len(queries),
        "documents_available": len(by_role["document"]),
        "documents_sampled": len(documents),
        "sampling_applied": len(queries) != len(by_role["query"])
        or len(documents) != len(by_role["document"]),
        "seed": seed,
        "source_fingerprint": _fingerprint(records),
        "sample_fingerprint": _fingerprint(sampled),
    }
    return sampled, details


def _encode_role(
    backend: EmbeddingBackend,
    role: WorkloadRole,
    texts: Sequence[str],
    batch_size: int,
) -> np.ndarray:
    method = getattr(backend, f"encode_{role}", None)
    if callable(method):
        role_backend = cast(RetrievalEmbeddingBackend, backend)
        role_method = role_backend.encode_query if role == "query" else role_backend.encode_document
        return np.asarray(role_method(texts, batch_size), dtype=np.float64)
    return np.asarray(backend.encode(texts, batch_size), dtype=np.float64)


def _structure(vectors: np.ndarray, expected_count: int) -> dict[str, Any]:
    matrix = vectors.ndim == 2
    count = int(vectors.shape[0]) if vectors.ndim else 0
    dimension = int(vectors.shape[1]) if matrix else None
    finite = bool(matrix and np.isfinite(vectors).all())
    norms = np.linalg.norm(vectors, axis=1) if matrix else np.array([], dtype=np.float64)
    finite_norms = norms[np.isfinite(norms)]
    return {
        "shape": list(vectors.shape),
        "count": count,
        "expected_count": expected_count,
        "count_matches": count == expected_count,
        "dimension": dimension,
        "all_finite": finite,
        "nan_count": int(np.isnan(vectors).sum()),
        "inf_count": int(np.isinf(vectors).sum()),
        "zero_vectors": int(np.sum(norms == 0)),
        "norm_mean": float(np.mean(finite_norms)) if len(finite_norms) else float("nan"),
    }


def _cosine_rows(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    numerator = np.sum(left * right, axis=1)
    denominator = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    return np.divide(
        numerator,
        denominator,
        out=np.full(numerator.shape, np.nan, dtype=np.float64),
        where=denominator != 0,
    )


def _finite_summary(values: Sequence[float]) -> dict[str, float]:
    array: np.ndarray = np.asarray(values, dtype=np.float64)
    finite: np.ndarray = array[np.isfinite(array)]
    if not len(finite):
        return {key: float("nan") for key in ("mean", "median", "minimum", "p05", "p95")}
    return {
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "minimum": float(np.min(finite)),
        "p05": float(np.percentile(finite, 5)),
        "p95": float(np.percentile(finite, 95)),
    }


def _histogram(values: Sequence[float], edges: Sequence[float]) -> list[dict[str, Any]]:
    finite: np.ndarray = np.asarray(
        [value for value in values if np.isfinite(value)], dtype=np.float64
    )
    counts, actual_edges = np.histogram(finite, bins=np.asarray(edges, dtype=np.float64))
    return [
        {
            "lower": float(actual_edges[index]),
            "upper": float(actual_edges[index + 1]),
            "count": int(count),
        }
        for index, count in enumerate(counts)
    ]


def _vector_rows(
    records: Sequence[WorkloadRecord], reference: np.ndarray, candidate: np.ndarray
) -> list[dict[str, Any]]:
    ref_dimension = reference.shape[1] if reference.ndim == 2 else None
    can_dimension = candidate.shape[1] if candidate.ndim == 2 else None
    dimensions_match = ref_dimension is not None and ref_dimension == can_dimension
    count = min(len(records), len(reference), len(candidate))
    cosines = (
        _cosine_rows(reference[:count], candidate[:count])
        if dimensions_match
        else np.full(count, np.nan)
    )
    rows: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        present = index < count
        ref_finite = bool(present and np.isfinite(reference[index]).all())
        can_finite = bool(present and np.isfinite(candidate[index]).all())
        ref_norm = float(np.linalg.norm(reference[index])) if present else float("nan")
        can_norm = float(np.linalg.norm(candidate[index])) if present else float("nan")
        norm_difference = abs(ref_norm - can_norm)
        rows.append(
            {
                "id": record.id,
                "type": record.type,
                "text": record.text,
                "cosine": float(cosines[index]) if present else float("nan"),
                "reference_norm": ref_norm,
                "candidate_norm": can_norm,
                "norm_difference": norm_difference,
                "relative_norm_difference": norm_difference / max(abs(ref_norm), 1e-12),
                "dimensions_match": bool(dimensions_match),
                "reference_dimension": ref_dimension,
                "candidate_dimension": can_dimension,
                "reference_finite": ref_finite,
                "candidate_finite": can_finite,
            }
        )
    return rows


def _normalized(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return np.divide(
        vectors,
        norms,
        out=np.full(vectors.shape, np.nan, dtype=np.float64),
        where=norms != 0,
    )


def _rank(scores: np.ndarray, document_ids: Sequence[str], k: int) -> np.ndarray:
    safe = np.nan_to_num(scores, nan=-np.inf)
    identifiers = np.asarray(document_ids, dtype=str)
    return np.lexsort((identifiers, -safe))[:k]


def _rbo(left: Sequence[str], right: Sequence[str], persistence: float = 0.9) -> float:
    depth = min(len(left), len(right))
    if depth == 0:
        return 1.0
    score = 0.0
    left_seen: set[str] = set()
    right_seen: set[str] = set()
    agreement = 0.0
    for index in range(depth):
        left_seen.add(left[index])
        right_seen.add(right[index])
        agreement = len(left_seen & right_seen) / (index + 1)
        score += (1.0 - persistence) * agreement * persistence**index
    return float(score + agreement * persistence**depth)


def _retrieval(
    query_records: Sequence[WorkloadRecord],
    document_records: Sequence[WorkloadRecord],
    ref_queries: np.ndarray,
    ref_documents: np.ndarray,
    can_queries: np.ndarray,
    can_documents: np.ndarray,
    query_vectors: dict[str, dict[str, Any]],
    thresholds: Thresholds,
) -> dict[str, Any] | None:
    usable = (
        ref_queries.ndim == ref_documents.ndim == can_queries.ndim == can_documents.ndim == 2
        and ref_queries.shape[1] == ref_documents.shape[1]
        and can_queries.shape[1] == can_documents.shape[1]
        and ref_queries.shape[0] == can_queries.shape[0] == len(query_records)
        and ref_documents.shape[0] == can_documents.shape[0] == len(document_records)
        and np.isfinite(ref_queries).all()
        and np.isfinite(ref_documents).all()
        and np.isfinite(can_queries).all()
        and np.isfinite(can_documents).all()
        and all(
            np.all(np.linalg.norm(vectors, axis=1) > 0)
            for vectors in (ref_queries, ref_documents, can_queries, can_documents)
        )
    )
    if not usable:
        return None
    ref_queries = _normalized(ref_queries)
    ref_documents = _normalized(ref_documents)
    can_queries = _normalized(can_queries)
    can_documents = _normalized(can_documents)
    document_ids = [record.id for record in document_records]
    max_k = min(10, len(document_records))
    rows: list[dict[str, Any]] = []
    for query_index, query in enumerate(query_records):
        ref_scores = ref_documents @ ref_queries[query_index]
        can_scores = can_documents @ can_queries[query_index]
        ref_order = _rank(ref_scores, document_ids, max_k)
        can_order = _rank(can_scores, document_ids, max_k)
        ref_ids = [document_ids[index] for index in ref_order]
        can_ids = [document_ids[index] for index in can_order]
        metrics: dict[str, float | bool] = {}
        for requested_k in (1, 5, 10):
            actual_k = min(requested_k, max_k)
            left = set(ref_ids[:actual_k])
            right = set(can_ids[:actual_k])
            intersection = len(left & right)
            union = len(left | right)
            overlap = intersection / actual_k if actual_k else 1.0
            metrics[f"top_{requested_k}_overlap"] = overlap
            metrics[f"jaccard_at_{requested_k}"] = intersection / union if union else 1.0
        metrics["top_1_agreement"] = bool(ref_ids[:1] == can_ids[:1])
        metrics["rbo_at_10"] = _rbo(ref_ids, can_ids)
        missing_rank = max_k + 1
        ref_positions = {identifier: rank + 1 for rank, identifier in enumerate(ref_ids)}
        can_positions = {identifier: rank + 1 for rank, identifier in enumerate(can_ids)}
        union_ids = set(ref_positions) | set(can_positions)
        largest_movement = max(
            (
                abs(
                    ref_positions.get(identifier, missing_rank)
                    - can_positions.get(identifier, missing_rank)
                )
                for identifier in union_ids
            ),
            default=0,
        )
        vector = query_vectors[query.id]
        structurally_valid = bool(
            vector["dimensions_match"]
            and vector["reference_finite"]
            and vector["candidate_finite"]
            and vector["relative_norm_difference"] <= thresholds.norm_relative_difference
        )
        silent_risk = bool(
            structurally_valid and metrics["top_5_overlap"] < thresholds.top5_overlap
        )
        rows.append(
            {
                "id": query.id,
                "text": query.text,
                "vector_cosine": vector["cosine"],
                **metrics,
                "largest_rank_movement": largest_movement,
                "structurally_valid": structurally_valid,
                "silent_risk": silent_risk,
                "reference_top": [
                    {
                        "rank": rank + 1,
                        "id": document_ids[index],
                        "score": float(ref_scores[index]),
                    }
                    for rank, index in enumerate(ref_order)
                ],
                "candidate_top": [
                    {
                        "rank": rank + 1,
                        "id": document_ids[index],
                        "score": float(can_scores[index]),
                    }
                    for rank, index in enumerate(can_order)
                ],
            }
        )
    summary = {
        "query_count": len(query_records),
        "document_count": len(document_records),
        "top_1_agreement": float(np.mean([row["top_1_agreement"] for row in rows])),
        "top_5_overlap": float(np.mean([row["top_5_overlap"] for row in rows])),
        "top_10_overlap": float(np.mean([row["top_10_overlap"] for row in rows])),
        "jaccard_at_5": float(np.mean([row["jaccard_at_5"] for row in rows])),
        "jaccard_at_10": float(np.mean([row["jaccard_at_10"] for row in rows])),
        "rbo_at_10": float(np.mean([row["rbo_at_10"] for row in rows])),
    }
    changed_rows = [
        row
        for row in rows
        if (
            not row["top_1_agreement"]
            or row["top_5_overlap"] < 1.0
            or row["top_10_overlap"] < 1.0
            or not np.isfinite(row["vector_cosine"])
            or row["vector_cosine"] < thresholds.vector_min
        )
    ]
    most_affected = sorted(
        changed_rows,
        key=lambda row: (
            row["top_5_overlap"],
            -row["largest_rank_movement"],
            np.nan_to_num(row["vector_cosine"], nan=-np.inf),
            row["id"],
        ),
    )[:10]
    return {
        "summary": summary,
        "queries": rows,
        "most_affected": most_affected,
        "silent_risk_ids": [row["id"] for row in rows if row["silent_risk"]],
        "overlap_distribution": _histogram(
            [float(row["top_5_overlap"]) for row in rows],
            (-0.001, 0.2, 0.4, 0.6, 0.8, 0.999999, 1.001),
        ),
    }


def compare_workload(
    model_id: str,
    reference: EmbeddingBackend,
    candidate: EmbeddingBackend,
    records: Sequence[WorkloadRecord],
    *,
    batch_size: int = 32,
    max_documents: int | None = 10_000,
    max_queries: int | None = 1_000,
    seed: int = 42,
    thresholds: Thresholds = DEFAULT_THRESHOLDS,
) -> dict[str, Any]:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    sampled, input_details = sample_workload(
        records, max_documents=max_documents, max_queries=max_queries, seed=seed
    )
    queries = [record for record in sampled if record.type == "query"]
    documents = [record for record in sampled if record.type == "document"]
    ref_queries = _encode_role(reference, "query", [record.text for record in queries], batch_size)
    ref_documents = _encode_role(
        reference, "document", [record.text for record in documents], batch_size
    )
    can_queries = _encode_role(candidate, "query", [record.text for record in queries], batch_size)
    can_documents = _encode_role(
        candidate, "document", [record.text for record in documents], batch_size
    )
    structures = {
        "query": {
            "reference": _structure(ref_queries, len(queries)),
            "candidate": _structure(can_queries, len(queries)),
        },
        "document": {
            "reference": _structure(ref_documents, len(documents)),
            "candidate": _structure(can_documents, len(documents)),
        },
    }
    query_rows = _vector_rows(queries, ref_queries, can_queries)
    document_rows = _vector_rows(documents, ref_documents, can_documents)
    vector_rows = [*query_rows, *document_rows]
    cosines = [float(row["cosine"]) for row in vector_rows]
    relative_norms = [float(row["relative_norm_difference"]) for row in vector_rows]
    vector_summary: dict[str, Any] = {
        **_finite_summary(cosines),
        "norm_difference": _finite_summary(relative_norms),
        "dimensions_match": all(row["dimensions_match"] for row in vector_rows),
        "all_finite": all(
            row["reference_finite"] and row["candidate_finite"] for row in vector_rows
        ),
    }
    worst_vectors = sorted(
        vector_rows,
        key=lambda row: (np.nan_to_num(row["cosine"], nan=-np.inf), row["id"]),
    )[:10]
    query_by_id = {row["id"]: row for row in query_rows}
    retrieval = _retrieval(
        queries,
        documents,
        ref_queries,
        ref_documents,
        can_queries,
        can_documents,
        query_by_id,
        thresholds,
    )
    structure_passed = bool(
        all(
            value[side]["count_matches"]
            and value[side]["all_finite"]
            and value[side]["zero_vectors"] == 0
            for value in structures.values()
            for side in ("reference", "candidate")
        )
        and vector_summary["dimensions_match"]
        and vector_summary["all_finite"]
        and vector_summary["norm_difference"]["p95"] <= thresholds.norm_relative_difference
    )
    retrieval_passed = bool(
        retrieval
        and retrieval["summary"]["top_1_agreement"] >= thresholds.top1_agreement
        and retrieval["summary"]["top_5_overlap"] >= thresholds.top5_overlap
        and retrieval["summary"]["top_10_overlap"] >= thresholds.top10_overlap
    )
    vector_passed = bool(
        np.isfinite(vector_summary["mean"])
        and np.isfinite(vector_summary["minimum"])
        and vector_summary["mean"] >= thresholds.vector_mean
        and vector_summary["minimum"] >= thresholds.vector_min
    )
    return {
        "schema_version": 4,
        "report_type": "workload_parity",
        "tool_version": __version__,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": model_id,
        "reference": reference.metadata,
        "candidate": candidate.metadata,
        "role_encoding": {
            "reference": "encode_query / encode_document when supported",
            "candidate": "TEI /embed using the server's configured default prompt",
        },
        "configuration": {
            "batch_size": batch_size,
            "max_documents": max_documents,
            "max_queries": max_queries,
            "seed": seed,
        },
        "thresholds": asdict(thresholds),
        "input": input_details,
        "structural": {**structures, "passed": structure_passed},
        "vector": {
            "summary": vector_summary,
            "worst": worst_vectors,
            "distribution": _histogram(cosines, (-1.001, 0.5, 0.8, 0.9, 0.95, 0.99, 0.999, 1.001)),
            "records": vector_rows,
            "passed": vector_passed,
        },
        "retrieval": retrieval,
        "baseline_comparison": None,
        "passed": bool(structure_passed and vector_passed and retrieval_passed),
        "interpretation": (
            "This report measures behavioral parity, not retrieval quality; no relevance labels "
            "were supplied."
        ),
    }


def add_baseline_comparison(report: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    if baseline.get("report_type") != "workload_parity":
        raise ValueError("baseline is not an embed-parity workload report")
    if baseline.get("model") != report.get("model"):
        raise ValueError("baseline model does not match the current model")
    if baseline.get("input", {}).get("sample_fingerprint") != report.get("input", {}).get(
        "sample_fingerprint"
    ):
        raise ValueError("baseline workload sample does not match the current sample")
    baseline_summary = baseline.get("retrieval", {}).get("summary")
    current_summary = report.get("retrieval", {}).get("summary")
    if not isinstance(baseline_summary, dict) or not isinstance(current_summary, dict):
        raise ValueError("baseline and current reports must contain retrieval metrics")
    metrics: dict[str, dict[str, float]] = {}
    for key in ("top_1_agreement", "top_5_overlap", "top_10_overlap"):
        before = float(baseline_summary[key])
        current = float(current_summary[key])
        metrics[key] = {
            "baseline": before,
            "current": current,
            "change_percentage_points": (current - before) * 100.0,
        }
    report["baseline_comparison"] = {
        "baseline_generated_at": baseline.get("generated_at"),
        "metrics": metrics,
        "retrieval_parity_regression": any(
            item["change_percentage_points"] < -1e-9 for item in metrics.values()
        ),
        "interpretation": (
            "A negative change is a retrieval-parity regression, not evidence of lower "
            "search quality."
        ),
    }
    return report
