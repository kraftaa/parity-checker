from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Callable, Sequence

import numpy as np

from .backends.base import EmbeddingBackend
from .probes import Probe, recommended_query_prefix


@dataclass(frozen=True)
class Thresholds:
    vector_mean: float = 0.999
    vector_min: float = 0.990
    geometry_spearman: float = 0.995
    geometry_mae: float = 0.010
    top1_agreement: float = 0.95
    top5_overlap: float = 0.95
    top10_overlap: float = 0.95
    batch_min: float = 0.999
    norm_relative_difference: float = 0.05
    prefix_improvement: float = 0.02

    def __post_init__(self) -> None:
        bounded = {
            "vector_mean": (self.vector_mean, -1.0, 1.0),
            "vector_min": (self.vector_min, -1.0, 1.0),
            "geometry_spearman": (self.geometry_spearman, -1.0, 1.0),
            "top1_agreement": (self.top1_agreement, 0.0, 1.0),
            "top5_overlap": (self.top5_overlap, 0.0, 1.0),
            "top10_overlap": (self.top10_overlap, 0.0, 1.0),
            "batch_min": (self.batch_min, -1.0, 1.0),
            "geometry_mae": (self.geometry_mae, 0.0, 2.0),
            "norm_relative_difference": (self.norm_relative_difference, 0.0, float("inf")),
            "prefix_improvement": (self.prefix_improvement, 0.0, 2.0),
        }
        for name, (value, lower, upper) in bounded.items():
            if not np.isfinite(value) or not lower <= value <= upper:
                raise ValueError(f"{name} must be between {lower} and {upper}")


def _cosine_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    numerator = np.sum(a * b, axis=1)
    denominator = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    return np.divide(
        numerator,
        denominator,
        out=np.full(numerator.shape, np.nan, dtype=np.float64),
        where=denominator != 0,
    )


def _cosine_matrix(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1)
    denominator = np.outer(norms, norms)
    return np.divide(
        vectors @ vectors.T,
        denominator,
        out=np.full(denominator.shape, np.nan, dtype=np.float64),
        where=denominator != 0,
    )


def _rankdata(values: np.ndarray) -> np.ndarray:
    """Average ranks for ties, equivalent to scipy.stats.rankdata(method='average')."""
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0 + 1.0
        start = end
    return ranks


def _correlation(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _safe_stats(values: np.ndarray) -> dict[str, float]:
    finite = values[np.isfinite(values)]
    if not len(finite):
        return {key: float("nan") for key in ("mean", "median", "minimum", "p01", "p05")}
    return {
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "minimum": float(np.min(finite)),
        "p01": float(np.percentile(finite, 1)),
        "p05": float(np.percentile(finite, 5)),
    }


def _structure(vectors: np.ndarray) -> dict[str, Any]:
    valid_matrix = vectors.ndim == 2
    finite = bool(valid_matrix and np.isfinite(vectors).all())
    norms = np.linalg.norm(vectors, axis=1) if valid_matrix else np.array([])
    finite_norms = norms[np.isfinite(norms)]
    return {
        "shape": list(vectors.shape),
        "count": int(vectors.shape[0]) if vectors.ndim else 0,
        "dimension": int(vectors.shape[1]) if valid_matrix else None,
        "all_finite": finite,
        "nan_count": int(np.isnan(vectors).sum()),
        "inf_count": int(np.isinf(vectors).sum()),
        "zero_vectors": int(np.sum(norms == 0)),
        "norms": {
            "mean": float(np.mean(finite_norms)) if len(finite_norms) else float("nan"),
            "std": float(np.std(finite_norms)) if len(finite_norms) else float("nan"),
            "min": float(np.min(finite_norms)) if len(finite_norms) else float("nan"),
            "max": float(np.max(finite_norms)) if len(finite_norms) else float("nan"),
            "p05": float(np.percentile(finite_norms, 5)) if len(finite_norms) else float("nan"),
            "median": float(np.median(finite_norms)) if len(finite_norms) else float("nan"),
            "p95": float(np.percentile(finite_norms, 95)) if len(finite_norms) else float("nan"),
        },
    }


def _metadata_compatibility(
    expected_model: str,
    reference: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    ref_model = reference.get("model_id") or expected_model
    can_model = candidate.get("model_id")
    model_match = None if can_model is None else str(ref_model).casefold() == str(can_model).casefold()
    ref_revision = reference.get("resolved_revision")
    can_revision = candidate.get("resolved_revision")
    revision_match: bool | None = None
    if ref_revision and can_revision:
        left, right = str(ref_revision).casefold(), str(can_revision).casefold()
        # Git commit hashes are commonly exposed in short and full forms.
        revision_match = left == right or (min(len(left), len(right)) >= 7 and (left.startswith(right) or right.startswith(left)))
    ref_dtype = reference.get("dtype")
    can_dtype = candidate.get("dtype")
    dtype_match = None if ref_dtype is None or can_dtype is None else str(ref_dtype).casefold() == str(can_dtype).casefold()
    return {
        "reference_model_id": ref_model,
        "candidate_model_id": can_model,
        "model_id_match": model_match,
        "reference_revision": ref_revision,
        "candidate_revision": can_revision,
        "revision_match": revision_match,
        "reference_dtype": ref_dtype,
        "candidate_dtype": can_dtype,
        "dtype_match": dtype_match,
        "passed": model_match is not False and revision_match is not False,
    }


def _probe_fingerprint(probes: Sequence[Probe]) -> str:
    canonical = json.dumps(
        [[probe.id, probe.category, probe.text] for probe in probes],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(canonical).hexdigest()


def _nearest_neighbors(ref: np.ndarray, candidate: np.ndarray, probes: Sequence[Probe]) -> dict[str, Any]:
    ref_sim = _cosine_matrix(ref)
    can_sim = _cosine_matrix(candidate)
    np.fill_diagonal(ref_sim, -np.inf)
    np.fill_diagonal(can_sim, -np.inf)
    ref_order = np.argsort(-ref_sim, axis=1, kind="stable")
    can_order = np.argsort(-can_sim, axis=1, kind="stable")
    metrics: dict[str, float] = {}
    row_overlap_10: list[float] = []
    for k in (1, 5, 10):
        actual_k = min(k, max(0, len(probes) - 1))
        if actual_k == 0:
            overlap = np.ones(len(probes))
        else:
            overlap = np.array([
                len(set(ref_order[i, :actual_k]) & set(can_order[i, :actual_k])) / actual_k
                for i in range(len(probes))
            ])
        metrics[f"top_{k}_overlap"] = float(np.mean(overlap))
        if k == 10:
            row_overlap_10 = overlap.tolist()
    changed = sorted(range(len(probes)), key=lambda i: (row_overlap_10[i], probes[i].id))[:10]
    return {
        **metrics,
        "most_changed": [
            {"id": probes[i].id, "text": probes[i].text, "top_10_overlap": row_overlap_10[i]}
            for i in changed
        ],
    }


def _core_metrics(ref: np.ndarray, candidate: np.ndarray, probes: Sequence[Probe]) -> dict[str, Any]:
    cosines = _cosine_rows(ref, candidate)
    worst = np.argsort(np.nan_to_num(cosines, nan=-np.inf))[:10]
    ref_matrix = _cosine_matrix(ref)
    can_matrix = _cosine_matrix(candidate)
    triangle = np.triu_indices(len(probes), k=1)
    a, b = ref_matrix[triangle], can_matrix[triangle]
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    diff = np.abs(a - b)
    geometry = {
        "pearson": _correlation(a, b),
        "spearman": _correlation(_rankdata(a), _rankdata(b)) if len(a) else float("nan"),
        "mean_absolute_difference": float(np.mean(diff)) if len(diff) else float("nan"),
        "maximum_difference": float(np.max(diff)) if len(diff) else float("nan"),
        "pairs_compared": int(len(diff)),
    }
    return {
        "vector": {
            **_safe_stats(cosines),
            "worst": [
                {"id": probes[i].id, "category": probes[i].category, "text": probes[i].text, "cosine": float(cosines[i])}
                for i in worst
            ],
        },
        "geometry": geometry,
        "neighbors": _nearest_neighbors(ref, candidate, probes),
    }


def _batch_consistency(
    backend: EmbeddingBackend,
    texts: Sequence[str],
    baseline: np.ndarray,
    base_batch: int,
    batch_sizes: Sequence[int],
    threshold: float,
) -> dict[str, Any]:
    comparisons = []
    passed = True
    for batch in batch_sizes:
        if batch == base_batch:
            continue
        other = backend.encode(texts, batch)
        compatible = other.shape == baseline.shape and np.isfinite(other).all()
        cosines = _cosine_rows(baseline, other) if compatible else np.array([np.nan])
        mean = _safe_stats(cosines)["mean"]
        item_passed = bool(compatible and np.isfinite(mean) and mean >= threshold)
        passed &= item_passed
        comparisons.append({
            "batch_a": base_batch, "batch_b": batch, "mean_cosine": mean,
            "minimum_cosine": _safe_stats(cosines)["minimum"], "passed": item_passed,
        })
    return {"passed": passed, "comparisons": comparisons}


def _prefix_analysis(
    model_id: str,
    reference: EmbeddingBackend,
    candidate: EmbeddingBackend,
    probes: Sequence[Probe],
    batch_size: int,
    improvement_threshold: float,
) -> dict[str, Any] | None:
    prefix = recommended_query_prefix(model_id)
    query_probes = [p for p in probes if p.category == "query"]
    if not prefix or not query_probes:
        return None
    raw = [p.text for p in query_probes]
    prefixed = [prefix + p.text for p in query_probes]
    ref_raw = reference.encode(raw, batch_size)
    ref_prefixed = reference.encode(prefixed, batch_size)
    can_raw = candidate.encode(raw, batch_size)
    can_prefixed = candidate.encode(prefixed, batch_size)
    scores = {
        "raw_to_raw": float(np.nanmean(_cosine_rows(ref_raw, can_raw))),
        "reference_prefixed_to_candidate_raw": float(np.nanmean(_cosine_rows(ref_prefixed, can_raw))),
        "reference_raw_to_candidate_prefixed": float(np.nanmean(_cosine_rows(ref_raw, can_prefixed))),
        "prefixed_to_prefixed": float(np.nanmean(_cosine_rows(ref_prefixed, can_prefixed))),
    }
    best_cross = max(scores["reference_prefixed_to_candidate_raw"], scores["reference_raw_to_candidate_prefixed"])
    return {
        "prefix": prefix,
        "scores": scores,
        "cross_variant_improvement": best_cross - scores["raw_to_raw"],
        "possible_mismatch": best_cross - scores["raw_to_raw"] >= improvement_threshold,
    }


def compare_backends(
    model_id: str,
    reference: EmbeddingBackend,
    candidate: EmbeddingBackend,
    probes: Sequence[Probe],
    *,
    batch_sizes: Sequence[int] = (1, 8, 32),
    thresholds: Thresholds = Thresholds(),
    length_factory: Callable[[int], str] | None = None,
    lengths: Sequence[int] = (32, 64, 128, 256, 384, 512, 768, 1024),
) -> dict[str, Any]:
    if not batch_sizes or any(size < 1 for size in batch_sizes):
        raise ValueError("batch sizes must contain positive integers")
    if len(set(batch_sizes)) != len(batch_sizes):
        raise ValueError("batch sizes must be unique")
    if len(probes) < 2:
        raise ValueError("at least two probes are required")
    if length_factory and (not lengths or tuple(sorted(set(lengths))) != tuple(lengths) or any(length < 1 for length in lengths)):
        raise ValueError("lengths must be positive, unique, and increasing")
    texts = [probe.text for probe in probes]
    base_batch = batch_sizes[0]
    ref = np.asarray(reference.encode(texts, base_batch), dtype=np.float64)
    candidate_vectors = np.asarray(candidate.encode(texts, base_batch), dtype=np.float64)
    ref_structure = _structure(ref)
    can_structure = _structure(candidate_vectors)
    same_count = ref_structure["count"] == can_structure["count"] == len(probes)
    same_dimension = (
        ref_structure["dimension"] is not None
        and ref_structure["dimension"] == can_structure["dimension"]
    )
    ref_norm = ref_structure["norms"]["mean"]
    can_norm = can_structure["norms"]["mean"]
    norm_relative = abs(ref_norm - can_norm) / max(abs(ref_norm), 1e-12)
    norm_distribution_relative = {
        key: abs(ref_structure["norms"][key] - can_structure["norms"][key])
        / max(abs(ref_structure["norms"][key]), 1e-12)
        for key in ("p05", "median", "p95")
    }
    norm_max_relative = max([norm_relative, *norm_distribution_relative.values()])
    structural = {
        "reference": ref_structure,
        "candidate": can_structure,
        "same_output_count": same_count,
        "same_dimension": same_dimension,
        "finite_vectors": ref_structure["all_finite"] and can_structure["all_finite"],
        "norm_relative_difference": norm_relative,
        "norm_distribution_relative_difference": norm_distribution_relative,
        "norm_max_relative_difference": norm_max_relative,
        "norm_compatible": bool(np.isfinite(norm_max_relative) and norm_max_relative <= thresholds.norm_relative_difference),
    }
    reference_metadata = reference.metadata
    candidate_metadata = candidate.metadata
    metadata_compatibility = _metadata_compatibility(model_id, reference_metadata, candidate_metadata)
    report: dict[str, Any] = {
        "schema_version": 1,
        "model": model_id,
        "thresholds": asdict(thresholds),
        "reference": reference_metadata,
        "candidate": candidate_metadata,
        "metadata_compatibility": metadata_compatibility,
        "probe_count": len(probes),
        "probe_fingerprint": _probe_fingerprint(probes),
        "configuration": {
            "batch_sizes": list(batch_sizes),
            "length_analysis_enabled": length_factory is not None,
            "lengths": list(lengths) if length_factory else [],
        },
        "structural": structural,
        "vector": None,
        "geometry": None,
        "neighbors": None,
        "batch_consistency": None,
        "length_analysis": None,
        "prefix_analysis": None,
        "diagnostics": [],
        "passed": False,
    }
    structurally_usable = same_count and same_dimension and structural["finite_vectors"]
    if structurally_usable:
        report.update(_core_metrics(ref, candidate_vectors, probes))
        report["batch_consistency"] = {
            "reference": _batch_consistency(reference, texts, ref, base_batch, batch_sizes, thresholds.batch_min),
            "candidate": _batch_consistency(candidate, texts, candidate_vectors, base_batch, batch_sizes, thresholds.batch_min),
        }
        report["prefix_analysis"] = _prefix_analysis(
            model_id, reference, candidate, probes, base_batch, thresholds.prefix_improvement
        )
        if length_factory:
            report["length_analysis"] = analyze_lengths(
                reference, candidate, length_factory, lengths, base_batch, thresholds.vector_min
            )
    from .diagnostics import diagnose

    report["diagnostics"] = diagnose(report)
    metrics_pass = bool(
        structurally_usable
        and metadata_compatibility["passed"]
        and structural["norm_compatible"]
        and report["vector"]["mean"] >= thresholds.vector_mean
        and report["vector"]["minimum"] >= thresholds.vector_min
        and report["geometry"]["spearman"] >= thresholds.geometry_spearman
        and report["geometry"]["mean_absolute_difference"] <= thresholds.geometry_mae
        and report["neighbors"]["top_1_overlap"] >= thresholds.top1_agreement
        and report["neighbors"]["top_5_overlap"] >= thresholds.top5_overlap
        and report["neighbors"]["top_10_overlap"] >= thresholds.top10_overlap
        and report["batch_consistency"]["reference"]["passed"]
        and report["batch_consistency"]["candidate"]["passed"]
        and (report["length_analysis"] is None or report["length_analysis"]["passed"])
    )
    report["passed"] = metrics_pass
    return report


def analyze_lengths(
    reference: EmbeddingBackend,
    candidate: EmbeddingBackend,
    factory: Callable[[int], str],
    lengths: Sequence[int],
    batch_size: int,
    threshold: float,
) -> dict[str, Any]:
    rows = []
    for length in lengths:
        text = factory(length)
        ref = reference.encode([text], batch_size)
        can = candidate.encode([text], batch_size)
        compatible = ref.shape == can.shape and ref.ndim == 2 and np.isfinite(ref).all() and np.isfinite(can).all()
        cosine = float(_cosine_rows(ref, can)[0]) if compatible else float("nan")
        rows.append({"tokens": length, "cosine": cosine, "passed": bool(np.isfinite(cosine) and cosine >= threshold)})
    failing = [row for row in rows if not row["passed"]]
    result: dict[str, Any] = {"threshold": threshold, "points": rows, "passed": not failing, "minimal_reproducer": None}
    passing_before_failure = any(row["passed"] and row["tokens"] < failing[0]["tokens"] for row in rows) if failing else False
    if failing and passing_before_failure:
        high = failing[0]["tokens"]
        low = max(row["tokens"] for row in rows if row["passed"] and row["tokens"] < high)
        original = high
        while low + 1 < high:
            middle = (low + high) // 2
            text = factory(middle)
            cosine = float(_cosine_rows(reference.encode([text], batch_size), candidate.encode([text], batch_size))[0])
            if np.isfinite(cosine) and cosine >= threshold:
                low = middle
            else:
                high = middle
        below_text, failing_text = factory(high - 1), factory(high)
        below_cos = float(_cosine_rows(reference.encode([below_text], batch_size), candidate.encode([below_text], batch_size))[0])
        fail_cos = float(_cosine_rows(reference.encode([failing_text], batch_size), candidate.encode([failing_text], batch_size))[0])
        result["minimal_reproducer"] = {
            "original_failing_tokens": original,
            "tokens": high,
            "previous_tokens": high - 1,
            "previous_cosine": below_cos,
            "cosine": fail_cos,
        }
    return result
