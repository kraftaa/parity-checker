from __future__ import annotations

from typing import Any


def diagnose(report: dict[str, Any]) -> list[dict[str, str]]:
    findings: list[dict[str, str]] = []
    structural = report["structural"]
    metadata = report.get("metadata_compatibility", {})
    if metadata.get("model_id_match") is False:
        findings.append({
            "code": "model_id_mismatch",
            "message": (
                f"Server metadata names model {metadata['candidate_model_id']!r}, not the "
                f"reference model {metadata['reference_model_id']!r}."
            ),
        })
    if metadata.get("revision_match") is False:
        findings.append({
            "code": "revision_mismatch",
            "message": (
                f"Known model revisions differ: reference {metadata['reference_revision']!r}, "
                f"candidate {metadata['candidate_revision']!r}."
            ),
        })
    if metadata.get("dtype_match") is False:
        findings.append({
            "code": "dtype_difference",
            "message": (
                f"Runtime dtypes differ ({metadata['reference_dtype']} vs {metadata['candidate_dtype']}); "
                "small numerical differences may be expected."
            ),
        })
    if metadata.get("pooling_match") is False:
        findings.append({
            "code": "pooling_mismatch",
            "message": (
                f"Known pooling modes differ: reference {metadata['reference_pooling']!r}, "
                f"candidate {metadata['candidate_pooling']!r}."
            ),
        })
    if not structural["same_output_count"]:
        findings.append({"code": "output_count", "message": "Output counts differ between runtimes."})
    if not structural["same_dimension"]:
        findings.append({
            "code": "dimension_mismatch",
            "message": "Possible pooling, projection, model, or export mismatch; embedding dimensions differ.",
        })
    if not structural["finite_vectors"]:
        findings.append({"code": "non_finite", "message": "One or both runtimes returned NaN or Inf values."})
    if structural["reference"]["zero_vectors"] or structural["candidate"]["zero_vectors"]:
        findings.append({"code": "zero_vectors", "message": "One or both runtimes returned zero vectors."})
    vector = report.get("vector")
    if vector and vector["mean"] >= 0.999 and not structural["norm_compatible"]:
        findings.append({
            "code": "normalization_mismatch",
            "message": "Possible normalization mismatch: directions agree but vector norm distributions differ.",
        })
    prefix = report.get("prefix_analysis")
    if prefix and prefix["possible_mismatch"]:
        findings.append({
            "code": "query_prefix_mismatch",
            "message": "A model-recommended query prefix substantially improves cross-runtime parity for query probes.",
        })
    length = report.get("length_analysis")
    if length and not length["passed"] and any(point["passed"] for point in length["points"]):
        findings.append({
            "code": "length_boundary",
            "message": "Possible truncation or maximum-sequence-length mismatch; short inputs pass while longer inputs diverge.",
        })
    batches = report.get("batch_consistency")
    if batches:
        for key, label in (("reference", "SentenceTransformers"), ("candidate", "TEI")):
            if not batches[key]["passed"]:
                findings.append({
                    "code": f"batch_sensitive_{key}",
                    "message": f"Possible batch-dependent inference behavior in {label}.",
                })
    if vector and vector["mean"] < report["thresholds"]["vector_mean"] and not any(
        item["code"] in {"normalization_mismatch", "query_prefix_mismatch", "length_boundary"} for item in findings
    ):
        findings.append({
            "code": "embedding_space_mismatch",
            "message": "Embedding directions differ materially; possible revision, dtype, preprocessing, or runtime mismatch.",
        })
    return findings
