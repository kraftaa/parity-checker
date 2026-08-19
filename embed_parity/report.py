from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import __version__


def _fmt(value: Any, digits: int = 6) -> str:
    if isinstance(value, float):
        return f"{value:.{digits}f}" if math.isfinite(value) else "n/a"
    return str(value)


def _status(value: bool) -> str:
    return "PASS" if value else "FAIL"


def render_text(report: dict[str, Any]) -> str:
    s = report["structural"]
    lines = [
        "embed-parity",
        "",
        "Model",
        f"  {report['model']}",
        f"  probes: {report['probe_count']} used, {len(report.get('skipped_probes', []))} skipped",
        "",
        "Reference",
        f"  {report['reference'].get('runtime', 'SentenceTransformers')}",
        f"  dimension: {s['reference']['dimension']}",
        "",
        "Candidate",
        f"  {report['candidate'].get('runtime', 'TEI')}",
        f"  dimension: {s['candidate']['dimension']}",
        "",
        "Thresholds (configurable; not universal)",
    ]
    for key, value in report["thresholds"].items():
        lines.append(f"  {key:<32} {_fmt(value)}")
    metadata = report.get("metadata_compatibility", {})
    lines.extend(["", "Metadata compatibility"])
    for key, label in (
        ("model_id_match", "model identifier"),
        ("revision_match", "resolved revision"),
        ("dtype_match", "dtype"),
        ("pooling_match", "pooling"),
    ):
        value = metadata.get(key)
        outcome = "UNKNOWN" if value is None else _status(value)
        lines.append(f"  {label:<32} {outcome}")
    lines.extend(
        [
            "",
            "Structural",
            f"  output count                     {_status(s['same_output_count'])}",
            f"  dimensions                       {_status(s['same_dimension'])}",
            f"  finite vectors                   {_status(s['finite_vectors'])}",
            f"  norm compatibility               {_status(s['norm_compatible'])}",
            f"  max norm-distribution difference {_fmt(s['norm_max_relative_difference'])}",
            f"  reference norm mean              {_fmt(s['reference']['norms']['mean'])}",
            f"  candidate norm mean              {_fmt(s['candidate']['norms']['mean'])}",
        ]
    )
    if not s["same_dimension"]:
        lines.extend(
            [
                "",
                "FAIL: output dimension mismatch",
                f"  SentenceTransformers: {s['reference']['dimension']}",
                f"  TEI:                  {s['candidate']['dimension']}",
            ]
        )
    if report["vector"]:
        v, g, n = report["vector"], report["geometry"], report["neighbors"]
        vector_pass = (
            v["mean"] >= report["thresholds"]["vector_mean"]
            and v["minimum"] >= report["thresholds"]["vector_min"]
        )
        geometry_pass = (
            g["spearman"] >= report["thresholds"]["geometry_spearman"]
            and g["mean_absolute_difference"] <= report["thresholds"]["geometry_mae"]
        )
        neighbors_pass = (
            n["top_1_overlap"] >= report["thresholds"]["top1_agreement"]
            and n["top_5_overlap"] >= report["thresholds"]["top5_overlap"]
            and n["top_10_overlap"] >= report["thresholds"]["top10_overlap"]
        )
        lines.extend(
            [
                "",
                "Vector parity",
                f"  mean cosine                      {_fmt(v['mean'])}",
                f"  median cosine                    {_fmt(v['median'])}",
                f"  p01 cosine                       {_fmt(v['p01'])}",
                f"  p05 cosine                       {_fmt(v['p05'])}",
                f"  minimum                          {_fmt(v['minimum'])}",
                f"                                   {_status(vector_pass)}",
                "",
                "Worst 10 probes",
            ]
        )
        for item in v["worst"]:
            preview = item["text"].replace("\n", " ")[:70]
            lines.append(f"  {item['id']:<8} {_fmt(item['cosine'])}  {preview}")
        lines.extend(
            [
                "",
                "Geometry parity",
                f"  pairwise Pearson                 {_fmt(g['pearson'])}",
                f"  pairwise Spearman                {_fmt(g['spearman'])}",
                f"  mean absolute difference         {_fmt(g['mean_absolute_difference'])}",
                f"  maximum difference               {_fmt(g['maximum_difference'])}",
                f"                                   {_status(geometry_pass)}",
                "",
                "Nearest-neighbor parity",
                f"  top-1 agreement                  {_fmt(n['top_1_overlap'] * 100, 2)}%",
                f"  top-5 overlap                    {_fmt(n['top_5_overlap'] * 100, 2)}%",
                f"  top-10 overlap                   {_fmt(n['top_10_overlap'] * 100, 2)}%",
                f"                                   {_status(neighbors_pass)}",
                "",
                "Neighborhoods changed most",
            ]
        )
        for item in n["most_changed"]:
            lines.append(
                f"  {item['id']:<8} top-10 overlap {_fmt(item['top_10_overlap'] * 100, 1)}%"
            )
    batches = report.get("batch_consistency")
    if batches:
        lines.extend(["", "Batch consistency"])
        for key, label in (("reference", "SentenceTransformers"), ("candidate", "TEI")):
            lines.append(f"  {label}  {_status(batches[key]['passed'])}")
            for item in batches[key]["comparisons"]:
                batch_label = f"batch {item['batch_a']} vs {item['batch_b']}"
                lines.append(
                    f"    {batch_label}: mean {_fmt(item['mean_cosine'])}  "
                    f"{_status(item['passed'])}"
                )
    server_batch = report.get("server_batch_consistency")
    if server_batch:
        lines.extend(
            [
                "",
                "Concurrent server batching",
                f"  probe                            {server_batch['probe_id']}",
                f"  concurrency                      {server_batch['concurrent_requests']}",
                f"  trials                           {server_batch['trials']}",
                f"  responses compared               {server_batch['responses_compared']}",
                f"  mean cosine                      {_fmt(server_batch['mean_cosine'])}",
                f"  minimum cosine                   {_fmt(server_batch['minimum_cosine'])}",
                f"  divergent responses              {server_batch['divergent_responses']}",
                f"                                   {_status(server_batch['passed'])}",
            ]
        )
    length = report.get("length_analysis")
    if length:
        lines.extend(["", "Input length parity"])
        for point in length["points"]:
            point_status = _status(point["passed"])
            lines.append(f"  {point['tokens']:>4} tokens  {_fmt(point['cosine'])}  {point_status}")
        minimal = length.get("minimal_reproducer")
        if minimal:
            previous = (
                f"  {minimal['previous_tokens']} tokens: cosine = "
                f"{_fmt(minimal['previous_cosine'])}"
            )
            lines.extend(
                [
                    "",
                    "Minimal reproducer",
                    f"  original failing probe: {minimal['original_failing_tokens']} tokens",
                    previous,
                    f"  {minimal['tokens']} tokens: cosine = {_fmt(minimal['cosine'])}",
                ]
            )
    prefix = report.get("prefix_analysis")
    if prefix:
        for name in ("query", "document"):
            analysis = prefix.get(name)
            if analysis:
                raw_score = _fmt(analysis["scores"]["raw_to_raw"])
                improvement = _fmt(analysis["cross_variant_improvement"])
                lines.extend(
                    [
                        "",
                        f"{name.title()}-prefix analysis",
                        f"  prefix                           {analysis['prefix']!r}",
                        f"  raw-to-raw                       {raw_score}",
                        f"  best cross-variant improvement   {improvement}",
                    ]
                )
    if report["diagnostics"]:
        lines.extend(["", "Possible causes / observations"])
        for item in report["diagnostics"]:
            lines.append(f"  - {item['message']}")
    lines.extend(["", f"Overall                            {_status(report['passed'])}"])
    return "\n".join(lines) + "\n"


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def write_json(report: dict[str, Any], path: str | Path) -> None:
    Path(path).write_text(
        json.dumps(_json_safe(report), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_error_json(
    path: str | Path, *, model: str | None, tei: str | None, error: BaseException
) -> None:
    payload = {
        "schema_version": 3,
        "tool_version": __version__,
        "status": "execution_error",
        "passed": False,
        "exit_code": 2,
        "model": model,
        "candidate_url": tei,
        "error": {"type": type(error).__name__, "message": str(error)},
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(payload, path)
