# ruff: noqa: E501
from __future__ import annotations

import html
import json
import math
from pathlib import Path
from typing import Any


def _number(value: Any, digits: int = 4) -> str:
    if isinstance(value, (float, int)):
        number = float(value)
        return f"{number:.{digits}f}" if math.isfinite(number) else "n/a"
    return "n/a"


def _percent(value: Any, digits: int = 1) -> str:
    if isinstance(value, (float, int)) and math.isfinite(float(value)):
        return f"{float(value) * 100:.{digits}f}%"
    return "n/a"


def _status(value: bool) -> str:
    return "PASS" if value else "FAIL"


def _preview(text: str, limit: int = 76) -> str:
    cleaned = " ".join(text.split())
    return cleaned if len(cleaned) <= limit else cleaned[: limit - 1] + "…"


def render_workload_text(report: dict[str, Any]) -> str:
    input_details = report["input"]
    vector = report["vector"]
    retrieval = report.get("retrieval")
    lines = [
        "embed-parity workload",
        "",
        "Workload parity",
        f"  model                            {report['model']}",
        f"  reference                        {report['reference'].get('runtime', 'reference')}",
        f"  candidate                        {report['candidate'].get('runtime', 'candidate')}",
        "",
        "Input",
        f"  documents available              {input_details['documents_available']:,}",
        f"  documents sampled                {input_details['documents_sampled']:,}",
        f"  queries available                {input_details['queries_available']:,}",
        f"  queries sampled                  {input_details['queries_sampled']:,}",
        f"  seed                             {input_details['seed']}",
    ]
    if input_details["sampling_applied"]:
        lines.extend(
            [
                "  note                             deterministic sample; results describe only this sample",
            ]
        )
    summary = vector["summary"]
    lines.extend(
        [
            "",
            "Actual embedding differences",
            f"  dimensions match                 {_status(summary['dimensions_match'])}",
            f"  all values finite                {_status(summary['all_finite'])}",
            f"  mean cosine                      {_number(summary['mean'])}",
            f"  minimum cosine                   {_number(summary['minimum'])}",
            f"  p95 relative norm difference     {_percent(summary['norm_difference']['p95'])}",
            f"                                   {_status(vector['passed'])}",
            "",
            "Worst vector differences",
            "  ID                 Type       Cosine   Text",
        ]
    )
    for item in vector["worst"]:
        lines.append(
            f"  {item['id'][:18]:<18} {item['type']:<10} {_number(item['cosine']):<8} "
            f"{_preview(item['text'], 58)}"
        )
    if retrieval is None:
        lines.extend(
            [
                "",
                "Retrieval behavior",
                "  unavailable: a runtime did not return a finite query/document embedding space",
            ]
        )
    else:
        metrics = retrieval["summary"]
        lines.extend(
            [
                "",
                "Retrieval behavior",
                f"  queries                          {metrics['query_count']:,}",
                f"  documents                        {metrics['document_count']:,}",
                f"  top-1 agreement                  {_percent(metrics['top_1_agreement'])}",
                f"  top-5 overlap                    {_percent(metrics['top_5_overlap'])}",
                f"  top-10 overlap                   {_percent(metrics['top_10_overlap'])}",
                "",
                "Most affected queries",
            ]
        )
        for query in retrieval["most_affected"]:
            marker = "  STRUCTURALLY VALID, RETRIEVAL CHANGED" if query["silent_risk"] else ""
            lines.extend(
                [
                    "",
                    f"  {query['id']}  {_preview(query['text'])}",
                    f"  vector cosine                    {_number(query['vector_cosine'], 3)}",
                    f"  top-5 retrieval overlap          {_percent(query['top_5_overlap'], 0)}",
                ]
            )
            if marker:
                lines.append(marker)
                lines.append(
                    "  Structurally valid embeddings produced materially different retrieval."
                )
            lines.append("  Reference                         Candidate")
            for rank in range(min(5, len(query["reference_top"]))):
                left = query["reference_top"][rank]["id"]
                right = query["candidate_top"][rank]["id"]
                lines.append(f"    {rank + 1}. {left:<26} {rank + 1}. {right}")
        if not retrieval["most_affected"]:
            lines.append("  none — evaluated rankings met the parity thresholds")
        if retrieval["most_affected"] and retrieval["most_affected"][0]["top_5_overlap"] < 0.8:
            lines.extend(["", "Retrieval behavior changed substantially for the examples above."])
    baseline = report.get("baseline_comparison")
    if baseline:
        labels = {
            "top_1_agreement": "Top-1 agreement",
            "top_5_overlap": "Top-5 overlap",
            "top_10_overlap": "Top-10 overlap",
        }
        lines.extend(["", "Compared with previous deployment"])
        for key, item in baseline["metrics"].items():
            lines.extend(
                [
                    f"  {labels[key]}",
                    f"    baseline                       {_percent(item['baseline'])}",
                    f"    current                        {_percent(item['current'])}",
                    f"    change                         {item['change_percentage_points']:+.1f} pp",
                ]
            )
        if baseline["retrieval_parity_regression"]:
            lines.append("  Retrieval-parity regression detected; this is not a quality judgment.")
    lines.extend(
        [
            "",
            "No relevance labels were supplied. This measures behavioral parity, not search quality.",
            f"Overall                            {_status(report['passed'])}",
        ]
    )
    return "\n".join(lines) + "\n"


def _histogram_markup(items: list[dict[str, Any]], *, percent_edges: bool = False) -> str:
    maximum = max((int(item["count"]) for item in items), default=1) or 1
    rows = []
    for item in items:
        lower = float(item["lower"])
        upper = float(item["upper"])
        label = (
            f"{max(0, lower) * 100:.0f}–{min(1, upper) * 100:.0f}%"
            if percent_edges
            else f"{max(-1, lower):.3f}–{min(1, upper):.3f}"
        )
        width = int(item["count"]) / maximum * 100
        rows.append(
            f'<div class="bar-row"><span>{html.escape(label)}</span><i style="width:{width:.2f}%"></i>'
            f"<b>{int(item['count'])}</b></div>"
        )
    return "".join(rows)


def _metadata_json(report: dict[str, Any]) -> str:
    payload = {
        "model": report["model"],
        "reference": report["reference"],
        "candidate": report["candidate"],
        "role_encoding": report["role_encoding"],
        "configuration": report["configuration"],
        "sample_fingerprint": report["input"]["sample_fingerprint"],
    }
    return html.escape(json.dumps(payload, indent=2, sort_keys=True, default=str))


def render_workload_html(report: dict[str, Any]) -> str:
    retrieval = report.get("retrieval")
    vector = report["vector"]
    input_details = report["input"]
    metric_cards = ""
    affected = (
        "<p>Retrieval comparison was unavailable because the embedding spaces were invalid.</p>"
    )
    overlap_histogram = ""
    record_text = {item["id"]: item["text"] for item in vector["records"]}
    if retrieval:
        summary = retrieval["summary"]
        metric_cards = "".join(
            f'<div class="metric"><span>{label}</span><strong>{_percent(summary[key])}</strong></div>'
            for key, label in (
                ("top_1_agreement", "Top-1 agreement"),
                ("top_5_overlap", "Top-5 overlap"),
                ("top_10_overlap", "Top-10 overlap"),
            )
        )
        overlap_histogram = _histogram_markup(retrieval["overlap_distribution"], percent_edges=True)
        sections = []
        for query in retrieval["most_affected"]:
            badge = (
                '<span class="risk">STRUCTURALLY VALID, RETRIEVAL CHANGED</span>'
                if query["silent_risk"]
                else ""
            )
            left = "".join(
                f"<li><b>{html.escape(item['id'])}</b>"
                f"<small>{html.escape(_preview(record_text[item['id']], 90))}</small></li>"
                for item in query["reference_top"][:5]
            )
            right = "".join(
                f"<li><b>{html.escape(item['id'])}</b>"
                f"<small>{html.escape(_preview(record_text[item['id']], 90))}</small></li>"
                for item in query["candidate_top"][:5]
            )
            sections.append(
                f"<article><h3>{html.escape(query['id'])}</h3>{badge}"
                f"<p>{html.escape(query['text'])}</p>"
                f'<div class="query-stats">Vector cosine '
                f"<b>{_number(query['vector_cosine'], 3)}</b> · Top-5 overlap "
                f"<b>{_percent(query['top_5_overlap'], 0)}</b></div>"
                f'<div class="rankings"><div><h4>Reference</h4><ol>{left}</ol></div>'
                f"<div><h4>Candidate</h4><ol>{right}</ol></div></div></article>"
            )
        affected = "".join(sections) or (
            '<div class="panel">No changed queries in the evaluated workload.</div>'
        )
    worst_rows = "".join(
        f"<tr><td>{html.escape(item['id'])}</td><td>{html.escape(item['type'])}</td>"
        f"<td>{_number(item['cosine'])}</td><td>{html.escape(_preview(item['text']))}</td></tr>"
        for item in vector["worst"]
    )
    sampling_note = (
        "Deterministic sample only; no claim of statistical representativeness."
        if input_details["sampling_applied"]
        else "All supplied workload records were evaluated."
    )
    status = _status(report["passed"])
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>embed-parity workload report</title><style>
:root{{--ink:#17212b;--muted:#5d6873;--line:#d9e0e5;--paper:#f5f7f8;--panel:#fff;--cyan:#087ca7;--red:#b42318;--green:#16794a}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--paper);color:var(--ink);font:15px/1.55 system-ui,-apple-system,sans-serif}}
main{{width:min(1120px,calc(100% - 32px));margin:auto;padding:52px 0 80px}}h1{{font-size:44px;letter-spacing:-.04em;margin:.1em 0}}h2{{margin-top:48px}}
.eyebrow{{font:700 12px ui-monospace,monospace;color:var(--cyan);letter-spacing:.1em}}.sub{{color:var(--muted)}}.status{{display:inline-block;padding:5px 10px;border-radius:99px;background:{"#e8f5ee" if report["passed"] else "#fff0ee"};color:{"var(--green)" if report["passed"] else "var(--red)"};font-weight:800}}
.metrics{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin:28px 0}}.metric,article,.panel{{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:20px}}.metric span{{display:block;color:var(--muted)}}.metric strong{{font-size:30px}}
.grid{{display:grid;grid-template-columns:1fr 1fr;gap:22px}}table{{width:100%;border-collapse:collapse;background:white}}th,td{{padding:11px;text-align:left;border-bottom:1px solid var(--line);vertical-align:top}}th{{font-size:12px;color:var(--muted)}}
.bar-row{{display:grid;grid-template-columns:90px 1fr 35px;gap:10px;align-items:center;margin:9px 0}}.bar-row span,.bar-row b{{font:12px ui-monospace,monospace}}.bar-row i{{display:block;height:11px;background:var(--cyan);border-radius:2px}}
article{{margin:16px 0}}article h3{{display:inline-block;margin:0 12px 8px 0}}.risk{{color:var(--red);font:800 11px ui-monospace,monospace}}.query-stats{{color:var(--muted)}}.rankings{{display:grid;grid-template-columns:1fr 1fr;gap:18px}}ol{{padding-left:26px}}li{{padding:7px 0}}li small{{display:block;color:var(--muted)}}pre{{overflow:auto;background:#13202b;color:#e7f1f6;padding:20px;border-radius:8px;font:12px/1.5 ui-monospace,monospace}}
@media(max-width:720px){{.metrics,.grid,.rankings{{grid-template-columns:1fr}}h1{{font-size:34px}}}}
</style></head><body><main>
<div class="eyebrow">EMBEDDING WORKLOAD PARITY</div><h1>{html.escape(report["model"])}</h1><span class="status">{status}</span>
<p class="sub">{input_details["queries_sampled"]:,} queries · {input_details["documents_sampled"]:,} documents · seed {input_details["seed"]}<br>{html.escape(sampling_note)}</p>
<h2>Overall retrieval parity</h2><div class="metrics">{metric_cards}</div>
<p class="sub">This measures behavioral parity, not retrieval quality. No relevance labels were supplied.</p>
<div class="grid"><section><h2>Vector cosine distribution</h2><div class="panel">{_histogram_markup(vector["distribution"])}</div></section>
<section><h2>Top-5 overlap distribution</h2><div class="panel">{overlap_histogram}</div></section></div>
<h2>Worst vector differences</h2><table><thead><tr><th>ID</th><th>Type</th><th>Cosine</th><th>Text</th></tr></thead><tbody>{worst_rows}</tbody></table>
<h2>Most affected queries</h2>{affected}
<h2>Runtime and model metadata</h2><pre>{_metadata_json(report)}</pre>
</main></body></html>"""


def write_workload_html(report: dict[str, Any], path: str | Path) -> None:
    Path(path).write_text(render_workload_html(report), encoding="utf-8")
