from __future__ import annotations

import json

from embed_parity.compare import compare_backends
from embed_parity.probes import built_in_probes
from embed_parity.report import render_text, write_json
from tests.fakes import FakeBackend


def test_human_and_json_reports(tmp_path):
    report = compare_backends(
        "model", FakeBackend(), FakeBackend(), built_in_probes()[:12], batch_sizes=(1,)
    )
    text = render_text(report)
    assert "Vector parity" in text
    assert "Thresholds (configurable; not universal)" in text
    path = tmp_path / "report.json"
    write_json(report, path)
    parsed = json.loads(path.read_text())
    assert parsed["schema_version"] == 3
    assert parsed["tool_version"] == "0.4.0"
    assert parsed["passed"] is True


def test_human_report_renders_concurrent_server_batching():
    report = compare_backends(
        "model",
        FakeBackend(),
        FakeBackend(concurrency_sensitive=True),
        built_in_probes()[:12],
        batch_sizes=(1,),
        concurrent_requests=4,
        concurrency_trials=2,
    )
    text = render_text(report)
    assert "Concurrent server batching" in text
    assert "divergent responses              4" in text
    assert "Possible server-side batching" in text
