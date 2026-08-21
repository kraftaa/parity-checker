from __future__ import annotations

import json

import pytest

import embed_parity.cli as cli
from embed_parity.cli import build_parser
from tests.fakes import FakeBackend


def test_parser_accepts_custom_probe_and_length_options():
    args = build_parser().parse_args(
        [
            "compare",
            "--model",
            "org/model",
            "--tei",
            "http://localhost:8080",
            "--probe-file",
            "probes.jsonl",
            "--lengths",
            "64,128,257",
            "--timeout",
            "30",
            "--concurrent-requests",
            "6",
            "--concurrency-trials",
            "2",
        ]
    )
    assert args.lengths == (64, 128, 257)
    assert args.probe_file == "probes.jsonl"
    assert args.timeout == 30
    assert args.concurrent_requests == 6
    assert args.concurrency_trials == 2


def test_parser_can_disable_concurrency_check():
    args = build_parser().parse_args(
        [
            "compare",
            "--model",
            "org/model",
            "--tei",
            "http://localhost:8080",
            "--no-concurrency-check",
        ]
    )
    assert args.concurrent_requests == 0


def test_parser_rejects_unsorted_lengths():
    with pytest.raises(SystemExit):
        build_parser().parse_args(
            [
                "compare",
                "--model",
                "org/model",
                "--tei",
                "http://localhost:8080",
                "--lengths",
                "128,64",
            ]
        )


def test_parser_accepts_tei_transport_options():
    args = build_parser().parse_args(
        [
            "compare",
            "--model",
            "org/model",
            "--tei",
            "http://localhost:8080",
            "--tei-api-key",
            "secret",
            "--tei-header",
            "X-Tenant=acme",
            "--tei-retries",
            "4",
            "--no-tei-truncate",
        ]
    )
    assert args.tei_api_key == "secret"
    assert args.tei_header == [("X-Tenant", "acme")]
    assert args.tei_retries == 4
    assert args.tei_truncate is False


def test_parser_accepts_workload_options():
    args = build_parser().parse_args(
        [
            "workload",
            "--model",
            "org/model",
            "--tei",
            "http://localhost:8080",
            "--input",
            "workload.jsonl",
            "--max-documents",
            "500",
            "--max-queries",
            "50",
            "--seed",
            "7",
            "--baseline",
            "old.json",
            "--html",
            "report.html",
        ]
    )
    assert args.command == "workload"
    assert args.max_documents == 500
    assert args.max_queries == 50
    assert args.seed == 7
    assert args.baseline == "old.json"
    assert args.html == "report.html"


def test_execution_failure_writes_json(monkeypatch, tmp_path):
    def fail(_args):
        raise RuntimeError("server unavailable")

    monkeypatch.setattr(cli, "run_compare", fail)
    output = tmp_path / "failure.json"
    exit_code = cli.main(
        [
            "compare",
            "--model",
            "org/model",
            "--tei",
            "http://localhost:8080",
            "--json",
            str(output),
        ]
    )
    payload = json.loads(output.read_text())
    assert exit_code == 2
    assert payload["status"] == "execution_error"
    assert payload["error"]["message"] == "server unavailable"


def test_unwritable_error_report_still_returns_exit_two(monkeypatch, tmp_path, capsys):
    def fail(_args):
        raise RuntimeError("server unavailable")

    monkeypatch.setattr(cli, "run_compare", fail)
    exit_code = cli.main(
        [
            "compare",
            "--model",
            "org/model",
            "--tei",
            "http://localhost:8080",
            "--json",
            str(tmp_path),
        ]
    )
    assert exit_code == 2
    assert "could not write error report" in capsys.readouterr().err


def test_workload_command_writes_json_html_and_baseline(monkeypatch, tmp_path):
    class Runtime(FakeBackend):
        def wait_until_ready(self, timeout):
            return None

        def close(self):
            return None

    monkeypatch.setattr(cli, "TEIBackend", lambda *args, **kwargs: Runtime())
    monkeypatch.setattr(cli, "SentenceTransformersBackend", lambda *args, **kwargs: Runtime())
    workload = tmp_path / "workload.jsonl"
    rows = [{"id": "q", "type": "query", "text": "query"}]
    rows.extend(
        {"id": f"d-{index}", "type": "document", "text": f"document {index}"} for index in range(6)
    )
    workload.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    output = tmp_path / "report.json"
    baseline = tmp_path / "baseline.json"
    artifact = tmp_path / "report.html"
    exit_code = cli.main(
        [
            "workload",
            "--model",
            "org/model",
            "--tei",
            "http://localhost:8080",
            "--input",
            str(workload),
            "--json",
            str(output),
            "--save-baseline",
            str(baseline),
            "--html",
            str(artifact),
        ]
    )
    assert exit_code == 0
    assert json.loads(output.read_text())["report_type"] == "workload_parity"
    assert json.loads(baseline.read_text())["passed"] is True
    assert "<!doctype html>" in artifact.read_text()
