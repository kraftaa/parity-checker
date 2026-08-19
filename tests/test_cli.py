from __future__ import annotations

import json

import pytest

import embed_parity.cli as cli
from embed_parity.cli import build_parser


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
        ]
    )
    assert args.lengths == (64, 128, 257)
    assert args.probe_file == "probes.jsonl"
    assert args.timeout == 30


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
