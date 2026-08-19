from __future__ import annotations

import pytest

from embed_parity.cli import build_parser


def test_parser_accepts_custom_probe_and_length_options():
    args = build_parser().parse_args([
        "compare", "--model", "org/model", "--tei", "http://localhost:8080",
        "--probe-file", "probes.jsonl", "--lengths", "64,128,257", "--timeout", "30",
    ])
    assert args.lengths == (64, 128, 257)
    assert args.probe_file == "probes.jsonl"
    assert args.timeout == 30


def test_parser_rejects_unsorted_lengths():
    with pytest.raises(SystemExit):
        build_parser().parse_args([
            "compare", "--model", "org/model", "--tei", "http://localhost:8080",
            "--lengths", "128,64",
        ])
