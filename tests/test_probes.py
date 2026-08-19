from __future__ import annotations

import json

import pytest

from embed_parity.probes import load_jsonl_probes


def test_load_jsonl_probes_with_defaults(tmp_path):
    path = tmp_path / "probes.jsonl"
    path.write_text(
        json.dumps({"id": "custom", "category": "query", "text": "find tea"})
        + "\n\n"
        + json.dumps({"text": "A document about tea."})
        + "\n",
        encoding="utf-8",
    )
    probes = load_jsonl_probes(path)
    assert [probe.id for probe in probes] == ["custom", "user-0001"]
    assert probes[1].category == "user"


@pytest.mark.parametrize(
    "content, message",
    [
        ('{"text": 3}\n{"text": "valid"}\n', "string 'text'"),
        ('{"id": "same", "text": "a"}\n{"id": "same", "text": "b"}\n', "duplicate probe id"),
        ('{"text": "only one"}\n', "at least two"),
    ],
)
def test_load_jsonl_probes_rejects_bad_input(tmp_path, content, message):
    path = tmp_path / "bad.jsonl"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_jsonl_probes(path)

