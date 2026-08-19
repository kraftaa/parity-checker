from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable


@dataclass(frozen=True)
class Probe:
    id: str
    category: str
    text: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


def built_in_probes() -> list[Probe]:
    """Return a deterministic, varied 100-probe corpus."""
    fixed: list[tuple[str, str]] = [
        ("short", "Hello."), ("short", "A"), ("short", "the quick brown fox"),
        ("empty", ""), ("empty", " "), ("empty", "\n"),
        ("semantic", "A dog runs through a grassy field."),
        ("semantic", "A canine is sprinting across the lawn."),
        ("semantic", "How do I reset my password?"),
        ("semantic", "Steps for changing an account password."),
        ("unrelated", "The telescope observed a distant galaxy."),
        ("unrelated", "Sourdough bread needs a fermented starter."),
        ("query", "best noise cancelling headphones"),
        ("query", "weather in Montréal tomorrow"),
        ("query", "python sort dictionary by value"),
        ("document", "Noise-cancelling headphones reduce ambient sound using active signal processing."),
        ("document", "To sort a Python mapping, pass its items to sorted with a key function."),
        ("paragraph", "Embedding models map text into numerical vectors. Similar meanings should occupy nearby regions, while unrelated ideas should be farther apart."),
        ("paragraph", "A database transaction groups operations into a unit of work. Atomicity ensures that either every operation succeeds or none is committed."),
        ("code", "def fib(n):\n    return n if n < 2 else fib(n-1) + fib(n-2)"),
        ("code", "const unique = values => [...new Set(values)];"),
        ("code", "SELECT customer_id, SUM(total) FROM orders GROUP BY customer_id;"),
        ("sql", "WITH recent AS (SELECT * FROM events WHERE created_at >= CURRENT_DATE - INTERVAL '7 days') SELECT count(*) FROM recent;"),
        ("numbers", "0 1 -1 3.1415926535 1e10 999999999999"),
        ("numbers", "Order 48392 costs $1,024.56 and has 17 items."),
        ("punctuation", "!?.,;:—–…()[]{} <> / \\ | @#$%^&*_+="),
        ("unicode", "🙂 🚀 🌍 café naïve résumé coöperate"),
        ("unicode", "中文文本用于测试嵌入模型。"),
        ("non_english", "La inteligencia artificial transforma la búsqueda semántica."),
        ("non_english", "L'intelligence artificielle améliore la recherche sémantique."),
        ("non_english", "Künstliche Intelligenz verändert die semantische Suche."),
        ("non_english", "人工知能は意味検索を改善します。"),
        ("non_english", "الذكاء الاصطناعي يحسن البحث الدلالي."),
        ("non_english", "Искусственный интеллект улучшает семантический поиск."),
        ("case", "EMBEDDING RUNTIME PARITY"), ("case", "embedding runtime parity"),
        ("whitespace", "multiple    spaces\tand\nnewlines"),
        ("markup", "<p>Hello <strong>world</strong></p>"),
        ("markup", "# Heading\n\n- alpha\n- beta\n- gamma"),
        ("path", "/var/lib/models/model.onnx?revision=abc123"),
    ]
    probes = [Probe(f"p{i:03d}", category, text) for i, (category, text) in enumerate(fixed)]
    topics = [
        "astronomy", "baking", "distributed systems", "gardening", "classical music",
        "public transit", "marine biology", "cybersecurity", "woodworking", "economics",
    ]
    templates = [
        "A concise introduction to {topic}, including practical examples and common terminology.",
        "What are the most important concepts to learn first in {topic}?",
        "This document explains {topic} for a reader with no prior experience.",
        "Recent research discusses tradeoffs, measurements, and reproducibility in {topic}.",
        "Beginner's checklist: define a goal, gather evidence, and evaluate results for {topic}.",
        "Frequently asked question number {n}: how can we test assumptions about {topic}?",
    ]
    for topic in topics:
        for template in templates:
            i = len(probes)
            probes.append(Probe(f"p{i:03d}", "generated", template.format(topic=topic, n=i)))
    assert len(probes) == 100
    return probes


DEFAULT_LENGTHS = (32, 64, 128, 256, 384, 512, 768, 1024)


def length_probes(factory: Callable[[int], str], lengths=DEFAULT_LENGTHS) -> list[Probe]:
    return [Probe(f"length-{n}", "length", factory(n)) for n in lengths]


def load_jsonl_probes(path: str | Path) -> list[Probe]:
    """Load JSONL objects containing text and optional id/category fields."""
    probes: list[Probe] = []
    seen_ids: set[str] = set()
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            try:
                item = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on probe line {line_number}: {exc.msg}") from exc
            if not isinstance(item, dict) or not isinstance(item.get("text"), str):
                raise ValueError(f"probe line {line_number} must be an object with a string 'text'")
            probe_id = item.get("id", f"user-{len(probes):04d}")
            category = item.get("category", "user")
            if not isinstance(probe_id, str) or not probe_id:
                raise ValueError(f"probe line {line_number} has an invalid 'id'")
            if not isinstance(category, str) or not category:
                raise ValueError(f"probe line {line_number} has an invalid 'category'")
            if probe_id in seen_ids:
                raise ValueError(f"duplicate probe id {probe_id!r} on line {line_number}")
            seen_ids.add(probe_id)
            probes.append(Probe(probe_id, category, item["text"]))
    if len(probes) < 2:
        raise ValueError("probe file must contain at least two probes")
    return probes


def recommended_query_prefix(model_id: str) -> str | None:
    lower = model_id.lower()
    if "e5-" in lower or "/e5" in lower:
        return "query: "
    if "bge-" in lower and "-en" in lower:
        return "Represent this sentence for searching relevant passages: "
    return None
