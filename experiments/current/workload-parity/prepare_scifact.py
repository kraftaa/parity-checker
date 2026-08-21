"""Download BEIR SciFact and create a deterministic workload JSONL file."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import tempfile
import urllib.request
import zipfile
from pathlib import Path

URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"
EXPECTED_MD5 = "5f7d1de60b170fc8027bb7898e2efca1"


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("scifact-workload.jsonl"))
    parser.add_argument("--documents", type=int, default=500)
    parser.add_argument("--queries", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    args = arguments()
    if args.documents < 1 or args.queries < 1:
        raise SystemExit("document and query counts must be positive")
    with tempfile.TemporaryDirectory(prefix="embed-parity-scifact-") as temporary:
        archive = Path(temporary) / "scifact.zip"
        urllib.request.urlretrieve(URL, archive)
        digest = hashlib.md5(archive.read_bytes()).hexdigest()  # noqa: S324 - published checksum
        if digest != EXPECTED_MD5:
            raise RuntimeError(f"SciFact checksum mismatch: expected {EXPECTED_MD5}, got {digest}")
        with zipfile.ZipFile(archive) as source:
            source.extractall(temporary)
        root = Path(temporary) / "scifact"
        documents = read_jsonl(root / "corpus.jsonl")
        queries = read_jsonl(root / "queries.jsonl")

        rng = random.Random(args.seed)
        selected_queries = rng.sample(queries, min(args.queries, len(queries)))
        selected_documents = rng.sample(documents, min(args.documents, len(documents)))
        rows = [
            {"id": str(item["_id"]), "type": "query", "text": str(item["text"])}
            for item in selected_queries
        ]
        rows.extend(
            {
                "id": str(item["_id"]),
                "type": "document",
                "text": " ".join(
                    part.strip()
                    for part in (str(item.get("title", "")), str(item["text"]))
                    if part.strip()
                ),
            }
            for item in selected_documents
        )
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )
    print(f"wrote {len(selected_queries)} queries and {len(selected_documents)} documents")


if __name__ == "__main__":
    main()
