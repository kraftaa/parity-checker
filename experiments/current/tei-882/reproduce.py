"""Reproduce TEI issue #882 with real pinned Qwen3 weights."""

from __future__ import annotations

import argparse
import asyncio
import json

import httpx
import numpy as np
from transformers import AutoTokenizer

MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"
MODEL_REVISION = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"

EQUAL_LENGTH_A = "Database indexes make repeated queries faster."
EQUAL_LENGTH_B = "Ocean currents shape regional weather patterns."
SHORT_CONTROL = "short text"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tei", required=True)
    return parser.parse_args()


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.dot(left, right) / (np.linalg.norm(left) * np.linalg.norm(right)))


def embed(client: httpx.Client, texts: list[str]) -> np.ndarray:
    response = client.post(
        "/embed",
        json={"inputs": texts, "normalize": True, "truncate": True},
    )
    response.raise_for_status()
    return np.asarray(response.json(), dtype=np.float32)


async def embed_concurrently(base_url: str, texts: list[str]) -> np.ndarray:
    async with httpx.AsyncClient(base_url=base_url, timeout=600) as client:
        responses = await asyncio.gather(
            *(
                client.post(
                    "/embed",
                    json={"inputs": [text], "normalize": True, "truncate": True},
                )
                for text in texts
            )
        )
    for response in responses:
        response.raise_for_status()
    return np.asarray([response.json()[0] for response in responses], dtype=np.float32)


def main() -> None:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    token_counts = {
        text: len(tokenizer(text)["input_ids"])
        for text in (EQUAL_LENGTH_A, EQUAL_LENGTH_B, SHORT_CONTROL)
    }
    assert token_counts[EQUAL_LENGTH_A] == token_counts[EQUAL_LENGTH_B]
    assert token_counts[EQUAL_LENGTH_A] != token_counts[SHORT_CONTROL]

    with httpx.Client(base_url=args.tei, timeout=600) as client:
        single_a = embed(client, [EQUAL_LENGTH_A])[0]
        single_b = embed(client, [EQUAL_LENGTH_B])[0]
        single_control = embed(client, [SHORT_CONTROL])[0]
        equal_batch = embed(client, [EQUAL_LENGTH_A, EQUAL_LENGTH_B])
        padded_batch = embed(client, [EQUAL_LENGTH_A, SHORT_CONTROL])
    concurrent_texts = [
        EQUAL_LENGTH_A,
        EQUAL_LENGTH_B,
        EQUAL_LENGTH_A,
        EQUAL_LENGTH_B,
    ]
    concurrent_batch = asyncio.run(embed_concurrently(args.tei, concurrent_texts))

    print(
        json.dumps(
            {
                "model": MODEL_ID,
                "revision": MODEL_REVISION,
                "tei": args.tei,
                "token_counts": token_counts,
                "client_list_batch_vs_single": {
                    "a": cosine(equal_batch[0], single_a),
                    "b": cosine(equal_batch[1], single_b),
                },
                "concurrent_router_batch_vs_single": [
                    cosine(candidate, single_a if text == EQUAL_LENGTH_A else single_b)
                    for text, candidate in zip(concurrent_texts, concurrent_batch, strict=True)
                ],
                "unequal_length_batch_vs_single": {
                    "a": cosine(padded_batch[0], single_a),
                    "control": cosine(padded_batch[1], single_control),
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
