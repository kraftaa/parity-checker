# Workload parity on BEIR SciFact

This experiment uses a deterministic 50-query, 500-document sample of the
public [BEIR SciFact](https://github.com/beir-cellar/beir) retrieval dataset. It
compares SentenceTransformers with matching TEI, then repeats the workload with
an unintended TEI default prompt. This is a real server configuration mismatch,
not a synthetic vector perturbation.

Generate the workload without adding a dataset-library dependency:

```bash
python prepare_scifact.py --documents 500 --queries 50 --seed 42
```

The script downloads the official `scifact.zip`, verifies its published MD5
checksum (`5f7d1de60b170fc8027bb7898e2efca1`), and writes
`scifact-workload.jsonl`. The generated corpus is intentionally excluded from
Git; anyone can reproduce the same records from the public source.

## Matching deployment

```bash
docker run --rm -p 8080:80 \
  -v "$PWD/data:/data" \
  ghcr.io/huggingface/text-embeddings-inference@sha256:35c50d7494de22deecdb783b8f5b7e1d05765709bd90071b03469b9440d28656 \
  --model-id sentence-transformers/all-MiniLM-L6-v2

embed-parity workload \
  --model sentence-transformers/all-MiniLM-L6-v2 \
  --tei http://127.0.0.1:8080 \
  --input scifact-workload.jsonl \
  --json matching.json \
  --html matching.html
```

## Realistic prompt mismatch

Restart the same image with `--default-prompt "query: "`, then run the same
workload with `--baseline matching.json` and write `prompt-mismatch.json`. The
server now applies a prefix that the reference deployment does not use. Endpoint
health and dimensions remain unchanged.

## Observed result

| Configuration | Vector minimum | Top-1 agreement | Top-5 overlap | Top-10 overlap |
|---|---:|---:|---:|---:|
| Matching | 1.0000 | 100.0% | 100.0% | 100.0% |
| Unintended default prompt | 0.9430 | 92.0% | 86.4% | 88.6% |

For query `788`, vector cosine was still `0.981` and both outputs were finite
384-dimensional vectors, but top-5 overlap fell to 60%. The report classified
it as `STRUCTURALLY VALID, RETRIEVAL CHANGED` and showed both result lists.

The two canonical JSON reports are included beside this README. The ARM64 image
is pinned by digest because the upstream release currently publishes the native
image under a moving `cpu-arm64-latest` tag.

This experiment compares behavior only. SciFact relevance judgments are not
loaded or evaluated, so the results must not be described as search-quality
changes.
