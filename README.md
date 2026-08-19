# embed-parity

`embed-parity` answers one deliberately narrow question: does a Hugging Face Text
Embeddings Inference (TEI) deployment reproduce the embedding behavior of the
same model through SentenceTransformers?

It checks vector direction and norms, pairwise geometry, nearest neighbors,
batch consistency, model-aware query prefixes, and behavior at controlled token
lengths. When a length failure is found after shorter inputs pass, it performs a
binary search for the shortest failing token prefix.

## Installation

Python 3.10 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[sentence-transformers,test]'
```

## Usage

Start TEI with the same model used by the reference. For example, using the
official TEI container (choose an image tag appropriate to your environment):

```bash
docker run --gpus all -p 8080:80 \
  -v "$PWD/data:/data" \
  ghcr.io/huggingface/text-embeddings-inference:latest \
  --model-id BAAI/bge-small-en-v1.5
```

Then compare the runtimes:

```bash
embed-parity compare \
  --model BAAI/bge-small-en-v1.5 \
  --tei http://localhost:8080
```

Write a JSON report and select batch sizes with:

```bash
embed-parity compare \
  --model BAAI/bge-small-en-v1.5 \
  --tei http://localhost:8080 \
  --batch-sizes 1,8,32 \
  --json report.json
```

The human report prints every default threshold. They are operational defaults,
not universal scientific constants, and every threshold has a CLI flag. Run
`embed-parity compare --help` for the full list. Use `--revision` to pin the
SentenceTransformers side and `--normalize` or `--no-normalize` to make the
reference normalization choice explicit.

To reproduce a failure with your own corpus, provide JSONL objects with a required
`text` and optional `id` and `category`:

```jsonl
{"id":"query-1","category":"query","text":"best waterproof boots"}
{"id":"doc-1","category":"document","text":"These boots use a sealed membrane."}
```

```bash
embed-parity compare --model org/model --tei http://localhost:8080 \
  --probe-file probes.jsonl --lengths 64,128,256,257,512
```

Use `--skip-length-analysis` for a fast corpus-only check. Reports include the
selected batch sizes and lengths plus a SHA-256 fingerprint of the exact corpus.

Exit codes are:

- `0`: configured parity checks passed
- `1`: a meaningful parity check failed
- `2`: execution or configuration failed

## What is measured

The built-in deterministic corpus contains exactly 100 probes covering short and
paragraph text, semantic and unrelated pairs, queries and documents, code, SQL,
numbers, punctuation, whitespace, Unicode, accented text, six non-English
languages, and empty or near-empty strings. Long probes are separate: they are
created through the reference tokenizer at 32, 64, 128, 256, 384, 512, 768, and
1024 tokens. No LLM is used to generate any probe.

The comparison has four layers:

1. Structural checks validate counts, dimensions, finite values, zero vectors,
   and norm distributions. A dimension mismatch stops similarity analysis.
2. Vector checks report mean, median, minimum, p01, and p05 cross-runtime cosine,
   plus the ten worst probes.
3. Geometry checks compare the upper triangles of pairwise cosine matrices with
   Pearson, Spearman, mean absolute difference, and maximum difference.
4. Neighbor checks measure top-1 agreement and average top-5/top-10 overlap and
   identify the most changed neighborhoods.

Both runtimes are rerun at every requested batch size. For E5 and English BGE
models, query probes also test the model-family-recommended prefix. This is a
diagnostic observation only: the tool says a prefix *may* explain a difference;
it does not claim an unobserved server configuration as fact.

TEI metadata endpoints vary by server version. The adapter tries `/info` and `/`,
records the complete JSON object, and promotes recognized model, revision, dtype,
dimension, and input-length fields. Known model or revision conflicts fail the
comparison. Unknown fields remain unknown rather than being inferred; a known
dtype difference is diagnostic but does not fail otherwise-equivalent vectors.

## Tests and broken fixtures

```bash
pytest
```

The deterministic fake backends and mocked HTTP adapter prove detection of all requested classes:
equivalent output, missing normalization, dimensional mismatch, small noise,
large perturbation, a 256-token truncation boundary, batch dependence, hidden
query prefix, NaN output, an incompatible embedding space, malformed/ragged TEI
responses, wrong response counts, metadata conflicts, and JSONL validation.

## Real-model experiment

The repository includes an opt-in three-model integration test for:

- `BAAI/bge-small-en-v1.5`
- `intfloat/e5-small-v2`
- `sentence-transformers/all-MiniLM-L6-v2`

Provide three live endpoints and run:

```bash
TEI_BGE_SMALL_URL=http://localhost:8081 \
TEI_E5_SMALL_URL=http://localhost:8082 \
TEI_MINILM_URL=http://localhost:8083 \
pytest -m integration -vv
```

No live TEI endpoint was available in the repository build environment, so no
real-model result is claimed here. The integration test records actual behavior
and fails instead of manufacturing a discrepancy. The unit suite then introduces
each configuration difference deliberately and verifies that it is caught.

## Scope and interpretation

This is not a general backend framework. It contains exactly two production
adapters: SentenceTransformers and TEI. A passing result means the selected probe
suite and thresholds found the runtimes operationally equivalent; it is not a
proof that every possible input will match. Diagnostics intentionally use phrases
such as “possible truncation mismatch” because measurements often narrow the
cause without proving the server's internal configuration.
