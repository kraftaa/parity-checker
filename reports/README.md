# Real runtime experiments

These experiments were run on August 19, 2026 using:

- SentenceTransformers 6.0.0 with Transformers 5.15.0 and Torch 2.13.0 on CPU
- Text Embeddings Inference 1.9.3 using the Apple Metal backend
- SentenceTransformers float32 versus TEI float16
- 97 accepted corpus probes; three empty/whitespace-only probes were recorded as
  skipped because TEI rejects empty inputs
- batch sizes 1, 8, and 32
- controlled input lengths 32, 64, 128, 256, 384, 512, 768, and 1024 tokens

No discrepancy was manufactured in the baseline runs. All three passed the
configured vector, geometry, nearest-neighbor, batch, norm, pooling, and length
checks.

| Model | Result | Mean cosine | Minimum | Geometry Spearman | Top-1 | Top-5 | Top-10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `BAAI/bge-small-en-v1.5` | PASS | 0.999998 | 0.999996 | 0.999996 | 98.97% | 99.38% | 99.79% |
| `intfloat/e5-small-v2` | PASS | 0.999997 | 0.999993 | 0.999989 | 96.91% | 99.79% | 99.90% |
| `sentence-transformers/all-MiniLM-L6-v2` | PASS | 0.999998 | 0.999994 | 0.999997 | 100.00% | 100.00% | 99.79% |

The observed float32/float16 difference was reported as context but did not
produce a meaningful parity failure. All length points passed, including inputs
beyond each model's configured maximum where both runtimes applied their normal
truncation behavior.

## Real negative control

MiniLM normally uses mean pooling. TEI was restarted with `--pooling cls` while
the SentenceTransformers reference remained on mean pooling. The checker failed
with exit code 1 and explicitly reported the known pooling mismatch.

| Configuration | Result | Mean cosine | Geometry Spearman | Geometry MAE | Top-1 | Top-10 |
|---|---:|---:|---:|---:|---:|---:|
| MiniLM, reference `mean`, TEI `cls` | FAIL | 0.486301 | 0.345629 | 0.507849 | 58.76% | 51.86% |

Machine-readable reports:

- [`bge-small-en-v1.5.json`](bge-small-en-v1.5.json)
- [`e5-small-v2.json`](e5-small-v2.json)
- [`all-MiniLM-L6-v2.json`](all-MiniLM-L6-v2.json)
- [`all-MiniLM-L6-v2-pooling-cls.json`](all-MiniLM-L6-v2-pooling-cls.json)

TEI did not expose its resolved model commit SHA through `/info`, so revision
compatibility is recorded as unknown rather than assumed. The reference commit
SHAs and complete server metadata remain available in each JSON report.
