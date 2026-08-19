# TEI issue #882: equal-length concurrent batching

This experiment reproduces the open
[TEI issue #882](https://github.com/huggingface/text-embeddings-inference/issues/882):
Qwen3 and Gemma3 Candle backends can omit their causal attention mask when the
router coalesces equal-length requests into a backend batch.

## Pinned inputs

- Model: `Qwen/Qwen3-Embedding-0.6B`
- Model revision: `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`
- Official image: `ghcr.io/huggingface/text-embeddings-inference:cpu-1.9.3`
- Official image digest: `sha256:ad950d30878eceb72aaf32024d26fa2b1d04a75304fa0b4776b49aa1941fea07`
- Proposed fix: [TEI PR #883](https://github.com/huggingface/text-embeddings-inference/pull/883)
- Community-built patched image digest: `sha256:4ae60ded45ee9060ce7bed6816e197099750156d62deded4b2d251a75aa167d5`

The patched image is provided by the issue/PR author and is evidence for this
local comparison, not an official Hugging Face release or a recommended
production artifact.

Both containers used `--max-batch-tokens 64` because the test ran the AMD64 CPU
image through emulation on an ARM64 Mac. The equal-length probes are eight tokens
each, so they do not approach that limit. Both servers forced a maximum backend
batch size of four during warmup.

## Result

| Request path | Official 1.9.3 | PR #883 patched image |
|---|---:|---:|
| Client list batch | 1.0 / 1.0 | 1.0 / 1.0 |
| Unequal-length control | 1.0 / 1.0 | 1.0 / 1.0 |
| Four concurrent single-input requests | 1.0 / **0.1586** / 1.0 / **0.1586** | 1.0 / 1.0 / 1.0 / 1.0 |

Each number is cosine similarity between an embedding returned under the named
request pattern and that text's embedding from an isolated request. The official
image silently corrupts embeddings for two requests only when the router batches
independent equal-length requests. The patched image eliminates the divergence.

Raw outputs are in `official-1.9.3.json` and `patched-pr-883.json`.

The v0.3 implementation was then run directly with one identical text repeated
across four independent requests for three trials. It detected the regression in
every trial: 6 of 12 responses diverged, with minimum cosine `0.266183`. See
`embed-parity-v0.3.json`. This score differs from the alternating-text reproducer
because the affected embedding is input-dependent; both compare each text only
against its own isolated embedding.

## Why this matters to embed-parity

The current TEI adapter implements batch sizes by placing multiple inputs in one
client `/embed` request. That path scored 1.0 and did not reproduce this bug.
Production-style concurrent single-input requests did reproduce it. Therefore,
the current batch-consistency check has a real blind spot: it tests client-side
list batching, but not router-coalesced server batching.

Run the probe against a ready server with:

```bash
python reproduce.py --tei http://127.0.0.1:8282
```
