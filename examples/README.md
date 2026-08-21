# Examples

Workload retrieval comparison:

```bash
embed-parity workload --model BAAI/bge-small-en-v1.5 \
  --tei http://localhost:8080 --input search-workload.jsonl \
  --json workload-report.json --html workload-report.html
```

The workload must label every row as `query` or `document`. The report compares
each runtime's top-k documents without claiming which ranking is better.

Basic comparison:

```bash
embed-parity compare --model sentence-transformers/all-MiniLM-L6-v2 \
  --tei http://localhost:8080 --json minilm-report.json
```

Explicitly compare normalized SentenceTransformers output:

```bash
embed-parity compare --model intfloat/e5-small-v2 \
  --tei http://localhost:8080 --normalize --batch-sizes 1,8,32
```

Relaxing a threshold is a domain decision and is visible in the saved report:

```bash
embed-parity compare --model BAAI/bge-small-en-v1.5 \
  --tei http://localhost:8080 --vector-min 0.985 --geometry-mae 0.015
```
