from __future__ import annotations

import argparse
import os
import sys

from . import __version__
from .backends.sentence_transformers import SentenceTransformersBackend
from .backends.tei import TEIBackend
from .compare import Thresholds, compare_backends
from .probes import DEFAULT_LENGTHS, built_in_probes, load_jsonl_probes
from .report import render_text, write_error_json, write_json


def _batch_sizes(value: str) -> tuple[int, ...]:
    try:
        sizes = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("batch sizes must be comma-separated integers") from exc
    if not sizes or any(size < 1 for size in sizes):
        raise argparse.ArgumentTypeError("batch sizes must be positive")
    if len(set(sizes)) != len(sizes):
        raise argparse.ArgumentTypeError("batch sizes must be unique")
    return sizes


def _lengths(value: str) -> tuple[int, ...]:
    lengths = _batch_sizes(value)
    if tuple(sorted(set(lengths))) != lengths:
        raise argparse.ArgumentTypeError("lengths must be unique and increasing")
    return lengths


def _header(value: str) -> tuple[str, str]:
    name, separator, header_value = value.partition("=")
    if not separator or not name.strip() or not header_value:
        raise argparse.ArgumentTypeError("headers must use NAME=VALUE")
    return name.strip(), header_value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="embed-parity", description="Compare SentenceTransformers and TEI embeddings"
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)
    compare = subparsers.add_parser("compare", help="run the parity comparison")
    compare.add_argument("--model", required=True, help="Hugging Face model identifier")
    compare.add_argument("--tei", required=True, help="TEI base URL")
    compare.add_argument("--revision", help="SentenceTransformers model revision")
    compare.add_argument("--device", help="SentenceTransformers device (cpu, cuda, mps)")
    compare.add_argument("--dtype", choices=("float16", "float32", "float64", "bfloat16"))
    norm = compare.add_mutually_exclusive_group()
    norm.add_argument("--normalize", action="store_true", help="normalize reference embeddings")
    norm.add_argument("--no-normalize", action="store_true", help="disable reference normalization")
    compare.add_argument("--batch-sizes", type=_batch_sizes, default=(1, 8, 32))
    compare.add_argument(
        "--lengths", type=_lengths, default=DEFAULT_LENGTHS, help="comma-separated token lengths"
    )
    compare.add_argument(
        "--skip-length-analysis", action="store_true", help="skip tokenizer-controlled long probes"
    )
    compare.add_argument(
        "--probe-file", metavar="JSONL", help="replace the built-in corpus with JSONL probes"
    )
    compare.add_argument(
        "--timeout", type=float, default=120.0, help="TEI request timeout in seconds"
    )
    compare.add_argument(
        "--readiness-timeout", type=float, default=30.0, help="seconds to wait for TEI health"
    )
    compare.add_argument(
        "--tei-retries", type=int, default=2, help="retries for transient TEI failures"
    )
    compare.add_argument(
        "--tei-retry-backoff", type=float, default=0.25, help="initial retry backoff in seconds"
    )
    compare.add_argument(
        "--tei-api-key",
        default=os.getenv("EMBED_PARITY_TEI_API_KEY"),
        help="TEI bearer token (or set EMBED_PARITY_TEI_API_KEY)",
    )
    compare.add_argument(
        "--tei-header", type=_header, action="append", default=[], metavar="NAME=VALUE"
    )
    truncation = compare.add_mutually_exclusive_group()
    truncation.add_argument(
        "--tei-truncate", dest="tei_truncate", action="store_true", default=True
    )
    truncation.add_argument("--no-tei-truncate", dest="tei_truncate", action="store_false")
    compare.add_argument(
        "--json", metavar="PATH", help="also write the complete machine-readable report"
    )
    compare.add_argument("--vector-mean", type=float, default=Thresholds.vector_mean)
    compare.add_argument("--vector-min", type=float, default=Thresholds.vector_min)
    compare.add_argument("--geometry-spearman", type=float, default=Thresholds.geometry_spearman)
    compare.add_argument("--geometry-mae", type=float, default=Thresholds.geometry_mae)
    compare.add_argument("--top1-agreement", type=float, default=Thresholds.top1_agreement)
    compare.add_argument("--top5-overlap", type=float, default=Thresholds.top5_overlap)
    compare.add_argument("--top10-overlap", type=float, default=Thresholds.top10_overlap)
    compare.add_argument("--batch-min", type=float, default=Thresholds.batch_min)
    compare.add_argument(
        "--norm-relative-difference", type=float, default=Thresholds.norm_relative_difference
    )
    compare.add_argument("--prefix-improvement", type=float, default=Thresholds.prefix_improvement)
    return parser


def run_compare(args: argparse.Namespace) -> int:
    normalize = True if args.normalize else False if args.no_normalize else None
    if args.timeout <= 0 or args.readiness_timeout <= 0:
        raise ValueError("timeouts must be positive")
    if args.tei_retries < 0 or args.tei_retry_backoff < 0:
        raise ValueError("TEI retries and retry backoff cannot be negative")
    thresholds = Thresholds(
        vector_mean=args.vector_mean,
        vector_min=args.vector_min,
        geometry_spearman=args.geometry_spearman,
        geometry_mae=args.geometry_mae,
        top1_agreement=args.top1_agreement,
        top5_overlap=args.top5_overlap,
        top10_overlap=args.top10_overlap,
        batch_min=args.batch_min,
        norm_relative_difference=args.norm_relative_difference,
        prefix_improvement=args.prefix_improvement,
    )
    candidate = TEIBackend(
        args.tei,
        timeout=args.timeout,
        api_key=args.tei_api_key,
        headers=dict(args.tei_header),
        truncate=args.tei_truncate,
        retries=args.tei_retries,
        retry_backoff=args.tei_retry_backoff,
    )
    try:
        candidate.wait_until_ready(args.readiness_timeout)
        reference = SentenceTransformersBackend(
            args.model,
            revision=args.revision,
            device=args.device,
            dtype=args.dtype,
            normalize=normalize,
        )
        probes = load_jsonl_probes(args.probe_file) if args.probe_file else built_in_probes()
        report = compare_backends(
            args.model,
            reference,
            candidate,
            probes,
            batch_sizes=args.batch_sizes,
            thresholds=thresholds,
            length_factory=None if args.skip_length_analysis else reference.text_at_token_length,
            lengths=args.lengths,
        )
        print(render_text(report), end="")
        if args.json:
            write_json(report, args.json)
        return 0 if report["passed"] else 1
    finally:
        candidate.close()


def _write_execution_error(args: argparse.Namespace, error: BaseException) -> None:
    path = getattr(args, "json", None)
    if not path:
        return
    try:
        write_error_json(
            path,
            model=getattr(args, "model", None),
            tei=getattr(args, "tei", None),
            error=error,
        )
    except OSError as report_error:
        print(f"embed-parity: could not write error report: {report_error}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return run_compare(args)
    except KeyboardInterrupt:
        _write_execution_error(args, KeyboardInterrupt())
        print("embed-parity: interrupted", file=sys.stderr)
        return 2
    except Exception as exc:
        _write_execution_error(args, exc)
        print(f"embed-parity: execution/configuration failure: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
