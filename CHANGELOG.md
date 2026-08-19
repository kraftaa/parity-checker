# Changelog

All notable changes to this project are documented here.

## 0.2.0 - 2026-08-19

### Added

- Real pinned-revision experiment support and saved TEI comparison reports.
- Query and document prefix diagnostics for model families such as E5.
- TEI bearer authentication, custom headers, readiness checks, configurable
  truncation, and retry/backoff controls.
- Machine-readable JSON reports for execution and configuration failures.
- Versioned report schema v2 with the producing tool version.
- Explicit pooling metadata comparison and diagnostics.
- Scheduled/manual live-TEI GitHub Actions workflow.
- Ruff, mypy, branch coverage, wheel-build, and release quality gates.
- Tag-driven GitHub Release and PyPI trusted-publishing workflow.

### Changed

- Empty probes unsupported by TEI are recorded as skipped rather than aborting a
  comparison.
- Real baseline and negative-control reports now include dtype and pooling context.

## 0.1.0 - 2026-08-18

- Initial SentenceTransformers-versus-TEI parity checker.
