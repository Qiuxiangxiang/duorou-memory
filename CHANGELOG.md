# duorou changelog

## 1.0.0 (implementation release, 2026-11-24)

- Consolidated local memory engine using existing SQLite schema; write/read grants, atomic batch input, immutable evidence, source supersession, scoped cache invalidation, deletion and source-backed entities, claims, events and task working sets.
- Pure Python scope-local BM25, metadata/context indexes, RRF, bounded graph links, optional local dense embedding and evidence budget.
- Conservative bilingual rule extraction with idempotent candidates; optional explicit extractor callback.
- 17 local stdio MCP tools; protocol runtime awaits dependency installation.
- New same-budget evidence evaluation and synthetic load tests. Public-corpus fresh runs and competitor end-to-end comparisons pending.

Historical v0.2 and v0.3 reports retain original results. No claim of public SOTA or production readiness.
