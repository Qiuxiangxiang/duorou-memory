#!/usr/bin/env python3
"""Import pinned, public LoCoMo RAW conversation data into the v0.2 SQLite schema.

This is a storage/constraint/query-plan demonstration, NOT a retrieval benchmark
and NOT an MCP server. Authentication must be performed by the hosting service.
No question, answer, evidence gold label, summary, observation or extracted fact
is imported. Only raw dialogue text enters FTS; image captions remain metadata.

Example, from the project root (Python 3.10+, SQLite 3.37+ with FTS5/JSON):
    python examples/sqlite_demo.py --locomo /path/to/locomo10.json \
        --db reports/demo.sqlite --report reports/sqlite_import_report.json
The database path must not already exist; no existing database is overwritten.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCOMO_SHA256 = "79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4"
IMPORTER_VERSION = "locomo-raw-import/0.2"
TABLES = (
    "principals", "scopes", "scope_grants", "entities", "sources",
    "evidence_spans", "entity_mentions", "entity_aliases", "events",
    "event_memberships", "claims", "claim_evidence", "claim_relations",
    "intentions", "tag_definitions", "tag_annotations", "working_sets",
    "working_set_members",
)

# These tags describe observed source structure, not inferred autobiographical facts.
TAG_DEFINITIONS = (
    ("source", "kind", "enum", '["dialogue_turn"]', "Observed source record type"),
    ("event", "boundary_kind", "enum", '["source_session"]', "Structural session boundary, not semantic event detection"),
    ("chronology", "session_ordinal", "number", None, "Order of sessions within the source conversation"),
)


def connect_database(path: str | Path = ":memory:") -> sqlite3.Connection:
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    if conn.execute("PRAGMA foreign_keys").fetchone()[0] != 1:
        raise RuntimeError("foreign key enforcement is required")
    return conn


def initialize_schema(conn: sqlite3.Connection) -> None:
    conn.executescript((PROJECT_ROOT / "schema" / "schema.sql").read_text(encoding="utf-8"))
    conn.execute("PRAGMA user_version = 2")


def insert_source(conn: sqlite3.Connection, *, scope_id: str, source_id: str,
                  speaker_entity_id: str | None, content: str, recorded_at_us: int,
                  source_time_raw: str | None = None,
                  metadata: dict[str, Any] | None = None,
                  source_system: str = "locomo", source_kind: str = "dialogue_turn",
                  external_key: str | None = None) -> None:
    """Trusted ingestion helper. SHA consistency is checked here, not by SQLite."""
    if not isinstance(content, str):
        raise TypeError("source content must be a string")
    conn.execute(
        """INSERT INTO sources (
               scope_id, source_id, source_system, source_kind, external_key,
               speaker_entity_id, content, content_sha256, recorded_at_us,
               source_time_raw, metadata_json
           ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (scope_id, source_id, source_system, source_kind, external_key or source_id,
         speaker_entity_id, content, hashlib.sha256(content.encode("utf-8")).hexdigest(),
         recorded_at_us, source_time_raw, json.dumps(metadata or {}, ensure_ascii=False)),
    )


def require_read(conn: sqlite3.Connection, *, authenticated_principal: str,
                 scope_id: str) -> None:
    """Apply an allowlist for a principal ALREADY authenticated by the server.

    Never populate authenticated_principal from an MCP argument or model-provided
    identity. This prototype implements no login, token validation or RLS. Any
    process with direct database access can bypass this Python function.
    """
    allowed = conn.execute(
        """SELECT 1 FROM scope_grants WHERE principal_id = ? AND scope_id = ?
           AND permission IN ('read', 'admin') LIMIT 1""",
        (authenticated_principal, scope_id),
    ).fetchone()
    if allowed is None:
        raise PermissionError("principal is not authorized to read the requested scope")


SEARCH_SQL = """SELECT s.source_id, s.speaker_entity_id,
       snippet(source_fts, 0, '[', ']', ' … ', 18) AS snippet,
       bm25(source_fts) AS fts_score
FROM source_fts JOIN sources s ON s.source_rowid = source_fts.rowid
WHERE source_fts MATCH ? AND s.scope_id = ?
  AND EXISTS (SELECT 1 FROM scope_grants g
      WHERE g.scope_id = s.scope_id AND g.principal_id = ?
        AND g.permission IN ('read', 'admin'))
ORDER BY fts_score, s.source_id LIMIT ?"""


def literal_fts_query(query: str) -> str:
    """Treat user input as literal tokens, not arbitrary FTS query syntax."""
    if not isinstance(query, str) or len(query) > 2048:
        raise ValueError("query must be a string of at most 2048 characters")
    tokens = re.findall(r"\w+", query, flags=re.UNICODE)
    if not tokens or len(tokens) > 64:
        raise ValueError("query must contain between 1 and 64 word tokens")
    return " OR ".join('"' + token.replace('"', '""') + '"' for token in tokens)


def search_sources(conn: sqlite3.Connection, *, authenticated_principal: str,
                   scope_id: str, query: str, limit: int = 5) -> list[dict[str, Any]]:
    """Permission-gated raw-source lookup, not a complete production API.

    FTS BM25 corpus statistics are global in this schema. Scope filtering prevents
    returning other scopes through this function, not timing/statistical leakage.
    The EXISTS predicate repeats authorization in the data-reading SQL so grant
    revocation after the precheck cannot expose rows through this query.
    """
    require_read(conn, authenticated_principal=authenticated_principal, scope_id=scope_id)
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 100:
        raise ValueError("limit must be an integer between 1 and 100")
    return [dict(row) for row in conn.execute(
        SEARCH_SQL, (literal_fts_query(query), scope_id, authenticated_principal, limit)
    )]


def _tag(conn: sqlite3.Connection, *, scope_id: str, annotation_id: str,
         namespace: str, tag_key: str, value: Any,
         source_id: str | None = None, event_id: str | None = None,
         evidence_span_id: str | None = None) -> None:
    conn.execute(
        """INSERT INTO tag_annotations (
               scope_id, annotation_id, namespace, tag_key, value_json,
               source_id, event_id, evidence_span_id, origin, status, extractor_version
           ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'metadata', 'accepted', ?)""",
        (scope_id, annotation_id, namespace, tag_key,
         json.dumps(value, ensure_ascii=False, separators=(",", ":")),
         source_id, event_id, evidence_span_id, IMPORTER_VERSION),
    )


def import_locomo_records(conn: sqlite3.Connection, records: list[dict[str, Any]],
                         *, recorded_at_us: int) -> list[str]:
    """Import raw conversation fields by allowlist; caller owns the transaction.

    This helper also permits tiny fixtures for constraint tests. The CLI always
    checks the public dataset's pinned SHA before calling it. Fixture tests are
    not public-dataset benchmark results.
    """
    conn.executemany("INSERT INTO tag_definitions VALUES (?, ?, ?, ?, ?)", TAG_DEFINITIONS)
    scopes: list[str] = []
    for sample in records:
        sample_id = sample["sample_id"]
        conversation = sample["conversation"]
        if not isinstance(sample_id, str) or not isinstance(conversation, dict):
            raise ValueError("invalid conversation structure")
        scope_id = "locomo:" + sample_id
        scopes.append(scope_id)
        conn.execute("INSERT INTO scopes VALUES (?, ?)", (scope_id, sample_id))
        sessions = []
        for key, turns in conversation.items():
            match = re.fullmatch(r"session_(\d+)", key)
            if match:
                if not isinstance(turns, list):
                    raise ValueError("session must be a list")
                sessions.append((int(match.group(1)), key, turns))
        sessions.sort(key=lambda item: item[0])
        # Speaker labels are dataset metadata bindings, not resolved human identities.
        names = {conversation[k] for k in ("speaker_a", "speaker_b") if k in conversation}
        names.update(turn["speaker"] for _, _, turns in sessions for turn in turns)
        if any(not isinstance(name, str) for name in names):
            raise ValueError("speaker metadata must be text")
        speakers = {name: f"speaker_{i:03d}" for i, name in enumerate(sorted(names), 1)}
        for name, entity_id in speakers.items():
            conn.execute("INSERT INTO entities VALUES (?, ?, 'person', ?, 'source_bound')",
                         (scope_id, entity_id, name))
        for session_ordinal, session_key, turns in sessions:
            raw_date = conversation.get(session_key + "_date_time")
            if raw_date is not None and not isinstance(raw_date, str):
                raise ValueError("session date metadata must be text or null")
            # No timezone is asserted by LoCoMo's string. Do not manufacture UTC.
            conn.execute(
                """INSERT INTO events (
                       scope_id, event_id, kind, title, boundary_origin, status,
                       session_ordinal, time_precision, source_time_raw
                   ) VALUES (?, ?, 'source_session', ?, 'source_metadata',
                       'provisional', ?, ?, ?)""",
                (scope_id, session_key, session_key, session_ordinal,
                 "source_local_unspecified" if raw_date else "unknown", raw_date),
            )
            _tag(conn, scope_id=scope_id, annotation_id=session_key + ":boundary",
                 namespace="event", tag_key="boundary_kind", value="source_session", event_id=session_key)
            _tag(conn, scope_id=scope_id, annotation_id=session_key + ":ordinal",
                 namespace="chronology", tag_key="session_ordinal", value=session_ordinal, event_id=session_key)
            for position, turn in enumerate(turns):
                # Deliberate field allowlist: never copy the whole sample or turn.
                source_id, speaker, content = turn["dia_id"], turn["speaker"], turn["text"]
                if not isinstance(source_id, str) or not isinstance(content, str) or not content:
                    raise ValueError("dialogue IDs and nonempty raw text are required")
                metadata = {"dia_id": source_id, "speaker": speaker, "session_key": session_key}
                for optional_key in ("blip_caption", "img_url"):
                    if optional_key in turn:
                        metadata[optional_key] = turn[optional_key]
                insert_source(conn, scope_id=scope_id, source_id=source_id,
                              speaker_entity_id=speakers[speaker], content=content,
                              recorded_at_us=recorded_at_us, source_time_raw=raw_date,
                              metadata=metadata, external_key=f"{sample_id}/{source_id}")
                span_id = source_id + ":full"
                conn.execute("INSERT INTO evidence_spans VALUES (?, ?, ?, 0, ?)",
                             (scope_id, span_id, source_id, len(content)))
                conn.execute("INSERT INTO event_memberships VALUES (?, ?, ?, ?)",
                             (scope_id, session_key, span_id, position))
                _tag(conn, scope_id=scope_id, annotation_id=source_id + ":kind",
                     namespace="source", tag_key="kind", value="dialogue_turn",
                     source_id=source_id, evidence_span_id=span_id)
    return scopes


def _plan(conn: sqlite3.Connection, sql: str, parameters: tuple[Any, ...]) -> dict[str, Any]:
    return {
        "sql": sql,
        "parameters": list(parameters),
        "explain_query_plan": [dict(row) for row in conn.execute("EXPLAIN QUERY PLAN " + sql, parameters)],
        "result_preview": [dict(row) for row in conn.execute(sql, parameters).fetchmany(5)],
    }


def make_report(conn: sqlite3.Connection, *, scopes: list[str], raw_path: Path,
                raw_sha256: str, raw_bytes: int, recorded_at_us: int) -> dict[str, Any]:
    principal, scope_id = "demo-reader", scopes[0]
    # This is a local, explicitly bootstrapped demo identity, not an authentication system.
    conn.execute("INSERT INTO principals VALUES (?, 'service')", (principal,))
    conn.execute("INSERT INTO scope_grants VALUES (?, ?, 'read')", (scope_id, principal))
    authorized = search_sources(conn, authenticated_principal=principal,
                                scope_id=scope_id, query="support", limit=5)
    unauthorized_scope = scopes[1]
    try:
        search_sources(conn, authenticated_principal=principal,
                       scope_id=unauthorized_scope, query="support")
    except PermissionError as exc:
        denied = {"scope_id": unauthorized_scope, "denied": True,
                  "exception": type(exc).__name__, "message": str(exc)}
    else:
        raise AssertionError("unauthorized read was not rejected")

    require_read(conn, authenticated_principal=principal, scope_id=scope_id)
    # Additional SQL below runs as the trusted local inspection process after the
    # same authorization precheck; it is not exposed as a generic remote SQL tool.
    plans = {
        "authorized_fts": _plan(conn, SEARCH_SQL, (literal_fts_query("support"), scope_id, principal, 5)),
        "source_by_entity_and_recorded_time": _plan(
            conn,
            "SELECT source_id FROM sources WHERE scope_id = ? AND speaker_entity_id = ? "
            "AND recorded_at_us >= ? ORDER BY recorded_at_us, source_id LIMIT 5",
            (scope_id, "speaker_001", recorded_at_us),
        ),
        "controlled_tag_lookup": _plan(
            conn,
            "SELECT source_id FROM tag_annotations WHERE scope_id = ? AND namespace = ? "
            "AND tag_key = ? AND value_json = ? AND status = 'accepted' LIMIT 5",
            (scope_id, "source", "kind", '"dialogue_turn"'),
        ),
        "entity_lookup": _plan(
            conn, "SELECT entity_id, identity_state FROM entities WHERE scope_id = ? AND label = ?",
            (scope_id, conn.execute("SELECT label FROM entities WHERE scope_id = ? ORDER BY entity_id LIMIT 1", (scope_id,)).fetchone()[0]),
        ),
        "structural_session_range": _plan(
            conn,
            "SELECT event_id, session_ordinal, source_time_raw FROM events WHERE scope_id = ? "
            "AND session_ordinal BETWEEN ? AND ? ORDER BY session_ordinal, event_id",
            (scope_id, 1, 3),
        ),
        "known_absolute_event_time_range": _plan(
            conn,
            "SELECT event_id FROM events WHERE scope_id = ? AND time_start_us >= ? "
            "AND time_start_us < ? ORDER BY time_start_us LIMIT 5",
            (scope_id, 0, recorded_at_us),
        ),
    }
    fk_errors = [dict(row) for row in conn.execute("PRAGMA foreign_key_check")]
    integrity = conn.execute("PRAGMA integrity_check").fetchone()[0]
    conn.execute("INSERT INTO source_fts(source_fts, rank) VALUES ('integrity-check', 1)")
    counts = {table: conn.execute(f'SELECT count(*) FROM "{table}"').fetchone()[0] for table in TABLES}
    if (counts["sources"], counts["events"], counts["scopes"]) != (5882, 272, 10):
        raise AssertionError("pinned LoCoMo import row counts differ from expected raw data")
    return {
        "artifact_kind": "SQLite data-layer import and constraint demonstration; not retrieval benchmark",
        "schema_version": 2,
        "sqlite_version": sqlite3.sqlite_version,
        "dataset": {"name": "LoCoMo locomo10.json", "input_filename": raw_path.name,
                    "sha256": raw_sha256, "bytes": raw_bytes, "sha256_verified": True},
        "importer_version": IMPORTER_VERSION,
        "recorded_at_us": recorded_at_us,
        "row_counts": counts,
        "indexes": [dict(row) for row in conn.execute(
            "SELECT name, tbl_name AS table_name, sql FROM sqlite_schema "
            "WHERE type = 'index' AND sql IS NOT NULL ORDER BY name")],
        "import_policy": {
            "sample_fields_consumed": ["sample_id", "conversation"],
            "conversation_fields_consumed": ["speaker_a", "speaker_b", "session_N", "session_N_date_time"],
            "turn_fields_consumed": ["dia_id", "speaker", "text", "blip_caption", "img_url"],
            "fts_fields": ["conversation.session_N[].text"],
            "metadata_only_fields": ["blip_caption", "img_url", "speaker", "dia_id", "session_key"],
            "not_imported": ["qa", "answer", "adversarial_answer", "evidence gold labels",
                             "event_summary", "observation", "session_summary"],
            "generated_claims": False,
            "semantic_event_segmentation": False,
            "entity_identity_semantics": "source speaker metadata bindings only; no verified real-world identity or automatic alias merge",
            "time_semantics": "recorded_at_us is import time; dataset local date strings are preserved without inventing timezone/UTC; event absolute timestamps remain NULL",
            "gold_label_leakage_control": "explicit field allowlist; no benchmark queries, answers or gold evidence participate in indexing",
        },
        "authorization_demo": {"principal": principal, "trusted_identity_source": "local demo bootstrap",
                               "authorized_scope": scope_id, "query": "support", "results": authorized,
                               "unauthorized_request": denied},
        "query_examples": plans,
        "integrity": {"foreign_key_check": fk_errors, "integrity_check": integrity,
                      "fts_external_content_integrity_check": "passed"},
        "limits": [
            "No MCP transport or authentication is implemented; authenticated_principal must come from a trusted server context.",
            "SQLite has no RLS; direct database access bypasses the application authorization check.",
            "FTS statistics and index structures are shared across scopes; this is not strict tenant statistical/timing isolation.",
            "unicode61 is not a Chinese word segmenter; Chinese retrieval requires a separately evaluated tokenizer/index.",
            "Tags in this import are deterministic source metadata, not inferred facts. Claims/mentions/aliases/intentions remain empty.",
            "SHA equality is computed by ingestion code; SQL validates SHA format only.",
            "Span ranges, types and provenance links do not prove that a claim, alias, tag or completion statement is true.",
            "Deleting a source cascades through linked spans and primary-evidence claims and updates FTS; it does not erase backups/WAL/exports/external caches.",
            "Deleting optional evidence removes that link; reconsideration of remaining derived content is an application responsibility.",
            "Event cycles beyond a direct self-reference, graph reasoning, extraction, merge decisions, retention and working-set token budgets are application responsibilities.",
            "No retrieval quality, latency, concurrency or cognitive advantage is established by this import/query-plan demo.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--locomo", type=Path, required=True, help="Pinned public locomo10.json file")
    parser.add_argument("--db", type=Path, default=PROJECT_ROOT / "reports" / "demo.sqlite")
    parser.add_argument("--report", type=Path, default=PROJECT_ROOT / "reports" / "sqlite_import_report.json")
    args = parser.parse_args()
    raw_bytes = args.locomo.read_bytes()
    actual_sha = hashlib.sha256(raw_bytes).hexdigest()
    if actual_sha != LOCOMO_SHA256:
        raise SystemExit(f"Dataset SHA256 mismatch: expected {LOCOMO_SHA256}, got {actual_sha}")
    records = json.loads(raw_bytes)
    if args.db.exists():
        raise SystemExit(f"Refusing to overwrite an existing database: {args.db}")
    args.db.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    recorded_at_us = time.time_ns() // 1000
    conn = connect_database(args.db)
    try:
        initialize_schema(conn)
        with conn:
            scopes = import_locomo_records(conn, records, recorded_at_us=recorded_at_us)
            report = make_report(conn, scopes=scopes, raw_path=args.locomo,
                                 raw_sha256=actual_sha, raw_bytes=len(raw_bytes),
                                 recorded_at_us=recorded_at_us)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    finally:
        conn.close()
    print(json.dumps({"database": str(args.db), "report": str(args.report),
                      "row_counts": report["row_counts"], "integrity": report["integrity"],
                      "unauthorized_scope_rejected": report["authorization_demo"]["unauthorized_request"]["denied"]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
