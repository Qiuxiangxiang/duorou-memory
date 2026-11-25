"""Read-only experimental retrieval adapters for the duorou data layer.

The raw FTS arm calls the existing permission-gated lookup unchanged. Enriched
FTS and neighbor expansion are separately named experiments, not capabilities
retroactively attributed to v0.2.1. No evaluator labels are accepted here.

This is a single-process local benchmark service with a deployment-bound
principal. It does not authenticate remote users. Both FTS indexes use global
corpus statistics; filtering results is not statistical tenant isolation.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from collections import defaultdict
from .fusion import fuse
import sqlite3
import time
from typing import Any

from examples.sqlite_demo import literal_fts_query, require_read, search_sources
from vendor.public_memory_bench.retrieval import ScopedRetriever


METHODS = (
    "bm25_raw", "fts_raw", "bm25_enriched", "fts_enriched",
    "fts_enriched_neighbors", "fts_enriched_context",
)


def source_documents(conn: sqlite3.Connection, *, enriched: bool) -> list[dict[str, str]]:
    """Project trusted source rows only; never use claims, questions or gold.

    The enriched text is byte-for-byte compatible with the original public
    LoCoMo converter for the same raw data. Captions are provided dataset text,
    not new image analysis or model-generated memory extraction.
    """
    documents = []
    for row in conn.execute(
        "SELECT scope_id, source_id, content, source_time_raw, metadata_json "
        "FROM sources ORDER BY scope_id, source_id"
    ):
        text = row["content"]
        if enriched:
            metadata = json.loads(row["metadata_json"])
            speaker = metadata.get("speaker")
            if not isinstance(speaker, str):
                raise ValueError("enriched projection requires source speaker metadata")
            timestamp = row["source_time_raw"] or ""
            parts = [f"Session timestamp: {timestamp}", f"Speaker: {speaker}",
                     f"Text: {text}"]
            caption = metadata.get("blip_caption")
            if caption:
                if not isinstance(caption, str):
                    raise ValueError("image caption must be a string when nonempty")
                parts.append(f"Image caption: {caption}")
            text = "\n".join(parts)
        documents.append({"id": f"{row['scope_id']}:{row['source_id']}",
                          "scope": row["scope_id"], "text": text})
    return documents


def build_enriched_documents(conn: sqlite3.Connection) -> list[dict[str, str]]:
    """Public source-only projection helper for pre-evaluation equivalence checks."""
    return source_documents(conn, enriched=True)


ENRICHED_SEARCH_SQL = """SELECT s.source_id, bm25(enriched_fts) AS fts_score
FROM enriched_fts
JOIN source_db.sources s ON s.source_rowid = enriched_fts.rowid
WHERE enriched_fts MATCH ? AND s.scope_id = ?
  AND EXISTS (SELECT 1 FROM source_db.scope_grants g
    WHERE g.scope_id = s.scope_id AND g.principal_id = ?
      AND g.permission IN ('read', 'admin'))
ORDER BY fts_score, s.source_id LIMIT ?"""


NEIGHBOR_SQL = """SELECT DISTINCT next_span.source_id, e.event_id,
       CASE WHEN next_member.position < seed_member.position THEN 0 ELSE 1 END AS side
FROM evidence_spans seed_span
JOIN event_memberships seed_member
  ON seed_member.scope_id = seed_span.scope_id AND seed_member.span_id = seed_span.span_id
JOIN events e ON e.scope_id = seed_member.scope_id AND e.event_id = seed_member.event_id
JOIN event_memberships next_member
  ON next_member.scope_id = seed_member.scope_id AND next_member.event_id = seed_member.event_id
 AND next_member.position IN (seed_member.position - 1, seed_member.position + 1)
JOIN evidence_spans next_span
  ON next_span.scope_id = next_member.scope_id AND next_span.span_id = next_member.span_id
WHERE seed_span.scope_id = ? AND seed_span.source_id = ? AND e.kind = 'source_session'
  AND EXISTS (SELECT 1 FROM scope_grants g
    WHERE g.scope_id = seed_span.scope_id AND g.principal_id = ?
      AND g.permission IN ('read', 'admin'))
ORDER BY e.event_id, side, next_span.source_id"""


class DuorouRetriever:
    """Expose fixed read-only retrieval methods for an immutable test corpus.

    Neighbor policy: take canonical enriched top100; preserve its first10;
    append predecessor then successor of its first5 seeds within the same
    source-session event; deduplicate; fill from lexical top100; finally take
    the requested prefix. It never returns an event in place of its children.
    Rank-based scores in this arm express the heuristic order, not BM25 units.

    The corpus and materialized indexes are static for this benchmark. Permission
    changes are checked per request, but live source updates require rebuilding
    derived indexes and are outside this adapter's supported lifecycle.
    """

    def __init__(self, db_path: str | Path, method: str,
                 documents_path: str | Path | None = None,
                 principal: str = "benchmark-reader") -> None:
        if method not in METHODS:
            raise ValueError(f"unknown method: {method}")
        if not isinstance(principal, str) or not principal:
            raise ValueError("principal must be a nonempty deployment-bound string")
        started = time.perf_counter()
        path = Path(db_path).resolve(strict=True)
        uri = path.as_uri() + "?mode=ro"
        self.conn = sqlite3.connect(uri, uri=True)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.execute("PRAGMA query_only=ON")
        self.method, self.principal = method, principal
        self.index: sqlite3.Connection | None = None
        self.lexical: ScopedRetriever | None = None
        try:
            row = self.conn.execute(
                "SELECT count(*) AS documents, count(DISTINCT scope_id) AS scopes FROM sources"
            ).fetchone()
            self.build_stats: dict[str, Any] = {
                "method": method, "document_count": row["documents"],
                "scope_count": row["scopes"], "query_result_cache": False,
                "primary_database_read_only": True,
                "database_bytes": path.stat().st_size,
                "response": "document IDs and scores only",
                "principal_binding": "trusted local service configuration, not tool input",
                "live_ingestion_supported": False,
            }
            if method.startswith("bm25_"):
                if documents_path is None:
                    raise ValueError("BM25 control requires a source-only documents file")
                expected = source_documents(self.conn, enriched=method == "bm25_enriched")
                supplied = [json.loads(line) for line in
                            Path(documents_path).read_text(encoding="utf-8").splitlines() if line.strip()]
                if (len(supplied) != len(expected) or
                    sorted(supplied, key=lambda d: d["id"]) != sorted(expected, key=lambda d: d["id"])):
                    raise ValueError("BM25 documents must exactly match the declared source projection")
                self.lexical = ScopedRetriever(documents_path, method="bm25")
                self.build_stats["lexical_index"] = self.lexical.build_stats
            elif method != "fts_raw":
                documents = source_documents(self.conn, enriched=True)
                rows = self.conn.execute(
                    "SELECT source_rowid, scope_id, source_id FROM sources ORDER BY scope_id, source_id"
                ).fetchall()
                self.index = sqlite3.connect(":memory:", uri=True)
                self.index.row_factory = sqlite3.Row
                self.index.execute("ATTACH DATABASE ? AS source_db", (uri,))
                self.index.execute("CREATE VIRTUAL TABLE enriched_fts USING fts5(content, tokenize='unicode61')")
                self.index.executemany("INSERT INTO enriched_fts(rowid, content) VALUES (?, ?)",
                                       [(row["source_rowid"], doc["text"]) for row, doc in zip(rows, documents)])
                if method == "fts_enriched_context":
                    self._build_context(rows, documents)
                self.index.commit()
                digest = hashlib.sha256()
                for doc in documents:
                    digest.update((json.dumps(doc, ensure_ascii=False, sort_keys=True,
                                              separators=(",", ":")) + "\n").encode())
                self.build_stats.update({
                    "index_projection": ["source_time_raw", "speaker", "content", "blip_caption"],
                    "projection_canonical_sha256": digest.hexdigest(),
                    "index_storage": "in-memory SQLite FTS5; primary database unchanged",
                    "index_page_bytes": self.index.execute("PRAGMA page_size").fetchone()[0] *
                                        self.index.execute("PRAGMA page_count").fetchone()[0],
                })
            if method.startswith("fts_"):
                self.build_stats["fts_spec"] = {
                    "tokenizer": "unicode61", "match": "literal word tokens joined with OR",
                    "rank": "SQLite FTS5 bm25 default", "stopword_removal": False,
                    "scope_statistics": "global across all indexed scopes",
                    "tie_break": "local source_id ascending",
                    "existing_query_unchanged": method == "fts_raw",
                    "neighbor_expansion": method == "fts_enriched_neighbors",
                }
            self.build_stats["build_seconds"] = time.perf_counter() - started
        except BaseException:
            self.close()
            raise

    def _build_context(self, rows, documents):
        # Each window is a retrieval key for one leaf, not an expanded result.
        text = {r["source_rowid"]: d["text"] for r, d in zip(rows, documents)}
        groups = defaultdict(list)
        for r in self.conn.execute("""SELECT s.source_rowid, m.scope_id, m.event_id, m.position
            FROM event_memberships m JOIN events e USING(scope_id,event_id)
            JOIN evidence_spans p USING(scope_id,span_id)
            JOIN sources s ON s.scope_id=p.scope_id AND s.source_id=p.source_id
            WHERE e.kind='source_session' ORDER BY m.scope_id,m.event_id,m.position,s.source_rowid"""):
            groups[(r["scope_id"],r["event_id"])].append((r["position"],r["source_rowid"]))
        windows = {i: {i} for i in text}
        for members in groups.values():
            at = defaultdict(set)
            for pos, i in members:
                at[pos].add(i)
            for pos, i in members:
                windows[i].update(at.get(pos-1, ()))
                windows[i].update(at.get(pos+1, ()))
        self.index.execute("CREATE VIRTUAL TABLE context_fts USING fts5(content, tokenize='unicode61')")
        self.index.executemany("INSERT INTO context_fts(rowid,content) VALUES (?,?)",
            ((i, "\n".join(text[j] for j in sorted(ids))) for i, ids in windows.items()))
        self.build_stats["context_index"] = "source-session +/-1, leaf anchor only; RRF depth 100"

    def _enriched(self, query: str, limit: int, scope: str, *, context: bool = False) -> list[dict[str, Any]]:
        assert self.index is not None
        return [{"id": f"{scope}:{row['source_id']}", "score": -row["fts_score"]}
                for row in self.index.execute(
                    ENRICHED_SEARCH_SQL.replace("enriched_fts", "context_fts") if context else ENRICHED_SEARCH_SQL,
                    (literal_fts_query(query), scope, self.principal, limit))]

    def _permitted_hits(self, hits: list[dict[str, Any]], scope: str) -> list[dict[str, Any]]:
        """Repeat authorization at the boundary that releases in-memory results."""
        require_read(self.conn, authenticated_principal=self.principal, scope_id=scope)
        if not hits:
            return []
        prefix = scope + ":"
        ids = [hit["id"][len(prefix):] for hit in hits if hit["id"].startswith(prefix)]
        marks = ",".join("?" for _ in ids)
        if not ids:
            return []
        valid = {prefix + row[0] for row in self.conn.execute(
            "SELECT s.source_id FROM sources s WHERE s.scope_id=? AND s.source_id IN (" + marks + ") "
            "AND EXISTS (SELECT 1 FROM scope_grants g WHERE g.scope_id=s.scope_id "
            "AND g.principal_id=? AND g.permission IN ('read','admin'))",
            (scope, *ids, self.principal))}
        return [hit for hit in hits if hit["id"] in valid]

    def search(self, query: str, top_k: int, scope: str) -> list[dict[str, Any]]:
        if not isinstance(scope, str) or not scope:
            raise ValueError("scope must be a nonempty string")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= 100:
            raise ValueError("top_k must be an integer between 1 and 100")
        # A common safe request contract applies even though BM25 uses its own analyzer.
        literal_fts_query(query)
        require_read(self.conn, authenticated_principal=self.principal, scope_id=scope)
        if self.method.startswith("bm25_"):
            assert self.lexical is not None
            return self._permitted_hits(self.lexical.search(query, top_k, scope), scope)
        if self.method == "fts_raw":
            return [{"id": f"{scope}:{hit['source_id']}", "score": -hit["fts_score"]}
                    for hit in search_sources(self.conn, authenticated_principal=self.principal,
                                              scope_id=scope, query=query, limit=top_k)]
        if self.method == "fts_enriched":
            return self._enriched(query, top_k, scope)
        direct = self._enriched(query, 100, scope)
        if self.method == "fts_enriched_context":
            windows = self._enriched(query, 100, scope, context=True)
            hits = fuse([[h["id"] for h in direct], [h["id"] for h in windows]], top_k)
            return self._permitted_hits(hits, scope)
        ranked: list[str] = []
        seen: set[str] = set()

        def add(doc_id: str) -> None:
            if doc_id not in seen:
                seen.add(doc_id)
                ranked.append(doc_id)

        for hit in direct[:10]:
            add(hit["id"])
        prefix = scope + ":"
        for hit in direct[:5]:
            local_id = hit["id"][len(prefix):]
            for row in self.conn.execute(NEIGHBOR_SQL, (scope, local_id, self.principal)):
                add(prefix + row["source_id"])
        for hit in direct:
            add(hit["id"])
        hits = [{"id": doc_id, "score": 1.0 / rank}
                for rank, doc_id in enumerate(ranked[:top_k], 1)]
        return self._permitted_hits(hits, scope)

    def close(self) -> None:
        if self.index is not None:
            self.index.close()
            self.index = None
        self.conn.close()

