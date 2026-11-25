"""Adapter invariants on small fixtures, not public-dataset quality scores."""
from __future__ import annotations

import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

from benchmarks.retriever import DuorouRetriever, METHODS, source_documents
from examples.sqlite_demo import connect_database, import_locomo_records, initialize_schema, search_sources
from vendor.public_memory_bench.locomo import convert_locomo


def fixture() -> list[dict]:
    return [{
        "sample_id": "A",
        "conversation": {
            "speaker_a": "Ada", "speaker_b": "Ben",
            "session_1_date_time": "10:00 am on 1 January, 2024",
            "session_1": [
                {"dia_id": "arbitrary-prev", "speaker": "Ben", "text": "A quiet preceding thought."},
                {"dia_id": "arbitrary-seed", "speaker": "Ada", "text": "Uniqueneedle anchors this memory.",
                 "blip_caption": "A captionmarker among flowers.", "qa": "EVALUATORSECRETSENTINEL"},
            ],
            "session_2": [
                {"dia_id": "arbitrary-next-session", "speaker": "Ada", "text": "An unrelated session begins."},
            ],
            "session_3": [turn for i in range(20) for turn in [
                {"dia_id": f"rank-seed-{i:02d}", "speaker": "Ada", "text": "Rankneedle matching text."},
                {"dia_id": f"rank-neighbor-{i:02d}", "speaker": "Ben", "text": "Context without the search term."},
            ]],
        },
        "qa": [], "session_summary": "EVALUATORSECRETSENTINEL",
    }, {
        "sample_id": "B",
        "conversation": {"speaker_a": "Ada", "speaker_b": "Ben", "session_1": [
            {"dia_id": "arbitrary-seed", "speaker": "Ben", "text": "Uniqueneedle forbiddenpayload."},
        ]},
        "qa": [],
    }]


class RetrieverTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.db_path = self.root / "fixture.sqlite"
        self.conn = connect_database(self.db_path)
        initialize_schema(self.conn)
        self.samples = fixture()
        import_locomo_records(self.conn, self.samples, recorded_at_us=1)
        self.conn.execute("INSERT INTO principals VALUES ('benchmark-reader', 'service')")
        self.conn.execute("INSERT INTO scope_grants VALUES ('locomo:A','benchmark-reader','read')")
        self.conn.commit()
        self.paths = {}
        for label, enriched in [("raw", False), ("enriched", True)]:
            path = self.root / f"{label}.jsonl"
            path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in
                                    source_documents(self.conn, enriched=enriched)), encoding="utf-8")
            self.paths[label] = path
        self.retrievers = []

    def tearDown(self) -> None:
        for retriever in self.retrievers:
            retriever.close()
        self.conn.close()
        self.temp.cleanup()

    def make(self, method: str) -> DuorouRetriever:
        path = self.paths[method.removeprefix("bm25_")] if method.startswith("bm25_") else None
        retriever = DuorouRetriever(self.db_path, method, path)
        self.retrievers.append(retriever)
        return retriever

    def test_enriched_projection_exactly_matches_original_converter(self) -> None:
        expected = sorted(convert_locomo(self.samples)["documents"], key=lambda row: row["id"])
        self.assertEqual(source_documents(self.conn, enriched=True), expected)
        text = "\n".join(row["text"] for row in expected)
        self.assertNotIn("EVALUATORSECRETSENTINEL", text)

    def test_current_raw_lookup_is_identical_and_primary_database_is_read_only(self) -> None:
        retriever = self.make("fts_raw")
        expected = search_sources(self.conn, authenticated_principal="benchmark-reader",
                                  scope_id="locomo:A", query="Uniqueneedle", limit=20)
        self.assertEqual(retriever.search("Uniqueneedle", 20, "locomo:A"), [
            {"id": "locomo:A:" + hit["source_id"], "score": -hit["fts_score"]} for hit in expected])
        with self.assertRaises(sqlite3.OperationalError):
            retriever.conn.execute("DELETE FROM sources")

    def test_all_methods_deny_unauthorized_scope_and_recheck_revocation(self) -> None:
        retrievers = [self.make(method) for method in METHODS]
        for retriever in retrievers:
            with self.subTest(method=retriever.method):
                with self.assertRaises(PermissionError):
                    retriever.search("Uniqueneedle", 20, "locomo:B")
                hits = retriever.search("Uniqueneedle", 20, "locomo:A")
                self.assertTrue(hits)
                self.assertTrue(all(hit["id"].startswith("locomo:A:") for hit in hits))
        self.conn.execute("DELETE FROM scope_grants WHERE scope_id='locomo:A'")
        self.conn.commit()
        for retriever in retrievers:
            with self.subTest(method=retriever.method), self.assertRaises(PermissionError):
                retriever.search("Uniqueneedle", 20, "locomo:A")

    def test_enriched_caption_is_retrievable_but_evaluator_fields_are_not(self) -> None:
        for method in METHODS:
            retriever = self.make(method)
            with self.subTest(method=method):
                self.assertEqual(retriever.search("EVALUATORSECRETSENTINEL", 20, "locomo:A"), [])
                result = retriever.search("captionmarker", 20, "locomo:A")
                if "enriched" in method:
                    self.assertIn("locomo:A:arbitrary-seed", [hit["id"] for hit in result])
                else:
                    self.assertEqual(result, [])

    def test_neighbors_use_membership_and_never_cross_session(self) -> None:
        retriever = self.make("fts_enriched_neighbors")
        ids = [hit["id"] for hit in retriever.search("Uniqueneedle", 20, "locomo:A")]
        self.assertEqual(ids, ["locomo:A:arbitrary-seed", "locomo:A:arbitrary-prev"])
        self.assertNotIn("locomo:A:arbitrary-next-session", ids)
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual([hit["id"] for hit in retriever.search("Uniqueneedle", 1, "locomo:A")], ids[:1])

    def test_invalid_requests_are_rejected_consistently(self) -> None:
        for method in METHODS:
            retriever = self.make(method)
            for query, top_k, scope in [("x", True, "locomo:A"), ("x", 0, "locomo:A"),
                                         ("x", 101, "locomo:A"), ("", 20, "locomo:A"),
                                         ("x " * 65, 20, "locomo:A"), ("x", 20, "")]:
                with self.subTest(method=method, query=query, top_k=top_k, scope=scope):
                    with self.assertRaises(ValueError):
                        retriever.search(query, top_k, scope)

    def test_expansion_preserves_top10_and_uses_a_consistent_bounded_prefix(self) -> None:
        direct = self.make("fts_enriched")
        expanded = self.make("fts_enriched_neighbors")
        lexical_ids = [hit["id"] for hit in direct.search("Rankneedle", 100, "locomo:A")]
        ranking = [hit["id"] for hit in expanded.search("Rankneedle", 100, "locomo:A")]
        self.assertEqual(ranking[:10], lexical_ids[:10])
        self.assertEqual(len(ranking), len(set(ranking)))
        self.assertEqual(sum("rank-neighbor" in doc_id for doc_id in ranking), 5)
        for k in (1, 5, 10, 20):
            self.assertEqual([hit["id"] for hit in expanded.search("Rankneedle", k, "locomo:A")], ranking[:k])

    def test_context_index_recalls_leaf_without_crossing_session(self):
        r = self.make("fts_enriched_context")
        ids = [h["id"] for h in r.search("Uniqueneedle", 20, "locomo:A")]
        self.assertIn("locomo:A:arbitrary-seed", ids)
        self.assertIn("locomo:A:arbitrary-prev", ids)
        self.assertNotIn("locomo:A:arbitrary-next-session", ids)
        self.assertEqual(len(ids), len(set(ids)))

    def test_control_rejects_documents_that_do_not_match_source_projection(self) -> None:
        bad_path = self.root / "bad.jsonl"
        documents = source_documents(self.conn, enriched=False)
        documents[0]["text"] += " EVALUATORSECRETSENTINEL"
        bad_path.write_text("".join(json.dumps(row) + "\n" for row in documents), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "exactly match"):
            DuorouRetriever(self.db_path, "bm25_raw", bad_path)


if __name__ == "__main__":
    unittest.main()
