"""Constraint/application-gate tests on tiny fixtures, NOT benchmark scores.

Run: python -m unittest discover -s tests -v
"""
from __future__ import annotations

import json
from pathlib import Path
import sqlite3
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from sqlite_demo import (  # noqa: E402
    TAG_DEFINITIONS, connect_database, import_locomo_records, initialize_schema,
    insert_source, search_sources,
)


class SchemaTests(unittest.TestCase):
    def setUp(self) -> None:
        self.db = connect_database()
        initialize_schema(self.db)
        self.db.executemany("INSERT INTO scopes VALUES (?, ?)", [("A", "A"), ("B", "B")])
        self.db.execute("INSERT INTO principals VALUES ('reader', 'service')")
        self.db.execute("INSERT INTO scope_grants VALUES ('A', 'reader', 'read')")
        self.db.executemany("INSERT INTO entities VALUES (?, ?, 'person', ?, 'source_bound')",
                            [("A", "alice", "Alice"), ("B", "bob", "Bob")])
        self.text = "Alice has a café and a brain 🧠."
        insert_source(self.db, scope_id="A", source_id="a", speaker_entity_id="alice",
                      content=self.text, recorded_at_us=100)
        insert_source(self.db, scope_id="B", source_id="b", speaker_entity_id="bob",
                      content="Bob owns forbiddenpayload.", recorded_at_us=100)
        self.db.executemany("INSERT INTO evidence_spans VALUES (?, ?, ?, 0, ?)",
                            [("A", "pa", "a", len(self.text)), ("B", "pb", "b", 26)])
        self.db.executemany(
            """INSERT INTO events (scope_id, event_id, kind, title, boundary_origin,
                   status, time_precision) VALUES (?, ?, 'source_session', 'session',
                   'source_metadata', 'provisional', 'unknown')""",
            [("A", "ea"), ("B", "eb")],
        )
        self.db.execute("INSERT INTO event_memberships VALUES ('A', 'ea', 'pa', 0)")
        self.db.executemany("INSERT INTO tag_definitions VALUES (?, ?, ?, ?, ?)", TAG_DEFINITIONS)

    def tearDown(self) -> None:
        self.db.close()

    def claim(self, **overrides) -> None:
        values = dict(scope_id="A", claim_id="c", version=1, subject_entity_id="alice",
                      predicate="likes", object_kind="text", object_text="coffee",
                      attribution_entity_id="alice", polarity="affirmative", modality="actual",
                      epistemic_mode="source_assertion", status="candidate", primary_span_id="pa",
                      recorded_from_us=100, extractor_version="test/1", extraction_confidence=0.8)
        values.update(overrides)
        self.db.execute("INSERT INTO claims (" + ",".join(values) + ") VALUES (" +
                        ",".join("?" for _ in values) + ")", tuple(values.values()))

    def tag(self, **overrides) -> None:
        values = dict(scope_id="A", annotation_id="t", namespace="source", tag_key="kind",
                      value_json='"dialogue_turn"', source_id="a", evidence_span_id="pa",
                      origin="metadata", status="accepted", extractor_version="test/1")
        values.update(overrides)
        self.db.execute("INSERT INTO tag_annotations (" + ",".join(values) + ") VALUES (" +
                        ",".join("?" for _ in values) + ")", tuple(values.values()))

    def test_cross_scope_references_rejected(self) -> None:
        attempts = [
            ("INSERT INTO evidence_spans VALUES ('A', 'bad', 'b', 0, 1)", ()),
            ("INSERT INTO event_memberships VALUES ('A', 'eb', 'pa', 1)", ()),
            ("INSERT INTO event_memberships VALUES ('A', 'ea', 'pb', 1)", ()),
        ]
        for statement, params in attempts:
            with self.subTest(statement=statement), self.assertRaises(sqlite3.IntegrityError):
                self.db.execute(statement, params)
        with self.assertRaises(sqlite3.IntegrityError):
            insert_source(self.db, scope_id="A", source_id="bad", speaker_entity_id="bob",
                          content="Wrong speaker scope", recorded_at_us=100)
        with self.assertRaises(sqlite3.IntegrityError):
            self.claim(subject_entity_id="bob")
        with self.assertRaises(sqlite3.IntegrityError):
            self.claim(primary_span_id="pb")
        with self.assertRaises(sqlite3.IntegrityError):
            self.tag(source_id="b")
        with self.assertRaises(sqlite3.IntegrityError):
            self.tag(evidence_span_id="pb")

    def test_unicode_span_bounds_and_exact_mention(self) -> None:
        start = self.text.index("🧠")
        self.db.execute("INSERT INTO evidence_spans VALUES ('A', 'emoji', 'a', ?, ?)", (start, start + 1))
        extracted = self.db.execute(
            "SELECT substr(s.content, p.char_start+1, p.char_end-p.char_start) "
            "FROM evidence_spans p JOIN sources s USING(scope_id,source_id) WHERE p.span_id='emoji'"
        ).fetchone()[0]
        self.assertEqual(extracted, "🧠")
        for lo, hi in [(-1, 1), (0, len(self.text) + 1), (1, 1), (4, 3)]:
            with self.subTest(lo=lo, hi=hi), self.assertRaises(sqlite3.IntegrityError):
                self.db.execute("INSERT INTO evidence_spans VALUES ('A', 'bad', 'a', ?, ?)", (lo, hi))
        self.db.execute(
            "INSERT INTO entity_mentions VALUES ('A', 'm', 'emoji', NULL, '🧠', 'unresolved', 'test/1', NULL)"
        )
        with self.assertRaises(sqlite3.IntegrityError):
            self.db.execute("UPDATE entity_mentions SET surface='brain' WHERE mention_id='m'")
        with self.assertRaises(sqlite3.IntegrityError):
            self.db.execute("UPDATE evidence_spans SET char_end=1 WHERE span_id='pa'")

    def test_source_immutable_and_nul_rejected(self) -> None:
        with self.assertRaises(sqlite3.IntegrityError):
            self.db.execute("UPDATE sources SET content='rewritten' WHERE scope_id='A'")
        with self.assertRaises(sqlite3.IntegrityError):
            self.db.execute("UPDATE sources SET metadata_json='{}' WHERE scope_id='A'")
        with self.assertRaises(sqlite3.IntegrityError):
            self.db.execute("INSERT OR REPLACE INTO sources SELECT * FROM sources WHERE scope_id='A'")
        with self.assertRaises(sqlite3.IntegrityError):
            self.db.execute("INSERT OR REPLACE INTO evidence_spans VALUES ('A','pa','a',0,1)")
        with self.assertRaises(sqlite3.IntegrityError):
            insert_source(self.db, scope_id="A", source_id="nul", speaker_entity_id="alice",
                          content="A\x00B", recorded_at_us=100)

    def test_fts_delete_cascade_and_primary_evidence_removal(self) -> None:
        self.claim()
        self.tag()
        self.assertEqual(self.db.execute("SELECT count(*) FROM source_fts WHERE source_fts MATCH 'Alice'").fetchone()[0], 1)
        self.db.execute("DELETE FROM sources WHERE scope_id='A' AND source_id='a'")
        self.assertEqual(self.db.execute("SELECT count(*) FROM source_fts WHERE source_fts MATCH 'Alice'").fetchone()[0], 0)
        for table in ("evidence_spans", "event_memberships", "claims", "tag_annotations"):
            self.assertEqual(self.db.execute(f"SELECT count(*) FROM {table} WHERE scope_id='A'").fetchone()[0], 0)
        self.assertEqual(self.db.execute("SELECT count(*) FROM sources WHERE scope_id='B'").fetchone()[0], 1)
        self.db.execute("INSERT INTO source_fts(source_fts,rank) VALUES ('integrity-check',1)")
        self.assertEqual(list(self.db.execute("PRAGMA foreign_key_check")), [])

    def test_authorization_before_fts_and_scope_filter(self) -> None:
        rows = search_sources(self.db, authenticated_principal="reader", scope_id="A", query="Alice")
        self.assertEqual([r["source_id"] for r in rows], ["a"])
        self.assertEqual(search_sources(self.db, authenticated_principal="reader", scope_id="A", query="forbiddenpayload"), [])
        with self.assertRaises(PermissionError):
            search_sources(self.db, authenticated_principal="reader", scope_id="B", query="forbiddenpayload")
        with self.assertRaises(PermissionError):
            search_sources(self.db, authenticated_principal="untrusted-model-name", scope_id="A", query="Alice")
        self.db.execute("DELETE FROM scope_grants WHERE principal_id='reader'")
        with self.assertRaises(PermissionError):
            search_sources(self.db, authenticated_principal="reader", scope_id="A", query="Alice")

    def test_query_syntax_is_literal(self) -> None:
        # Query syntax cannot inject a scope predicate or use a bare FTS operator.
        rows = search_sources(self.db, authenticated_principal="reader", scope_id="A", query='Alice OR "*')
        self.assertEqual([r["source_id"] for r in rows], ["a"])
        with self.assertRaises(ValueError):
            search_sources(self.db, authenticated_principal="reader", scope_id="A", query="***")

    def test_actual_negative_hypothetical_are_distinct(self) -> None:
        self.claim(claim_id="actual", polarity="affirmative", modality="actual")
        self.claim(claim_id="negative", polarity="negative", modality="actual")
        self.claim(claim_id="hypothesis", polarity="affirmative", modality="hypothetical")
        self.claim(claim_id="inferred", epistemic_mode="model_inference", extraction_confidence=0.99)
        rows = self.db.execute("SELECT claim_id, polarity, modality, epistemic_mode, status FROM claims ORDER BY claim_id").fetchall()
        self.assertEqual(len(rows), 4)
        self.assertEqual({(r["polarity"], r["modality"]) for r in rows},
                         {("affirmative", "actual"), ("negative", "actual"), ("affirmative", "hypothetical")})
        self.assertTrue(all(r["status"] == "candidate" for r in rows))
        self.assertEqual(next(r for r in rows if r["claim_id"] == "inferred")["epistemic_mode"], "model_inference")

    def test_claim_object_type_and_intervals(self) -> None:
        cases = [dict(object_number=1.0), dict(object_kind="boolean", object_text=None, object_boolean=2),
                 dict(valid_from_us=20, valid_to_us=20), dict(valid_from_us=21, valid_to_us=20),
                 dict(recorded_to_us=100), dict(recorded_to_us=99), dict(extraction_confidence=1.1)]
        for overrides in cases:
            with self.subTest(overrides=overrides), self.assertRaises(sqlite3.IntegrityError):
                self.claim(**overrides)
        self.claim(object_kind="boolean", object_text=None, object_boolean=0,
                   valid_from_us=20, valid_to_us=30, recorded_to_us=200)

    def test_recorded_version_append_and_as_of(self) -> None:
        self.claim(valid_from_us=10)
        with self.assertRaises(sqlite3.IntegrityError):
            self.claim(version=2, recorded_from_us=200)
        with self.assertRaises(sqlite3.IntegrityError):
            self.db.execute("UPDATE claims SET object_text='tea' WHERE claim_id='c'")
        self.db.execute("UPDATE claims SET recorded_to_us=200 WHERE claim_id='c' AND version=1")
        with self.assertRaises(sqlite3.IntegrityError):
            self.claim(version=2, recorded_from_us=199)
        with self.assertRaises(sqlite3.IntegrityError):
            self.claim(version=3, recorded_from_us=200)
        self.claim(version=2, recorded_from_us=200, object_text="tea", valid_from_us=10)
        for when, expected in [(199, "coffee"), (200, "tea")]:
            value = self.db.execute(
                "SELECT object_text FROM claims WHERE scope_id='A' AND claim_id='c' "
                "AND recorded_from_us <= ? AND (recorded_to_us IS NULL OR ? < recorded_to_us)",
                (when, when),
            ).fetchone()[0]
            self.assertEqual(value, expected)
        # A retrospective correction keeps valid time in the past and records now.
        version_two = self.db.execute("SELECT valid_from_us, recorded_from_us FROM claims WHERE version=2").fetchone()
        self.assertEqual(tuple(version_two), (10, 200))
        with self.assertRaises(sqlite3.IntegrityError):
            self.db.execute("UPDATE claims SET recorded_to_us=201 WHERE version=1")

    def test_conflict_edges_do_not_overwrite_claims(self) -> None:
        self.claim(claim_id="c1", polarity="affirmative")
        self.claim(claim_id="c2", polarity="negative")
        self.db.execute("INSERT INTO claim_relations VALUES ('A','c2',1,'contradicts','c1',1,'pa','human','test/1')")
        self.assertEqual(self.db.execute("SELECT count(*) FROM claims").fetchone()[0], 2)
        with self.assertRaises(sqlite3.IntegrityError):
            self.db.execute("INSERT INTO claim_relations VALUES ('A','c2',1,'same_as','c1',1,'pa','human','test/1')")
        with self.assertRaises(sqlite3.IntegrityError):
            self.db.execute("INSERT INTO claim_relations VALUES ('A','c2',1,'supports','c1',1,'pb','human','test/1')")

    def test_same_alias_does_not_merge_entities(self) -> None:
        self.db.execute("INSERT INTO entities VALUES ('A', 'another_alice', 'person', 'Alice', 'unresolved')")
        self.db.executemany(
            "INSERT INTO entity_aliases VALUES ('A', ?, ?, 'Alex', 'pa', 'human', 'candidate', 'test/1', NULL)",
            [("alias1", "alice"), ("alias2", "another_alice")],
        )
        self.assertEqual(self.db.execute("SELECT count(DISTINCT entity_id) FROM entity_aliases WHERE alias_text='Alex'").fetchone()[0], 2)

    def test_tag_definition_type_scope_and_evidence(self) -> None:
        self.tag()
        for overrides in [dict(annotation_id="bad", value_json='"unknown"'),
                          dict(annotation_id="bad", value_json="1"),
                          dict(annotation_id="bad", tag_key="unregistered"),
                          dict(annotation_id="bad", origin="model", evidence_span_id=None),
                          dict(annotation_id="bad", event_id="ea"),
                          dict(annotation_id="bad", source_id=None)]:
            with self.subTest(overrides=overrides), self.assertRaises(sqlite3.IntegrityError):
                self.tag(**overrides)
        with self.assertRaises(sqlite3.IntegrityError):
            self.db.execute("UPDATE tag_annotations SET value_json='123' WHERE annotation_id='t'")
        with self.assertRaises(sqlite3.IntegrityError):
            self.db.execute("UPDATE tag_definitions SET value_type='number' WHERE namespace='source'")
        with self.assertRaises(sqlite3.IntegrityError):
            self.db.execute("INSERT OR REPLACE INTO tag_definitions VALUES ('source','kind','number',NULL,'changed')")
        self.tag(annotation_id="ordinal", namespace="chronology", tag_key="session_ordinal",
                 value_json="1", source_id=None, event_id="ea", evidence_span_id=None)

    def test_completion_requires_same_scope_evidence(self) -> None:
        statement = """INSERT INTO intentions (scope_id, intention_id, owner_entity_id,
            evidence_span_id, completion_span_id, description, cue_json, state, origin,
            extractor_version) VALUES ('A','intent','alice','pa',?,'do task','{}',?,'human','test/1')"""
        with self.assertRaises(sqlite3.IntegrityError):
            self.db.execute(statement, (None, "completed"))
        with self.assertRaises(sqlite3.IntegrityError):
            self.db.execute(statement, ("pb", "completed"))
        self.db.execute(statement, (None, "active"))
        with self.assertRaises(sqlite3.IntegrityError):
            self.db.execute("UPDATE intentions SET state='completed'")
        self.db.execute("UPDATE intentions SET state='completed', completion_span_id='pa'")
        # This test checks the pointer constraint, not whether that text proves completion.

    def test_working_set_cannot_reference_cross_scope_span(self) -> None:
        self.db.execute("INSERT INTO working_sets VALUES ('A','working','reader','task',1000,NULL)")
        with self.assertRaises(sqlite3.IntegrityError):
            self.db.execute("INSERT INTO working_set_members VALUES ('A','working','pb',0)")
        self.db.execute("INSERT INTO working_set_members VALUES ('A','working','pa',0)")


class ImportAllowlistTests(unittest.TestCase):
    def test_gold_and_caption_never_enter_fts_or_claims(self) -> None:
        db = connect_database()
        self.addCleanup(db.close)
        initialize_schema(db)
        fixture = [{
            "sample_id": "fixture",
            "qa": [{"question": "GOLDQUESTIONTOKEN", "answer": "GOLDANSWERTOKEN", "evidence": ["D9:9"]}],
            "event_summary": "GOLDEVENTTOKEN", "observation": "GOLDOBSERVATIONTOKEN",
            "session_summary": "GOLDSUMMARYTOKEN",
            "conversation": {"speaker_a": "One", "speaker_b": "Two", "session_1_date_time": "local date",
                "session_1": [{"dia_id": "D1:1", "speaker": "One", "text": "Raw dialogue only.",
                               "blip_caption": "CAPTIONTOKEN", "img_url": ["https://example.invalid/image"],
                               "unapproved_extra": "EXTRATOKEN"}]},
        }]
        scopes = import_locomo_records(db, fixture, recorded_at_us=100)
        self.assertEqual(scopes, ["locomo:fixture"])
        source = db.execute("SELECT content, metadata_json FROM sources").fetchone()
        self.assertEqual(source["content"], "Raw dialogue only.")
        metadata = json.loads(source["metadata_json"])
        self.assertEqual(metadata["blip_caption"], "CAPTIONTOKEN")
        self.assertNotIn("unapproved_extra", metadata)
        for token in ("GOLDQUESTIONTOKEN", "GOLDANSWERTOKEN", "GOLDEVENTTOKEN", "GOLDOBSERVATIONTOKEN",
                      "GOLDSUMMARYTOKEN", "CAPTIONTOKEN", "EXTRATOKEN"):
            self.assertEqual(db.execute("SELECT count(*) FROM source_fts WHERE source_fts MATCH ?", (token,)).fetchone()[0], 0)
        for table in ("claims", "claim_relations", "entity_mentions", "entity_aliases", "intentions"):
            self.assertEqual(db.execute(f"SELECT count(*) FROM {table}").fetchone()[0], 0)
        event = db.execute("SELECT kind,status,time_start_us,time_precision FROM events").fetchone()
        self.assertEqual(tuple(event), ("source_session", "provisional", None, "source_local_unspecified"))


if __name__ == "__main__":
    unittest.main()
