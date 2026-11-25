"""Download and convert the fixed, public LongMemEval-S cleaned release.

This is an independent evidence-retrieval conversion, not the official QA
evaluation. Gold annotations never enter document text. Each question has an
isolated history; abstention questions are excluded from retrieval metrics.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import statistics
from typing import Any
import urllib.request


DATASET_REVISION = "98d7416c24c778c2fee6e6f3006e7a073259d48f"
REPOSITORY_REVISION = "9e0b455f4ef0e2ab8f2e582289761153549043fc"
DATASET_URL = (
    "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/"
    + DATASET_REVISION + "/longmemeval_s_cleaned.json"
)
DATASET_SHA256 = "d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442"
DATASET_BYTES = 277_383_467
README_URL = (
    "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/"
    + DATASET_REVISION + "/README.md"
)
LICENSE_URL = (
    "https://raw.githubusercontent.com/xiaowu0162/LongMemEval/"
    + REPOSITORY_REVISION + "/LICENSE"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url: str, destination: Path, *, expected_sha256: str | None = None,
             expected_bytes: int | None = None) -> dict[str, Any]:
    """Stream, verify, and atomically publish a fixed public source file."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        observed = sha256_file(destination)
        if ((expected_sha256 is None or observed == expected_sha256)
                and (expected_bytes is None or destination.stat().st_size == expected_bytes)):
            return {"url": url, "path": str(destination), "sha256": observed,
                    "bytes": destination.stat().st_size, "verified_from_local_bytes": True}
        raise ValueError(f"Existing source fails integrity check: {destination.name}")
    temporary = destination.with_name(destination.name + ".part")
    digest = hashlib.sha256()
    size = 0
    next_progress = 32 * 1024 * 1024
    request = urllib.request.Request(url, headers={"User-Agent": "public-memory-benchmark/0.1"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as stream:
            while block := response.read(1024 * 1024):
                stream.write(block)
                digest.update(block)
                size += len(block)
                if size >= next_progress:
                    print(f"Downloading {destination.name}: {size} bytes", flush=True)
                    next_progress += 32 * 1024 * 1024
        observed = digest.hexdigest()
        if expected_bytes is not None and size != expected_bytes:
            raise ValueError(f"Size mismatch for {destination.name}: {size}")
        if expected_sha256 is not None and observed != expected_sha256:
            raise ValueError(f"SHA256 mismatch for {destination.name}: {observed}")
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return {"url": url, "path": str(destination), "sha256": observed,
            "bytes": size, "verified_from_local_bytes": True}


def json_write(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                         encoding="utf-8")
    os.replace(temporary, path)


def convert_record(entry: Any, index: int, duplicate_question_ids: set[str] | None = None
                   ) -> tuple[list[dict], dict | None, dict | None, dict | None]:
    """Return documents, strict QueryCase, metadata, or a whole-question exclusion."""
    question_id = entry.get("question_id") if isinstance(entry, dict) else None

    def exclude(reason: str, **detail: Any):
        return [], None, None, {"source_index": index, "question_id": question_id,
                               "reason": reason, **detail}

    if not isinstance(entry, dict):
        return exclude("invalid_record_object")
    if not isinstance(question_id, str) or not question_id.strip():
        return exclude("invalid_question_id")
    if question_id in (duplicate_question_ids or set()):
        return exclude("duplicate_question_id")
    if question_id.endswith("_abs"):
        return exclude("abstention_not_scored_as_empty_retrieval")
    question = entry.get("question")
    if not isinstance(question, str) or not question.strip():
        return exclude("invalid_question_text")
    raw_gold = entry.get("answer_session_ids")
    if not isinstance(raw_gold, list) or not raw_gold:
        return exclude("empty_or_invalid_gold")
    if any(not isinstance(value, str) or not value.strip() for value in raw_gold):
        return exclude("invalid_gold_id")
    gold = list(dict.fromkeys(raw_gold))
    session_ids = entry.get("haystack_session_ids")
    sessions = entry.get("haystack_sessions")
    dates = entry.get("haystack_dates")
    if not all(isinstance(value, list) for value in (session_ids, sessions, dates)):
        return exclude("invalid_history_arrays")
    if not (len(session_ids) == len(sessions) == len(dates)):
        return exclude("history_array_length_mismatch")
    if any(not isinstance(value, str) or not value.strip() for value in session_ids):
        return exclude("invalid_session_id")
    missing = [value for value in gold if value not in set(session_ids)]
    if missing:
        return exclude("gold_missing_from_history", missing_gold_ids=missing)
    scope = "longmemeval:" + question_id
    indexed_sessions: dict[str, dict[str, dict]] = {}
    duplicate_sessions = []
    for session_index, (session_id, date, turns) in enumerate(zip(session_ids, dates, sessions)):
        if not isinstance(date, str) or not isinstance(turns, list):
            return exclude("invalid_session_structure", session_id=session_id)
        lines = ["Session date: " + date]
        for turn_index, turn in enumerate(turns):
            if (not isinstance(turn, dict) or not isinstance(turn.get("role"), str)
                    or not isinstance(turn.get("content"), str)):
                return exclude("invalid_turn_structure", session_id=session_id,
                               turn_index=turn_index)
            # Deliberately whitelist fields, never stringify the annotated turn.
            lines.append(turn["role"] + ": " + turn["content"])
        text = "\n".join(lines)
        variants = indexed_sessions.setdefault(session_id, {})
        if text in variants:
            previous = variants[text]
            duplicate_sessions.append({"session_id": session_id,
                                       "kept_session_index": previous["first_index"],
                                       "removed_session_index": session_index})
            previous["source_indices"].append(session_index)
            continue
        variants[text] = {"first_index": session_index, "source_indices": [session_index],
                          "turn_count": len(turns)}

    # IDs depend only on the indexed history, never on answer labels. A reused
    # upstream ID with different date/text keeps every distinct occurrence.
    reserved_ids = {scope + ":" + value for value in session_ids}
    used_ids = set()
    ordered_documents = []
    session_output_ids = {}
    disambiguation = []
    turn_count = 0
    for session_id, variants in indexed_sessions.items():
        output_ids = []
        for occurrence, (text, detail) in enumerate(variants.items(), start=1):
            base_id = scope + ":" + session_id
            if len(variants) == 1:
                output_id = base_id
            else:
                output_id = base_id + ":occurrence:" + str(occurrence)
                # Preserve any original ID that happens to look like our suffix.
                while output_id in reserved_ids or output_id in used_ids:
                    output_id += ":disambiguated"
            used_ids.add(output_id)
            output_ids.append(output_id)
            ordered_documents.append((detail["first_index"],
                                      {"id": output_id, "scope": scope, "text": text}))
            turn_count += detail["turn_count"]
        session_output_ids[session_id] = output_ids
        if len(variants) > 1:
            disambiguation.append({"source_session_id": session_id,
                                   "source_occurrence_count": sum(len(v["source_indices"]) for v in variants.values()),
                                   "distinct_index_text_count": len(variants),
                                   "output_ids": output_ids,
                                   "source_indices_by_output": [v["source_indices"] for v in variants.values()]})
    ambiguous_gold = [value for value in gold if len(session_output_ids[value]) != 1]
    if ambiguous_gold:
        return exclude("ambiguous_gold_session_id", ambiguous_gold_ids=ambiguous_gold,
                       duplicate_id_disambiguation=disambiguation)
    documents = [document for _, document in sorted(ordered_documents, key=lambda pair: pair[0])]
    query = {"id": scope, "query": question, "scope": scope,
             "expected_ids": [session_output_ids[value][0] for value in gold]}
    metadata = {"category": entry.get("question_type", "unspecified"),
                "source_question_id": question_id, "source_index": index,
                "question_date": entry.get("question_date"),
                "session_count": len(documents), "turn_count": turn_count,
                "source_session_count": len(session_ids),
                "gold_session_count": len(gold),
                "deduplicated_gold_count": len(raw_gold) - len(gold),
                "deduplicated_session_count": len(duplicate_sessions),
                "deduplicated_sessions": duplicate_sessions,
                "duplicate_id_disambiguation": disambiguation}
    return documents, query, metadata, None


def convert(source: Path, output: Path, *, source_metadata: dict | None = None,
            verify_release: bool = True) -> dict:
    source_hash = sha256_file(source)
    if verify_release and (source_hash != DATASET_SHA256 or source.stat().st_size != DATASET_BYTES):
        raise ValueError("Source does not match the pinned LongMemEval-S cleaned release")
    entries = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(entries, list):
        raise ValueError("LongMemEval source must be a JSON array")
    question_counts = Counter(entry.get("question_id") for entry in entries
                              if isinstance(entry, dict) and isinstance(entry.get("question_id"), str))
    duplicates = {key for key, count in question_counts.items() if count > 1}
    output.mkdir(parents=True, exist_ok=True)
    documents_path = output / "longmemeval.documents.jsonl"
    queries_path = output / "longmemeval.queries.jsonl"
    metadata_path = output / "longmemeval.query_metadata.json"
    exclusions_path = output / "longmemeval.exclusions.json"
    metadata = {}
    exclusions = []
    counts = Counter()
    scopes = []
    gold_sizes = []
    document_count = 0
    turn_count = 0
    doc_temp = documents_path.with_suffix(".jsonl.tmp")
    query_temp = queries_path.with_suffix(".jsonl.tmp")
    with doc_temp.open("w", encoding="utf-8") as documents_stream, query_temp.open("w", encoding="utf-8") as queries_stream:
        for index, entry in enumerate(entries):
            documents, query, query_metadata, exclusion = convert_record(entry, index, duplicates)
            if exclusion is not None:
                exclusions.append(exclusion)
                continue
            for document in documents:
                documents_stream.write(json.dumps(document, ensure_ascii=False, allow_nan=False) + "\n")
            queries_stream.write(json.dumps(query, ensure_ascii=False, allow_nan=False) + "\n")
            metadata[query["id"]] = query_metadata
            counts[query_metadata["category"]] += 1
            scopes.append(len(documents))
            gold_sizes.append(len(query["expected_ids"]))
            document_count += len(documents)
            turn_count += query_metadata["turn_count"]
    os.replace(doc_temp, documents_path)
    os.replace(query_temp, queries_path)
    json_write(metadata_path, metadata)
    json_write(exclusions_path, exclusions)
    manifest = {
        "dataset": "LongMemEval-S cleaned",
        "conversion_version": "session-retrieval-v2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_url": DATASET_URL,
        "source_revision": DATASET_REVISION,
        "source_file_sha256": source_hash,
        "source_file_bytes": source.stat().st_size,
        "source_verified_against_pinned_release": verify_release,
        "license": "MIT",
        "repository_url": "https://github.com/xiaowu0162/LongMemEval",
        "repository_revision": REPOSITORY_REVISION,
        "source_files": source_metadata or {},
        "total_source_questions": len(entries),
        "eligible_queries": len(metadata),
        "excluded_queries": len(exclusions),
        "exclusions_by_reason": dict(Counter(item["reason"] for item in exclusions)),
        "scope_count": len(scopes),
        "document_count": document_count,
        "turn_count": turn_count,
        "documents_per_scope": {
            "min": min(scopes) if scopes else None,
            "median": statistics.median(scopes) if scopes else None,
            "mean": statistics.mean(scopes) if scopes else None,
            "max": max(scopes) if scopes else None,
            "histogram": dict(sorted(Counter(scopes).items())),
        },
        "max_gold_documents": max(gold_sizes) if gold_sizes else 0,
        "categories": dict(sorted(counts.items())),
        "session_deduplication": {
            "policy": "Same session ID is collapsed only when indexed date+role+content text is exactly equal",
            "removed_session_count": sum(value["deduplicated_session_count"] for value in metadata.values()),
            "by_question": {key: value["deduplicated_sessions"] for key, value in metadata.items()
                            if value["deduplicated_session_count"]},
        },
        "duplicate_id_disambiguation": {
            "policy": "Preserve every distinct indexed text; append :occurrence:<1-based distinct occurrence> to all variants of a reused source ID. If that collides with an original ID, append :disambiguated until unique. Generation is independent of gold labels.",
            "affected_questions": sum(bool(value["duplicate_id_disambiguation"]) for value in metadata.values()),
            "by_question": {key: value["duplicate_id_disambiguation"] for key, value in metadata.items()
                            if value["duplicate_id_disambiguation"]},
        },
        "index_fields": ["haystack_dates", "haystack_sessions.role", "haystack_sessions.content"],
        "excluded_index_fields": ["answer", "answer_session_ids", "has_answer", "question", "question_type"],
        "query_text": "question only; question_date retained in query metadata",
        "retrieval_unit": "whole conversation session",
        "isolation_unit": "question_id",
        "gold_policy": "All unique answer_session_ids; exclude whole question if any gold ID is invalid, absent, or maps to multiple distinct indexed texts",
        "abstention_policy": "Excluded from retrieval metrics, following upstream retrieval aggregation; not labelled expect_empty",
        "scope_note": "Evidence retrieval only, not official answer-generation accuracy",
        "outputs": {path.name: {"sha256": sha256_file(path), "bytes": path.stat().st_size}
                    for path in (documents_path, queries_path, metadata_path, exclusions_path)},
    }
    json_write(output / "longmemeval.manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    raw = root / "data/raw"
    source = raw / "longmemeval_s_cleaned.json"
    files = {
        "dataset": download(DATASET_URL, source, expected_sha256=DATASET_SHA256,
                            expected_bytes=DATASET_BYTES),
        "dataset_readme": download(README_URL, raw / "longmemeval.UPSTREAM_README.md"),
        "license": download(LICENSE_URL, raw / "longmemeval.UPSTREAM_LICENSE"),
    }
    # Preserve relative paths so the manifest remains useful after unpacking.
    for info in files.values():
        info["path"] = str(Path(info["path"]).relative_to(root))
    manifest = convert(source, root / "data/converted", source_metadata=files)
    print(json.dumps({key: manifest[key] for key in
                      ("dataset", "eligible_queries", "excluded_queries", "document_count",
                       "max_gold_documents", "categories", "exclusions_by_reason")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
