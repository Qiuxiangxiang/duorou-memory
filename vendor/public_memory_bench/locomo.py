"""Pinned LoCoMo download and conservative turn-level retrieval conversion.

Only the conversation fields are indexed. Questions and evidence are evaluation
inputs; answers, generated observations, and summaries never enter documents.
This is an evidence-retrieval adaptation, not the official answer-F1 benchmark.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any
import urllib.request


REVISION = "3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376"
BASE_URL = f"https://raw.githubusercontent.com/snap-research/locomo/{REVISION}"
DATA_URL = BASE_URL + "/data/locomo10.json"
LICENSE_URL = BASE_URL + "/LICENSE.txt"
DATA_SHA256 = "79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4"
LICENSE_SHA256 = "41003d4a74749c0220e33dd415042164b5a1093ed401f36277234f772d22d3d0"
CATEGORIES = {1: "multi_hop", 2: "temporal", 3: "open_domain", 4: "single_hop", 5: "adversarial"}
SESSION_KEY = re.compile(r"session_(\d+)")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def _write_json(path: Path, value: Any) -> None:
    _atomic_bytes(path, (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8"))


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    payload = "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows)
    _atomic_bytes(path, payload.encode("utf-8"))


def download_verified(url: str, destination: Path, expected_sha256: str | None, max_bytes: int) -> None:
    """Download public pinned data; reject a mismatching local cache or payload."""
    if destination.exists():
        data = destination.read_bytes()
        if len(data) > max_bytes:
            raise ValueError(f"Cached file exceeds limit: {destination.name}")
    else:
        request = urllib.request.Request(url, headers={"User-Agent": "public-memory-benchmark/0.1"})
        with urllib.request.urlopen(request, timeout=30) as response:
            data = response.read(max_bytes + 1)
        if len(data) > max_bytes:
            raise ValueError(f"Downloaded file exceeds limit: {destination.name}")
    actual = hashlib.sha256(data).hexdigest()
    if expected_sha256 is not None and actual != expected_sha256:
        raise ValueError(f"SHA256 mismatch for {destination.name}: {actual}")
    if not destination.exists():
        _atomic_bytes(destination, data)


def convert_locomo(samples: list[dict[str, Any]], *, include_captions: bool = True) -> dict[str, Any]:
    """Keep only answerable questions whose entire raw gold list resolves.

    Deliberately do not repair punctuation, strip leading zeroes, infer missing
    IDs, or drop only the invalid portion of a gold set. Exclusions are explicit.
    """
    documents: list[dict[str, Any]] = []
    queries: list[dict[str, Any]] = []
    metadata: dict[str, Any] = {}
    exclusions: list[dict[str, Any]] = []
    scopes: set[str] = set()
    counts: Counter[str] = Counter()
    source_categories: Counter[str] = Counter()
    eligible_categories: Counter[str] = Counter()
    scope_stats: dict[str, Any] = {}
    max_gold = 0

    for sample in samples:
        sample_id = sample["sample_id"]
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError("sample_id must be a nonempty string")
        scope = "locomo:" + sample_id
        if scope in scopes:
            raise ValueError(f"Duplicate sample scope: {scope}")
        scopes.add(scope)
        conversation = sample["conversation"]
        turn_ids: set[str] = set()
        session_keys = sorted(
            (key for key in conversation if SESSION_KEY.fullmatch(key)),
            key=lambda key: int(SESSION_KEY.fullmatch(key).group(1)),
        )
        counts["sessions"] += len(session_keys)
        caption_count = 0
        for key in session_keys:
            timestamp = conversation.get(key + "_date_time", "")
            if not isinstance(timestamp, str):
                raise ValueError(f"Invalid session timestamp in {scope}")
            for turn in conversation[key]:
                turn_id = turn["dia_id"]
                if not isinstance(turn_id, str) or not turn_id:
                    raise ValueError(f"Invalid turn ID in {scope}")
                if turn_id in turn_ids:
                    raise ValueError(f"Duplicate turn ID {scope}:{turn_id}")
                turn_ids.add(turn_id)
                text = turn["text"]
                speaker = turn["speaker"]
                if not isinstance(text, str) or not isinstance(speaker, str):
                    raise ValueError(f"Invalid text or speaker in {scope}:{turn_id}")
                parts = [f"Session timestamp: {timestamp}", f"Speaker: {speaker}", f"Text: {text}"]
                caption = turn.get("blip_caption")
                if include_captions and caption:
                    if not isinstance(caption, str):
                        raise ValueError(f"Invalid image caption in {scope}:{turn_id}")
                    parts.append(f"Image caption: {caption}")
                    caption_count += 1
                documents.append({"id": f"{scope}:{turn_id}", "scope": scope, "text": "\n".join(parts)})
                counts["image_turns"] += bool(turn.get("img_url"))
        counts["captions_included"] += caption_count
        eligible_in_scope = 0
        for index, question in enumerate(sample["qa"]):
            query_id = f"{scope}:q{index:04d}"
            category = question["category"]
            if category not in CATEGORIES:
                raise ValueError(f"Unknown question category: {category}")
            source_categories[CATEGORIES[category]] += 1
            entry = {"id": query_id, "sample_id": sample_id,
                     "source_question_index": index, "category": category}
            evidence = question.get("evidence", [])
            if category == 5:
                entry["reason"] = "adversarial_requires_answer_abstention"
            elif not evidence:
                entry["reason"] = "missing_gold_evidence"
            elif not isinstance(evidence, list) or not all(isinstance(item, str) for item in evidence):
                entry["reason"] = "invalid_gold_format"
            else:
                missing = [item for item in evidence if item not in turn_ids]
                if missing:
                    entry["reason"] = "invalid_gold_reference"
                    entry["unresolved_evidence"] = missing
            if "reason" in entry:
                exclusions.append(entry)
                counts[entry["reason"]] += 1
                continue
            gold = list(dict.fromkeys(evidence))
            question_text = question["question"]
            if not isinstance(question_text, str) or not question_text.strip():
                raise ValueError(f"Empty question text: {query_id}")
            queries.append({"id": query_id, "query": question_text, "scope": scope,
                            "expected_ids": [f"{scope}:{item}" for item in gold]})
            metadata[query_id] = {"category": category, "category_name": CATEGORIES[category],
                                  "sample_id": sample_id, "source_question_index": index,
                                  "source_question_index_base": 0, "gold_count": len(gold)}
            eligible_categories[CATEGORIES[category]] += 1
            eligible_in_scope += 1
            max_gold = max(max_gold, len(gold))
        scope_stats[scope] = {"documents": len(turn_ids), "sessions": len(session_keys),
                              "source_questions": len(sample["qa"]), "eligible_questions": eligible_in_scope,
                              "captions_included": caption_count}

    summary = {
        "conversations": len(samples), "scopes": len(scopes), "documents": len(documents),
        "source_questions": sum(source_categories.values()), "eligible_questions": len(queries),
        "excluded_questions": len(exclusions), "max_gold_count": max_gold,
        "source_category_counts": dict(sorted(source_categories.items())),
        "eligible_category_counts": dict(sorted(eligible_categories.items())),
        "exclusion_counts": dict(sorted(Counter(row["reason"] for row in exclusions).items())),
        "sessions": counts["sessions"], "image_turns": counts["image_turns"],
        "captions_included": counts["captions_included"], "scope_stats": scope_stats,
    }
    assert summary["source_questions"] == len(queries) + len(exclusions)
    return {"documents": documents, "queries": queries, "metadata": metadata,
            "exclusions": exclusions, "summary": summary}


def prepare(root: Path, *, include_captions: bool = True) -> dict[str, Any]:
    root = root.resolve()
    raw = root / "data/raw"
    converted = root / "data/converted"
    data_path = raw / "locomo10.json"
    license_path = raw / "locomo-LICENSE.txt"
    download_verified(DATA_URL, data_path, DATA_SHA256, 32_000_000)
    download_verified(LICENSE_URL, license_path, LICENSE_SHA256, 100_000)
    if not license_path.read_text(encoding="utf-8").startswith("Attribution-NonCommercial 4.0 International"):
        raise ValueError("Unexpected upstream LoCoMo license")
    samples = json.loads(data_path.read_text(encoding="utf-8"))
    result = convert_locomo(samples, include_captions=include_captions)
    names = {
        "documents": "locomo.documents.jsonl", "queries": "locomo.queries.jsonl",
        "metadata": "locomo.query_metadata.json", "exclusions": "locomo.exclusions.json",
    }
    _write_jsonl(converted / names["documents"], result["documents"])
    _write_jsonl(converted / names["queries"], result["queries"])
    _write_json(converted / names["metadata"], result["metadata"])
    _write_json(converted / names["exclusions"], {
        "counts": result["summary"]["exclusion_counts"], "excluded": result["exclusions"],
    })
    manifest = {
        "schema_version": 1, "dataset": "LoCoMo-10", "task": "annotated_turn_evidence_retrieval",
        "official_qa_score": False, "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": {"repository": "https://github.com/snap-research/locomo", "revision": REVISION,
                   "url": DATA_URL, "sha256": sha256(data_path), "bytes": data_path.stat().st_size,
                   "license": "CC-BY-NC-4.0", "license_url": LICENSE_URL,
                   "license_sha256": sha256(license_path),
                   "paper": "https://aclanthology.org/2024.acl-long.747/"},
        "index_fields": ["session_date_time", "speaker", "text"] + (["blip_caption"] if include_captions else []),
        "image_pixels_included": False, "captions_included": include_captions,
        "caption_policy": "Append every nonempty upstream blip_caption, even if img_url is absent." if include_captions else "Exclude all image captions.",
        "excluded_index_fields": ["qa", "answer", "evidence", "observation", "session_summary", "event_summary"],
        "gold_policy": "Exclude category 5, missing evidence, or any unresolved raw evidence ID; deduplicate exact IDs only; no repairs.",
        "evaluation_policy": "Fixed full eligible set, no tuning or training; question and evidence labels are separate from indexed documents.",
        "category_5_policy": "Excluded from retrieval recall; unanswerable questions may have related evidence, so retrieved content is not required to be empty.",
        "data_origin": "Public upstream LLM-generated conversations with human verification and editing, not production conversation logs.",
        "summary": result["summary"],
        "files": {key: {"path": f"data/converted/{name}", "sha256": sha256(converted / name),
                         "bytes": (converted / name).stat().st_size} for key, name in names.items()},
    }
    _write_json(converted / "locomo.manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--no-captions", action="store_true", help="Explicit text-only variant; default includes provided BLIP captions.")
    args = parser.parse_args()
    manifest = prepare(args.root, include_captions=not args.no_captions)
    print(json.dumps({"dataset": manifest["dataset"], "sha256": manifest["source"]["sha256"],
                      "summary": manifest["summary"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
