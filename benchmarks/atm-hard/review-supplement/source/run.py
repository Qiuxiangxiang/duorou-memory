"""Run duorou v1.0.0 retrieval on the two ATM-Bench QA splits."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import statistics
import sys
import tempfile
from time import perf_counter


def load_inputs(data):
    manifest = {}

    def load(name):
        file = data / f"{name}.json"
        raw = file.read_bytes()
        manifest[file.name] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)

    records = {}
    sources = {}
    for source, name, id_field, fields in (
        ("email", "emails", "id", ("timestamp", "short_summary", "detail")),
        ("image", "image_batch_results", "image_path",
         ("timestamp", "location_name", "city", "caption", "short_caption", "ocr_text", "tags", "entities")),
        ("video", "video_batch_results", "video_path",
         ("timestamp", "location_name", "city", "caption", "short_caption", "ocr_text", "tags", "entities", "duration_formatted")),
    ):
        for row in load(name):
            key = row[id_field] if source == "email" else Path(row[id_field]).stem
            if key in records:
                raise ValueError(f"duplicate evidence ID: {key}")
            lines = [f"Source: {source}"]
            for field in fields:
                value = row.get(field)
                if value:
                    formatted = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
                    lines.append(f"{field}: {formatted}")
            records[key] = dict(source_id=key, text="\n".join(lines), source_system="atm-bench")
            sources[key] = source

    splits = {split: load(name) for split, name in (("Main", "atm-bench"), ("Hard", "atm-bench-hard"))}
    assert Counter(sources.values()) == {"email": 6742, "image": 3759, "video": 533}
    assert {split: len(rows) for split, rows in splits.items()} == {"Main": 1013, "Hard": 31}
    return records, sources, splits, manifest


def run(release, data, output):
    sys.path.insert(0, str(release.resolve()))
    from duorou import Memory

    output.mkdir(parents=True, exist_ok=True)
    records, sources, splits, manifest = load_inputs(data)
    budget = 32768
    groups = defaultdict(list)
    overall_rows = []
    with tempfile.TemporaryDirectory() as temp:
        memory = Memory(Path(temp) / "atm.sqlite", "bench")
        try:
            memory.provision("atm")
            started = perf_counter()
            memory.ingest_many("atm", records.values())
            ingest_seconds = perf_counter() - started
            started = perf_counter()
            memory.search("atm", "memory", top_k=10, budget=budget)
            first_search_seconds = perf_counter() - started
            for split, questions in splits.items():
                for question in questions:
                    gold = set(question["evidence_ids"])
                    if not gold or gold - sources.keys():
                        raise ValueError(f"incomplete evidence mapping for {question['id']}")
                    since = perf_counter()
                    hits = memory.search("atm", question["question"], top_k=10, budget=budget)
                    elapsed = (perf_counter() - since) * 1000
                    ranked = [hit["id"] for hit in hits]
                    size = sum(len(hit["text"].encode("utf-8")) for hit in hits)
                    assert len(ranked) == len(set(ranked)) <= 10 and size <= budget
                    found = len(gold.intersection(ranked))
                    kinds = "+".join(sorted({sources[key] for key in gold}))
                    row = dict(split=split, id=question["id"], qtype=question["qtype"],
                               gold=sorted(gold), ranked=ranked, source_types=kinds, recall=found/len(gold),
                               any_hit=bool(found), all_hit=found == len(gold),
                               reciprocal_rank=next((1/i for i, key in enumerate(ranked, 1) if key in gold), 0),
                               bytes=size, ms=elapsed)
                    overall_rows.append(row)
                    for label in ("overall", f"qtype:{row['qtype']}", f"sources:{kinds}",
                                  f"modality_count:{len(kinds.split('+'))}",
                                  f"evidence_count:{'multiple' if len(gold)>1 else 'one'}"):
                        groups[split, label].append(row)
        finally:
            memory.close()

    with (output / "rankings.jsonl").open("w") as handle:
        for row in overall_rows:
            handle.write(json.dumps(row) + "\n")
    with (output / "metrics.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("split", "group", "n", "recall@10", "any_hit@10", "all_hit@10", "mrr@10", "median_ms"))
        for (split, label), rows in groups.items():
            writer.writerow([split, label, len(rows), *(
                statistics.mean(row[key] for row in rows)
                for key in ("recall", "any_hit", "all_hit", "reciprocal_rank")),
                statistics.median(row["ms"] for row in rows)])
    (output / "manifest.json").write_text(json.dumps(
        dict(revision="78e826dc07e97466b2f54443831ef9a83ab8b27c", file_sha256=manifest,
             index=Counter(sources.values()), budget_utf8_bytes=budget, k=10,
             ingest_seconds=ingest_seconds, first_search_seconds=first_search_seconds,
             mode="default hybrid without embedding"), indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("release", type=Path)
    parser.add_argument("data", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    run(args.release, args.data, args.output)
