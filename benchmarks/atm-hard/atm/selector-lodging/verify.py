"""Independently audit the saved Hard leaderboard gate and evidence budgets."""
import argparse
from collections import defaultdict
from fractions import Fraction
import hashlib
import json
from pathlib import Path
from statistics import mean


def keyed(path):
    with path.open() as file:
        rows = list(map(json.loads, file))
    by_id = {row["id"]: row for row in rows}
    assert len(by_id) == len(rows)
    return by_id


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "baseline", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    ranks = keyed(args.output / "rankings.jsonl")
    packets = keyed(args.output / "reader-packets.jsonl")
    baseline = keyed(args.baseline)
    questions = json.loads((args.data / "atm-bench-hard.json").read_text())
    summary = json.loads((args.output / "summary.json").read_text())
    assert len(questions) == len(ranks) == len(packets) == 31
    assert {q["id"] for q in questions} == ranks.keys() == packets.keys() == baseline.keys()
    assert sum(len(q["evidence_ids"]) for q in questions) == 194
    for file, digest in summary["data_sha256"].items():
        assert hashlib.sha256((args.data / file).read_bytes()).hexdigest() == digest

    original = set()
    for kind, filename in (("email", "emails"), ("image", "image_batch_results"),
                           ("video", "video_batch_results")):
        for record in json.loads((args.data / (filename + ".json")).read_bytes()):
            original.add(record["id"] if kind == "email" else Path(record[kind + "_path"]).stem)
    assert len(original) == 11034
    audited = {}
    changes = []
    for arm, metrics in summary["metrics"].items():
        recalls, reciprocal = [], []
        by_type = defaultdict(list)
        max_bytes = hits = 0
        for q in questions:
            key = q["id"]
            row, packet = ranks[key], packets[key]
            assert row["reference"] == baseline[key]["day_only"]
            assert packet["question"] == q["question"]
            ranking = row[arm]
            docs = packet["evidence"][arm]
            assert ranking == [doc["id"] for doc in docs]
            assert len(ranking) == len(set(ranking)) == 10 and set(ranking) <= original
            size = sum(len(doc["text"].encode("utf-8")) for doc in docs)
            assert size <= summary["budget"] == 32768
            max_bytes = max(size, max_bytes)
            gold = set(q["evidence_ids"])
            n = len(gold.intersection(ranking))
            hits += n
            recall = Fraction(n, len(gold))
            recalls.append(recall)
            by_type[q["qtype"]].append(recall)
            reciprocal.append(next((Fraction(1, rank) for rank, doc in enumerate(ranking, 1)
                                    if doc in gold), Fraction(0)))
            if arm == "combined_arrival" and ranking != row["reference"]:
                old, new = set(row["reference"]), set(ranking)
                changes.append(dict(id=key, question=q["question"], added=sorted(new-old),
                                    removed=sorted(old-new), added_gold=len((new-old)&gold),
                                    removed_gold=len((old-new)&gold)))
        macro = mean(recalls)
        assert abs(float(macro)-metrics["recall_at_10"]) < 1e-12
        assert abs(float(mean(reciprocal))-metrics["mrr_at_10"]) < 1e-12
        assert hits == metrics["gold_hits"] and recalls.count(1) == metrics["all_hit"]
        assert recalls.count(0) == metrics["zero_hit"]
        for group, recalls_in_group in by_type.items():
            assert abs(float(mean(recalls_in_group))-metrics["by_qtype"][group]) < 1e-12
        audited[arm] = dict(macro_recall_at_10=float(macro), exact_fraction=str(macro),
                            total_gold_hits=hits, total_gold=194, max_evidence_bytes=max_bytes)
    assert Fraction(audited["combined_arrival"]["exact_fraction"]) > Fraction(4, 5)
    assert len(changes)==4 and sum(x["added_gold"]-x["removed_gold"] for x in changes)==4
    print(json.dumps(dict(questions=31, original_sources=len(original),
                          budget=summary["budget"], results=audited,
                          final_changes=changes), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
