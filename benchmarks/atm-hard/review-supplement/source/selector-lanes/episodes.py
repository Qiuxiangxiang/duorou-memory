"""Require new visit-count evidence to establish a distinct place-and-time episode."""
import argparse
from datetime import datetime, timedelta
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "p0-full")]
from evaluate import BUDGET, read_module
from run import load_inputs

VISIT_COUNT = re.compile(r"\bhow many (?:times|visits|trips)\b", re.I)
VISIT = re.compile(r"\b(?:visit\w*|trip\w*|been to|went to)\b", re.I)


def observation(row):
    if not row or row["recorded_time_kind"] != "capture metadata":
        return None
    when, place = row["recorded_time"], row["recorded_place"]
    if not when or not place:
        return None
    try:
        return datetime.fromisoformat(when).date(), place.casefold()
    except ValueError:
        return None


def guard(question, previous, ranks, captures, proposals):
    if not (VISIT_COUNT.search(question) and VISIT.search(question)):
        return ranks, []
    prior_visits = [captures[key] for key in previous if captures.get(key)]
    retained = set(previous)
    kept = []
    rejected = []
    for key in ranks:
        if key in retained:
            kept.append(key)
            continue
        candidate = captures.get(key)
        roles = proposals.get(key, [])
        direct = any(proof["strength"] == 3 for proof in roles)
        distinct = candidate is not None and any(place == candidate[1] for _,place in prior_visits) and all(
            place != candidate[1] or abs(when-candidate[0]) > timedelta(days=7)
            for when,place in prior_visits)
        if direct and distinct:
            kept.append(key)
        else:
            rejected.append(key)
    if not rejected:
        return ranks, []
    return list(dict.fromkeys([*kept, *previous]))[:10], rejected


def main(args):
    sys.path.insert(0, str(args.official.resolve()))
    from memqa.utils.evaluator.evaluate_retrieval import comprehensive_eval

    helpful = read_module("episodes_files", ROOT / "selector-panels/run.py")
    metrics = read_module("episodes_metrics", ROOT / "selector-corpus/run.py")
    records, _, splits, hashes = load_inputs(args.data)
    assert hashes == read_module("episodes_pins", ROOT / "iterative-search/search.py").PUBLIC_SHA256
    baseline = helpful.keyed(args.lanes / "rankings.jsonl")
    packets = helpful.keyed(args.lanes / "reader-packets.jsonl")
    original = helpful.keyed(ROOT / "selector-contribution/fusion-reader-packets.jsonl")
    captures = {row["id"]: observation(row) for row in helpful.rows(ROOT / "event-hierarchy/source-tags.jsonl")}
    traces = []
    output = []
    for q in splits["Hard"]:
        key, query = q["id"], q["question"]
        row = baseline[key]
        old, lane_order = (row[name] for name in ("previous", "covered"))
        proposals = {item["id"]: item["facts"] for item in row["decisions"]["covered"]}
        result, refused = guard(query, old, lane_order, captures, proposals)
        assert len(result) == len(set(result)) == 10 and set(result) <= records.keys()
        docs = {item["id"]: item for item in original[key]["evidence"]["ordering"]}
        docs.update({item["id"]: item for item in packets[key]["evidence"]["covered"]})
        evidence = [docs[source] for source in result]
        assert sum(len(item["text"].encode()) for item in evidence)<=BUDGET
        traces.append(dict(id=key, previous=old, lanes=lane_order, episode_guard=result,
                           refused_same_episode_or_unproven=refused))
        output.append(dict(id=key, question=query, evidence=evidence))
    args.output.mkdir(parents=True, exist_ok=True)
    for name, rows in (("episode-rankings", traces), ("episode-reader-packets", output)):
        (args.output / f"{name}.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    stat = metrics.measure(splits["Hard"], traces, ("previous", "lanes", "episode_guard"), comprehensive_eval)
    report = dict(n=len(traces), data_sha256=hashes, budget=BUDGET, api_tokens=0,
                  adjusted_questions=sum(bool(row["refused_same_episode_or_unproven"]) for row in traces),
                  metrics=stat)
    (args.output / "episode-summary.json").write_text(json.dumps(report,indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ("data", "official", "lanes", "output"):
        p.add_argument(arg,type=Path)
    main(p.parse_args())
